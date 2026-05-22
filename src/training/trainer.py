"""Training loop with wandb logging and best-checkpoint tracking."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import wandb
from torch.utils.data import DataLoader

from src.evaluation.metrics import evaluate_transform
from src.training.losses import total_loss


def _aggregate(metrics_list: list[dict[str, float]]) -> dict[str, float]:
    if not metrics_list:
        return {}
    keys = metrics_list[0].keys()
    return {k: float(sum(m[k] for m in metrics_list) / len(metrics_list)) for k in keys}


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    lambda_orth: float,
    grad_clip: float,
    device: torch.device,
) -> dict[str, float]:
    model.train()
    metrics: list[dict[str, float]] = []
    for batch in loader:
        A = batch["A"].to(device)
        B = batch["B"].to(device)

        optimizer.zero_grad()
        T = model(A, B)
        loss, components = total_loss(T, A, B, lambda_orth=lambda_orth)
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()

        metrics.append({"loss_total": float(loss.detach()), **components})
    return _aggregate(metrics)


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    loader: DataLoader,
    lambda_orth: float,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    metrics: list[dict[str, float]] = []
    for batch in loader:
        A = batch["A"].to(device)
        B = batch["B"].to(device)
        T = model(A, B)
        loss, components = total_loss(T, A, B, lambda_orth=lambda_orth)
        m = {"loss_total": float(loss.detach()), **components}
        # Per-sample geometric metrics on the first item of the batch.
        m.update(evaluate_transform(T[0], A[0], B[0]))
        metrics.append(m)
    return _aggregate(metrics)


def train(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: dict[str, Any],
    device: torch.device,
    checkpoint_dir: Path,
) -> None:
    tcfg = config["training"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=tcfg["lr"],
        weight_decay=tcfg.get("weight_decay", 0.0),
    )
    scheduler = None
    if tcfg.get("scheduler") == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=tcfg["epochs"]
        )

    lambda_orth = tcfg["lambda_orth"]
    grad_clip = tcfg.get("grad_clip", 0.0)

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_path = checkpoint_dir / "best_model.pt"
    last_path = checkpoint_dir / "last_model.pt"

    best_val = float("inf")
    for epoch in range(tcfg["epochs"]):
        train_m = train_one_epoch(model, train_loader, optimizer, lambda_orth, grad_clip, device)
        val_m = validate(model, val_loader, lambda_orth, device)
        if scheduler is not None:
            scheduler.step()

        log = {f"train/{k}": v for k, v in train_m.items()}
        log.update({f"val/{k}": v for k, v in val_m.items()})
        log["epoch"] = epoch
        log["lr"] = optimizer.param_groups[0]["lr"]
        wandb.log(log)

        torch.save({"model": model.state_dict(), "config": config, "epoch": epoch}, last_path)
        if val_m["loss_total"] < best_val:
            best_val = val_m["loss_total"]
            torch.save(
                {"model": model.state_dict(), "config": config, "epoch": epoch, "val": val_m},
                best_path,
            )
            wandb.run.summary["best_val_loss"] = best_val
            wandb.run.summary["best_epoch"] = epoch

        print(
            f"epoch {epoch:4d} | "
            f"train loss {train_m['loss_total']:.6f} | "
            f"val loss {val_m['loss_total']:.6f} | "
            f"val lower_A {val_m['lower_ratio_A']:.4f} | "
            f"val lower_B {val_m['lower_ratio_B']:.4f}"
        )

    wandb.save(str(best_path), policy="now")
