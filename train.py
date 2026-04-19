import argparse
import json
import os
from typing import Dict, Tuple

import numpy as np
import torch
from tqdm import tqdm

from model.model import TriangularizerModel


def lower_triangular_penalty_batch(M: torch.Tensor) -> torch.Tensor:
    lower = torch.tril(M, diagonal=-1)
    return torch.abs(lower).sum(dim=(1, 2))


def _batch_loss(
    model: TriangularizerModel,
    A_batch: torch.Tensor,
    B_batch: torch.Tensor,
    y_batch: torch.Tensor,
    reg_identity: float,
    class_margin_weight: float,
    class_margin: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    T = model(A_batch, B_batch)
    T_t = torch.transpose(T, dim0=1, dim1=2)
    A_trans = T_t @ A_batch @ T
    B_trans = T_t @ B_batch @ T

    per_sample_tri = lower_triangular_penalty_batch(A_trans) + lower_triangular_penalty_batch(B_trans)
    tri_loss = per_sample_tri.mean()
    ident = torch.eye(T.shape[1], dtype=torch.float64, device=T.device).unsqueeze(0).expand(T.shape[0], -1, -1)
    reg = torch.mean((T - ident) ** 2)
    pos_mask = y_batch > 0.5
    neg_mask = ~pos_mask
    if pos_mask.any() and neg_mask.any():
        pos_mean = per_sample_tri[pos_mask].mean()
        neg_mean = per_sample_tri[neg_mask].mean()
        # Encourage positive pairs to have lower triangularization score than negatives.
        margin_loss = torch.relu(class_margin + pos_mean - neg_mean)
    else:
        margin_loss = torch.tensor(0.0, dtype=torch.float64, device=A_batch.device)
    total = tri_loss + reg_identity * reg + class_margin_weight * margin_loss
    return total, tri_loss, margin_loss


def _evaluate_epoch(
    model: TriangularizerModel,
    A_val: torch.Tensor,
    B_val: torch.Tensor,
    y_val: torch.Tensor,
    batch_size: int,
    reg_identity: float,
    class_margin_weight: float,
    class_margin: float,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_tri_loss = 0.0
    total_margin = 0.0
    total_count = 0
    with torch.no_grad():
        for start in range(0, A_val.shape[0], batch_size):
            A_batch = A_val[start:start + batch_size]
            B_batch = B_val[start:start + batch_size]
            y_batch = y_val[start:start + batch_size]
            loss, tri_loss, margin_loss = _batch_loss(
                model,
                A_batch,
                B_batch,
                y_batch,
                reg_identity,
                class_margin_weight,
                class_margin,
            )
            count = A_batch.shape[0]
            total_loss += float(loss.item()) * count
            total_tri_loss += float(tri_loss.item()) * count
            total_margin += float(margin_loss.item()) * count
            total_count += count
    return {
        "loss": total_loss / max(total_count, 1),
        "tri_loss": total_tri_loss / max(total_count, 1),
        "margin_loss": total_margin / max(total_count, 1),
    }


def train(
    dataset_path: str = "dataset/dataset.npz",
    weights_path: str = "model_weights.pt",
    history_path: str = "train_history.json",
    epochs: int = 120,
    batch_size: int = 64,
    lr: float = 2e-3,
    reg_identity: float = 1e-3,
    class_margin_weight: float = 0.25,
    class_margin: float = 0.25,
    val_ratio: float = 0.15,
    patience: int = 20,
    seed: int = 42,
) -> None:
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    np.random.seed(seed)
    torch.manual_seed(seed)

    data = np.load(dataset_path)
    A_np = data["A"]
    B_np = data["B"]
    y_np = data["y"].astype(np.float64)
    n_samples, n, _ = A_np.shape

    val_size = max(1, int(n_samples * val_ratio))
    perm = np.random.permutation(n_samples)
    val_idx = perm[:val_size]
    train_idx = perm[val_size:]
    if train_idx.size == 0:
        raise ValueError("Validation split is too large; no samples left for training.")

    A_train = torch.tensor(A_np[train_idx], dtype=torch.float64)
    B_train = torch.tensor(B_np[train_idx], dtype=torch.float64)
    y_train = torch.tensor(y_np[train_idx], dtype=torch.float64)
    A_val = torch.tensor(A_np[val_idx], dtype=torch.float64)
    B_val = torch.tensor(B_np[val_idx], dtype=torch.float64)
    y_val = torch.tensor(y_np[val_idx], dtype=torch.float64)

    model = TriangularizerModel(n=n).to(torch.float64)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val = float("inf")
    best_state = None
    bad_epochs = 0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        indices = torch.randperm(A_train.shape[0])
        epoch_loss = 0.0
        epoch_tri = 0.0
        epoch_margin = 0.0

        pbar = tqdm(range(0, A_train.shape[0], batch_size), desc=f"Epoch {epoch}/{epochs}", unit="batch")
        for start in pbar:
            idx = indices[start:start + batch_size]
            A_batch = A_train[idx]
            B_batch = B_train[idx]
            y_batch = y_train[idx]
            loss, tri_loss, margin_loss = _batch_loss(
                model,
                A_batch,
                B_batch,
                y_batch,
                reg_identity,
                class_margin_weight,
                class_margin,
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            count = A_batch.shape[0]
            epoch_loss += float(loss.item()) * count
            epoch_tri += float(tri_loss.item()) * count
            epoch_margin += float(margin_loss.item()) * count
            pbar.set_postfix(loss=f"{loss.item():.5f}", tri=f"{tri_loss.item():.5f}", margin=f"{margin_loss.item():.5f}")

        train_loss = epoch_loss / A_train.shape[0]
        train_tri_loss = epoch_tri / A_train.shape[0]
        train_margin_loss = epoch_margin / A_train.shape[0]
        val_metrics = _evaluate_epoch(
            model, A_val, B_val, y_val, batch_size, reg_identity, class_margin_weight, class_margin
        )
        val_loss = val_metrics["loss"]

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_tri_loss": train_tri_loss,
            "val_loss": val_loss,
            "val_tri_loss": val_metrics["tri_loss"],
            "train_margin_loss": train_margin_loss,
            "val_margin_loss": val_metrics["margin_loss"],
        }
        history.append(record)
        print(
            f"[train] epoch={epoch} train_loss={train_loss:.6f} train_tri={train_tri_loss:.6f} "
            f"train_margin={train_margin_loss:.6f} val_loss={val_loss:.6f} "
            f"val_tri={val_metrics['tri_loss']:.6f} val_margin={val_metrics['margin_loss']:.6f}"
        )

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"[train] early stopping at epoch={epoch}, patience={patience}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    torch.save(model.state_dict(), weights_path)
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=True, indent=2)
    print(f"[train] best_val_loss={best_val:.6f}")
    print(f"[train] weights saved to {weights_path}")
    print(f"[train] history saved to {history_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train neural simultaneous triangularization model.")
    parser.add_argument("--dataset", default="dataset/dataset.npz", help="Path to dataset NPZ file.")
    parser.add_argument("--weights", default="model_weights.pt", help="Output path for model weights.")
    parser.add_argument("--history", default="train_history.json", help="Output path for training history JSON.")
    parser.add_argument("--epochs", type=int, default=120, help="Max number of epochs.")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size.")
    parser.add_argument("--lr", type=float, default=2e-3, help="Learning rate.")
    parser.add_argument("--reg-identity", type=float, default=1e-3, help="Identity regularization coefficient.")
    parser.add_argument("--class-margin-weight", type=float, default=0.25, help="Weight of label-aware margin loss.")
    parser.add_argument("--class-margin", type=float, default=0.25, help="Desired separation margin between classes.")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation split ratio.")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        dataset_path=args.dataset,
        weights_path=args.weights,
        history_path=args.history,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        reg_identity=args.reg_identity,
        class_margin_weight=args.class_margin_weight,
        class_margin=args.class_margin,
        val_ratio=args.val_ratio,
        patience=args.patience,
        seed=args.seed,
    )
