import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

try:
	from model.model import TriangularizerModel
except ModuleNotFoundError:
	from model import TriangularizerModel


class MatrixPairDataset(Dataset):
	def __init__(self, a: np.ndarray, b: np.ndarray, y: np.ndarray):
		self.a = torch.tensor(a, dtype=torch.float64)
		self.b = torch.tensor(b, dtype=torch.float64)
		self.y = torch.tensor(y, dtype=torch.float64)

	def __len__(self) -> int:
		return self.a.shape[0]

	def __getitem__(self, idx: int):
		return self.a[idx], self.b[idx], self.y[idx]


@dataclass
class TrainConfig:
	dataset_path: str = "dataset/dataset.npz"
	weights_path: str = "model_weights.pt"
	epochs: int = 40
	batch_size: int = 512
	lr: float = 1e-3
	weight_decay: float = 1e-5
	val_ratio: float = 0.1
	seed: int = 42
	positive_weight: float = 1.25
	negative_weight: float = 1.0
	cond_reg_weight: float = 1e-4
	grad_clip: float = 2.0
	patience: int = 8


def calc_lower_diag_metric_batch(m: torch.Tensor) -> torch.Tensor:
	lower = torch.tril(m, diagonal=-1)
	return torch.abs(lower).sum(dim=(1, 2))


def make_transformed_scores(model: nn.Module, a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
	t = model(a, b)
	t_inv = torch.linalg.inv(t)

	a_trans = t_inv @ a @ t
	b_trans = t_inv @ b @ t

	score_a = calc_lower_diag_metric_batch(a_trans)
	score_b = calc_lower_diag_metric_batch(b_trans)
	return score_a + score_b, t, t_inv


def compute_loss(scores: torch.Tensor, y: torch.Tensor, t: torch.Tensor, t_inv: torch.Tensor, cfg: TrainConfig) -> torch.Tensor:
	# Главная цель: улучшить именно метрику evaluate.py (mean(scores) по всему датасету).
	pos_mask = y > 0.5
	neg_mask = ~pos_mask

	global_loss = scores.mean()

	pos_loss = scores[pos_mask].mean() if pos_mask.any() else scores.new_tensor(0.0)
	neg_loss = scores[neg_mask].mean() if neg_mask.any() else scores.new_tensor(0.0)

	# Регуляризация на кондиционирование T для более стабильной обратимости и меньшего разлета ошибок.
	cond_reg = (t.pow(2).mean() + t_inv.pow(2).mean())

	return (
		global_loss
		+ cfg.positive_weight * pos_loss
		+ cfg.negative_weight * neg_loss
		+ cfg.cond_reg_weight * cond_reg
	)


def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, cfg: TrainConfig) -> float:
	model.train()
	total = 0.0
	for a, b, y in tqdm(loader, desc="train", leave=False):
		optimizer.zero_grad(set_to_none=True)
		scores, t, t_inv = make_transformed_scores(model, a, b)
		loss = compute_loss(scores, y, t, t_inv, cfg)
		loss.backward()
		torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
		optimizer.step()
		total += float(loss.item()) * a.shape[0]
	return total / len(loader.dataset)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, cfg: TrainConfig) -> tuple[float, float, float]:
	model.eval()
	total_loss = 0.0
	total_score = 0.0
	total_pos_score = 0.0
	total_pos_count = 0.0

	for a, b, y in tqdm(loader, desc="valid", leave=False):
		scores, t, t_inv = make_transformed_scores(model, a, b)
		loss = compute_loss(scores, y, t, t_inv, cfg)

		total_loss += float(loss.item()) * a.shape[0]
		total_score += float(scores.sum().item())
		pos_mask = y > 0.5
		if pos_mask.any():
			total_pos_score += float(scores[pos_mask].sum().item())
			total_pos_count += float(pos_mask.sum().item())

	mean_loss = total_loss / len(loader.dataset)
	mean_score = total_score / len(loader.dataset)
	mean_pos_score = total_pos_score / max(total_pos_count, 1.0)
	return mean_loss, mean_score, mean_pos_score


def main() -> None:
	parser = argparse.ArgumentParser(description="Train deep model for matrix joint triangulation")
	parser.add_argument("--dataset", type=str, default="dataset/dataset.npz")
	parser.add_argument("--weights", type=str, default="model_weights.pt")
	parser.add_argument("--epochs", type=int, default=40)
	parser.add_argument("--batch-size", type=int, default=512)
	parser.add_argument("--lr", type=float, default=1e-3)
	parser.add_argument("--seed", type=int, default=42)
	args = parser.parse_args()

	cfg = TrainConfig(
		dataset_path=args.dataset,
		weights_path=args.weights,
		epochs=args.epochs,
		batch_size=args.batch_size,
		lr=args.lr,
		seed=args.seed,
	)

	dataset_path = Path(cfg.dataset_path)
	if not dataset_path.exists():
		raise FileNotFoundError(f"Dataset not found: {dataset_path}")

	np.random.seed(cfg.seed)
	torch.manual_seed(cfg.seed)

	data = np.load(dataset_path)
	a_data, b_data, y_data = data["A"], data["B"], data["y"]

	n_samples, n_size, _ = a_data.shape
	dataset = MatrixPairDataset(a_data, b_data, y_data)

	n_val = max(1, int(n_samples * cfg.val_ratio))
	n_train = n_samples - n_val
	gen = torch.Generator().manual_seed(cfg.seed)
	train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=gen)

	train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
	val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)

	model = TriangularizerModel(n=n_size)
	optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
	scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)

	print(f"Train samples: {n_train}, Val samples: {n_val}, Matrix size: {n_size}")

	best_val = float("inf")
	best_epoch = -1
	wait = 0

	for epoch in range(1, cfg.epochs + 1):
		train_loss = train_one_epoch(model, train_loader, optimizer, cfg)
		val_loss, val_score, val_pos_score = evaluate(model, val_loader, cfg)
		scheduler.step()

		print(
			f"[Epoch {epoch:03d}] "
			f"train_loss={train_loss:.6f} | "
			f"val_loss={val_loss:.6f} | "
			f"val_general_score={val_score:.6f} | "
			f"val_triang_score={val_pos_score:.6f}"
		)

		if val_loss < best_val:
			best_val = val_loss
			best_epoch = epoch
			wait = 0
			torch.save(model.state_dict(), cfg.weights_path)
		else:
			wait += 1
			if wait >= cfg.patience:
				print(f"Early stopping at epoch {epoch} (best epoch: {best_epoch})")
				break

	print(f"Best model saved to: {cfg.weights_path} (epoch {best_epoch}, val_loss={best_val:.6f})")


if __name__ == "__main__":
	main()
