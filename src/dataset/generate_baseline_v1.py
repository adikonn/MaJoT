import os

from dataset.generate_data import create_and_save_dataset


BASELINE_SIZES = [4, 8, 16]
BASELINE_SAMPLES_PER_CONFIG = 100
BASELINE_NOISE_LEVEL = 1e-3
BASELINE_SEED = 42


def generate_baseline_dataset_v1(save_path=None):
    if save_path is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        save_path = os.path.join(current_dir, "baseline_dataset_v1.pt")

    create_and_save_dataset(
        save_path=save_path,
        sizes=BASELINE_SIZES,
        samples_per_config=BASELINE_SAMPLES_PER_CONFIG,
        noise_level=BASELINE_NOISE_LEVEL,
        seed=BASELINE_SEED,
    )
    return save_path


if __name__ == "__main__":
    output_path = generate_baseline_dataset_v1()
    print(f"Baseline dataset v1 saved to {output_path}")
