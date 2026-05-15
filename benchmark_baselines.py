import torch
import time
import pandas as pd
import os
from tqdm import tqdm

from baseline.pencil_schur import joint_triangularize as schur_jt
from baseline.jacobi_type import joint_triangularize as jacobi_jt
from baseline.optim_newton import joint_triangularize as newton_jt

BASELINES = {
    "Schur": schur_jt,
    "Jacobi": jacobi_jt,
    "Newton": newton_jt
}

def compute_rel_residual(A, B, Q):
    """
    Вычисляет относительный нижнетреугольный residual после применения преобразования Q.
    """
    A_prime = Q.T @ A @ Q
    B_prime = Q.T @ B @ Q
    
    def tril_sq_sum(M):
        # Сумма квадратов элементов строго ниже главной диагонали
        return torch.sum(torch.tril(M, diagonal=-1)**2)
        
    num = tril_sq_sum(A_prime) + tril_sq_sum(B_prime)
    den = torch.sum(A**2) + torch.sum(B**2) + 1e-12
    return (num / den).item()

def run_benchmark():
    dataset_path = "dataset/dataset.pt"
    if not os.path.exists(dataset_path):
        print(f"Dataset not found at {dataset_path}. Please generate it first.")
        return

    dataset = torch.load(dataset_path)
    print(f"Loaded {len(dataset)} samples. Starting benchmark...")
    
    results = []
    
    for sample in tqdm(dataset, desc="Benchmarking"):
        n = sample["n"]
        mtype = sample["type"]
        A = sample["A"]
        B = sample["B"]
        
        for alg_name, alg_fn in BASELINES.items():
            try:
                # Измерение времени (1 прогон без warmup для максимальной скорости бенчмарка, 
                # так как матрицы могут быть большими и итеративные методы работают долго)
                start_time = time.perf_counter()
                Q = alg_fn(A, B)
                end_time = time.perf_counter()
                
                elapsed = end_time - start_time
                residual = compute_rel_residual(A, B, Q)
                
                results.append({
                    "Algorithm": alg_name,
                    "Type": mtype,
                    "n": n,
                    "Time (s)": elapsed,
                    "Rel_Residual": residual
                })
            except Exception as e:
                # В случае краша алгоритма (например расходится)
                results.append({
                    "Algorithm": alg_name,
                    "Type": mtype,
                    "n": n,
                    "Time (s)": float('nan'),
                    "Rel_Residual": float('nan')
                })
                
    df = pd.DataFrame(results)
    
    # Агрегируем результаты - среднее по размерам и типам 
    agg_df = df.groupby(["Algorithm", "Type", "n"]).mean().reset_index()
    
    # Сортируем для красивого вывода
    agg_df = agg_df.sort_values(by=["Type", "n", "Algorithm"])
    
    print("\n# Результаты бенчмарка бейзлайнов")
    print("Сравнение алгоритмов по метрикам относительного нижнетреугольного residual'а и времени выполнения.\n")
    
    try:
        print(agg_df.to_markdown(index=False, floatfmt=".6f"))
    except ImportError:
        print("\nБиблиотека tabulate не установлена. Выводим без форматирования markdown:")
        print(agg_df.to_string(index=False))

if __name__ == "__main__":
    run_benchmark()
