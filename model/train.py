import torch
import torch.optim as optim
import time
from tqdm import tqdm  # Добавили импорт tqdm

# Импортируем нашу модель из соседнего файла
try:
    from model import TriangularizerModel
except ImportError:
    print("❌ ОШИБКА: Не найден файл model.py. Убедитесь, что он лежит в этой же папке.")
    exit(1)


def generate_training_batch(batch_size: int, n: int = 3, device: str = 'cpu') -> tuple[torch.Tensor, torch.Tensor]:
    """
    Генерирует батч матриц для обучения (во float64).
    Смешиваем идеальные совместно триангуляризуемые матрицы и матрицы с легким шумом,
    чтобы нейросеть училась обобщать.
    """
    A_list, B_list = [], []

    for _ in range(batch_size):
        # 1. Создаем идеальные верхнетреугольные матрицы
        T_A = torch.triu(torch.randn(n, n, dtype=torch.float64, device=device))
        T_B = torch.triu(torch.randn(n, n, dtype=torch.float64, device=device))

        # 2. Генерируем случайную ортогональную матрицу Q
        H = torch.randn(n, n, dtype=torch.float64, device=device)
        Q, _ = torch.linalg.qr(H)

        # 3. Прячем треугольную структуру
        A = Q @ T_A @ Q.T
        B = Q @ T_B @ Q.T

        # 4. В 50% случаев добавляем небольшой математический шум
        if torch.rand(1).item() > 0.5:
            noise_level = 0.05
            A += torch.randn(n, n, dtype=torch.float64, device=device) * noise_level
            B += torch.randn(n, n, dtype=torch.float64, device=device) * noise_level

        A_list.append(A)
        B_list.append(B)

    return torch.stack(A_list), torch.stack(B_list)


def compute_loss(A_trans: torch.Tensor, B_trans: torch.Tensor) -> torch.Tensor:
    """
    Функция потерь (Loss).
    Штрафует модель за любые ненулевые элементы СТРОГО ПОД главной диагональю.
    """
    n = A_trans.shape[-1]
    # Создаем маску, где 1 стоят только под диагональю
    mask = torch.tril(torch.ones(n, n, device=A_trans.device, dtype=torch.float64), diagonal=-1)

    # Считаем сумму квадратов элементов под диагональю (MSE)
    loss_A = torch.sum((A_trans * mask) ** 2)
    loss_B = torch.sum((B_trans * mask) ** 2)

    # Усредняем по размеру батча
    batch_size = A_trans.shape[0]
    return (loss_A + loss_B) / batch_size


def train():
    # 1. Настройки обучения
    n_size = 3
    batch_size = 128  # Размер порции данных за один шаг
    iterations = 15000  # Количество шагов градиентного спуска
    learning_rate = 1e-3  # Скорость обучения

    # Выбираем видеокарту, если она есть, иначе процессор
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Запуск обучения на устройстве: {device}")
    print("Используем точность: Double Precision (float64)\n")

    # 2. Инициализация модели и оптимизатора
    model = TriangularizerModel(n=n_size).to(device)
    # AdamW работает стабильнее и лучше предотвращает переобучение
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    # Планировщик: будет плавно уменьшать learning_rate к концу обучения
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iterations)

    model.train()
    start_time = time.time()

    # 3. Главный цикл обучения с использованием tqdm
    pbar = tqdm(range(1, iterations + 1), desc="Обучение модели")
    for i in pbar:
        optimizer.zero_grad()

        # Генерируем свежие данные
        A_batch, B_batch = generate_training_batch(batch_size, n=n_size, device=device)

        # Прогоняем через модель
        T_pred = model(A_batch, B_batch)

        # ВАЖНО: Так как наша архитектура гарантирует, что T - ортогональная,
        # обратная матрица T^-1 - это просто транспонированная T^T.
        # Это очень быстро и абсолютно стабильно для градиентов!
        T_inv = T_pred.transpose(-2, -1)

        # Применяем преобразование подобия
        A_trans = T_inv @ A_batch @ T_pred
        B_trans = T_inv @ B_batch @ T_pred

        # Считаем ошибку и обновляем веса
        loss = compute_loss(A_trans, B_trans)
        loss.backward()

        # Обрезка градиентов (защита от "взрыва" математики)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()

        # 4. Логирование прогресса в tqdm
        current_lr = scheduler.get_last_lr()[0]
        # Обновляем метрики в прогресс-баре каждую 10-ю итерацию (или каждую, но так чуть быстрее)
        if i % 10 == 0 or i == iterations:
            pbar.set_postfix({"Loss": f"{loss.item():.6f}", "LR": f"{current_lr:.6f}"})

    # 5. Сохранение результатов
    elapsed = time.time() - start_time
    print(f"\n✅ Обучение завершено за {elapsed:.1f} сек.")

    save_path = "model_weights.pt"
    # Сохраняем веса, предварительно перекинув их в оперативную память (cpu)
    torch.save(model.state_dict(), save_path)
    print(f"💾 Веса модели успешно сохранены в файл: {save_path}")
    print("Теперь вы можете сделать git commit и отправить код в GitHub Actions!")


if __name__ == "__main__":
    train()