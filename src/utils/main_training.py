import os
import pickle
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Callable

from .dataset import MyDataset
from .config_run import set_seed, set_device
from .metrics import calculate_tptnfpfn, calculate_all_tptnfpfn, calculate_metrics

project_path = os.getenv("project_root")


def main_training(
    model: nn.Module,
    dimension: int,
    model_name: str,
    dataset_name: str,
    parameters: dict,
    device: str = "cuda",
    bar: Callable[[str], None] = print,
):
    """Основная функция запуска обучения и тестирования модели.
    Включает в себя:
    * Загрузку датасета (из dataset.py)

    Args:
        model (nn.Module): модель для обучения
        dimension (int): размерность входных данных
        model_name (str): имя, под которым будет сохранена обученная модель
        dataset_name (str): имя датасета, из которого будут загружены данные
        params_nn (dict): параметры модели в виде словаря
        params_other (dict): гиперпараметры обучения в виде словаря (epochs, batch_size, learning_rate, l2_decay, optimizer)
        device (str, optional): устройство. Defaults to "cuda".
        bar (Callable[[str], None], optional): выводит прогресс обучения. Defaults to print.
    """

    # meta.keys() = (n_sig, fs, n_classes, labels, n_MI, n_STTC, n_CD, n_HYP, n_train_dev_test, n_train, n_dev, n_test)
    meta = Path(
        project_path, "data", "processed", f"{dimension}D", dataset_name, "meta.pkl"
    )
    with open(meta, "rb") as f:
        meta = pickle.load(f)

    # установка устройства, на котором запускается обучение
    set_device(device)
    model = model.to(device)
    # установка seed на все модули рандома
    set_seed(42)

    dataset_train = MyDataset(dataset_name, dimension, "train", meta["n_train"])
    dataset_dev = MyDataset(dataset_name, dimension, "dev", meta["n_dev"])
    dataset_test = MyDataset(dataset_name, dimension, "test", meta["n_test"])

    dataloader_train = DataLoader(
        dataset_train, batch_size=parameters["batch_size"], shuffle=True
    )
    dataloader_dev = DataLoader(dataset_dev, batch_size=1, shuffle=False)
    dataloader_test = DataLoader(dataset_test, batch_size=1, shuffle=False)

    optimizer = (
        torch.optim.Adam if parameters["optimizer"] == "adam" else torch.optim.SGD
    )
    optimizer = optimizer(
        model.parameters(),
        lr=parameters["learning_rate"],
        weight_decay=parameters["l2_decay"],
    )

    # https://discuss.pytorch.org/t/weighted-binary-cross-entropy/51156/6
    # https://discuss.pytorch.org/t/multi-label-multi-class-class-imbalance/37573/2
    labels_weights = torch.tensor(
        [
            meta["n_train_dev_test"] / meta["n_MI"],
            meta["n_train_dev_test"] / meta["n_STTC"],
            meta["n_train_dev_test"] / meta["n_CD"],
            meta["n_train_dev_test"] / meta["n_HYP"],
        ],
        dtype=torch.float,
    )
    loss_function = nn.BCEWithLogitsLoss(
        pos_weight=labels_weights
    )  # https://learnopencv.com/multi-label-image-classification-with-pytorch-image-tagging/
    # https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html

    # эпохи в виде массива, для дальнейшего plot
    epochs = torch.arange(1, parameters["epochs"] + 1)
    # потери за каждый батч
    train_losses = np.zeros(np.ceil(meta["n_train"] // parameters["batch_size"]))
    # Среднии потери за каждую эпоху
    train_mean_losses = np.zeros(parameters["epochs"])
    for epoch in epochs:
        bar(f"Epoch {epoch}")
        train_losses[:] = 0
        for i, (X, label) in enumerate(dataloader_train):
            loss = train_batch(X, label, model, optimizer, loss_function)

            del X
            del label
            torch.cuda.empty_cache()

            train_losses[i] = loss

        mean_loss = train_losses.mean().item()
        bar(f"Epoch {epoch} loss: {mean_loss}")
        train_mean_losses[epoch - 1] = mean_loss


def train_batch(
    X: torch.Tensor,
    label: torch.Tensor,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_function: nn.BCEWithLogitsLoss,
):
    # Обнуляем градиенты всех параметров
    optimizer.zero_grad()
    # Вызов метода forward(). Прямой проход по сети
    out = model(X)
    # Вычисление функции потерь. criterion - функция потерь
    loss = loss_function(out, label)
    # Обратное распространение функции потерь. Вычисление градиентов функции потерь
    loss.backward()
    # Обновление параметров оптимизатором на основе вычисленных ранее градиентов
    optimizer.step()
    # Возвращаем значение функции потерь
    return loss.item()


def predict(model: nn.Module, X: torch.Tensor) -> np.ndarray:
    """Предсказание модели

    Args:
        model (nn.Module): модель
        X (torch.Tensor): (batch_size, ...)

    Returns:
        np.ndarray: (batch_size, n_classes)
    """
    # logits_ - логиты (ненормализованные вероятности) для каждого класса для каждого примера в пакете данных.
    logits = model(X)  # (batch_size, n_classes)
    # sigmoid - преобразование логитов в вероятности, т.е. в числа в диапазоне [0,1]
    probabilities = torch.sigmoid(logits).cpu()

    return (probabilities > 0.5).numpy()


def evaluate(model: nn.Module, dataloader: DataLoader, diagnostic_classes: tuple[str]):
    """Оценка качества модели

    Args:
        model (nn.Module): модель
        dataloader (DataLoader): dev_dataloader or test_dataloader
        diagnostic_classes (tuple[str]): классы диагнозов meta["labels"] = {0: "MI", 1: "STTC", 2: "CD", 3: "HYP"}
    """
    # Перевод модели в режим оценки
    model.eval()
    # Отключение вычисления градиентов
    with torch.no_grad():
        """ 
            |      | TP | TN | FP | FN | Sensitivity | Specificity | G-mean |
            |------|----|----|----|----|-------------|-------------|--------|
            | MI   |    |    |    |    |             |             |        |
            | STTC |    |    |    |    |             |             |        |
            | CD   |    |    |    |    |             |             |        |
            | HYP  |    |    |    |    |             |             |        |
            | all  |    |    |    |    |             |             |        |
        """
        diagnostic_classes += ("all",)
        quality_metrics = pd.DataFrame(
            data=np.zeros((len(diagnostic_classes), 7)),
            index=diagnostic_classes,
            columns=("TP", "TN", "FP", "FN", "Sensitivity", "Specificity", "G-mean"),
        )
        for i, (X, label) in enumerate(dataloader):
            print("eval {} of {}".format(i + 1, len(dataloader)), end="\r")
            # прогноз модели
            y_predict = predict(model, X)
            # метки
            y_true = label.cpu().numpy()

            quality_metrics = calculate_tptnfpfn(y_predict, y_true, quality_metrics)

            # Очистка памяти
            del X
            del label
            torch.cuda.empty_cache()
    quality_metrics = calculate_all_tptnfpfn(quality_metrics)
    quality_metrics = calculate_metrics(quality_metrics)
    # Возврат модели в режим обучения
    model.train()

    return quality_metrics
