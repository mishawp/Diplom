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
from .metrics import compute_tptnfpfn, compute_all_tptnfpfn, compute_metrics
from ploting import save_plot

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
    epochs = np.arange(1, parameters["epochs"] + 1)
    statistics = pd.DataFrame(
        data=np.zeros((parameters["epochs"], 4)),
        index=epochs,
        columns=("Training-loss", "Validation-loss", "Sensitivity", "Specificity"),
    )
    # потери за каждый батч
    train_losses = np.zeros(np.ceil(meta["n_train"] / dataloader_train.batch_size))

    # best_model_dev_loss = (model, float("inf"))
    best_model_sensitivity = (model, 0)
    for epoch in epochs:
        bar(f"Epoch {epoch}")

        train_losses[:] = 0
        for i, (X, label) in enumerate(dataloader_train):
            loss = train_batch(X, label, model, optimizer, loss_function)

            del X
            del label
            torch.cuda.empty_cache()

            train_losses[i] = loss

        train_mean_loss = train_losses.mean().item()
        dev_mean_loss = compute_dev_loss(
            model, dataloader_dev, loss_function, meta["n_dev"]
        )
        statistics[epoch]["Training-loss"] = train_mean_loss
        statistics[epoch]["Validation-loss"] = dev_mean_loss
        quality_metrics = evaluate(model, dataloader_dev, meta["labels"].values())
        statistics[epoch]["Sensitivity"] = quality_metrics["all"]["Sensitivity"]
        statistics[epoch]["Specificity"] = quality_metrics["all"]["Specificity"]

        bar(f"Epoch {epoch} loss: {train_mean_loss}")
        bar(f"Valid sensitivity: {quality_metrics["all"]["Sensitivity"]:.4f}")
        bar(f"Valid specificity: {quality_metrics["all"]["Specificity"]:.4f}")

        # if dev_mean_loss < best_model_dev_loss[1]:
        #     best_model_dev_loss = (model, dev_mean_loss)
        if quality_metrics["all"]["Specificity"] > best_model_sensitivity[1]:
            best_model_sensitivity = (model, quality_metrics["all"]["Specificity"])

    # Сохраняем модель
    path_save_model = Path(project_path, "models")
    torch.save(
        best_model_sensitivity[0].state_dict(), path_save_model / f"{model_name}.pth"
    )

    # Сохраняем эпохи обучения
    path_reports = Path(project_path, "reports")
    statistics.to_csv(path_reports / f"{model_name}_training_report.csv")

    # Тестирование на тестовой выборке
    test_quality_metrics = evaluate(
        best_model_sensitivity[0], dataloader_test, meta["labels"].values()
    )
    test_quality_metrics.to_csv(path_reports / f"{model_name}_test_report.csv")

    save_plot(
        statistics["Training-loss"],
        "Training-Loss",
        path_reports,
        f"{model_name}_training_loss",
    )
    save_plot(
        statistics["Sensitivity"],
        "Sensitivity",
        path_reports,
        f"{model_name}_sensitivity",
    )
    save_plot(
        statistics["Specificity"],
        "Specificity",
        path_reports,
        f"{model_name}_specificity",
    )


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


def compute_dev_loss(
    model: nn.Module,
    dataloader: DataLoader,
    loss_function: nn.BCEWithLogitsLoss,
    n_dev: int,
) -> float:
    """Вычисление потерь на валидационном датасете

    Args:
        model (nn.Module): модель
        dataloader (DataLoader): датасет
        loss_function (nn.BCEWithLogitsLoss): функция потерь
        dev_int (int): количество образцов в валидационном датасете

    Returns:
        float: потери
    """
    model.eval()
    with torch.no_grad():
        dev_losses = [np.zeros(np.ceil(n_dev / dataloader.batch_size))]
        for i, (X, label) in enumerate(dataloader):
            print("eval {} of {}".format(i + 1, len(dataloader)), end="\r")
            y_pred = model(X)
            loss = loss_function(y_pred, label)
            dev_losses.append(loss.item())
            del X
            del label
            torch.cuda.empty_cache()

    model.train()

    return np.mean(dev_losses)


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


def evaluate(
    model: nn.Module, dataloader: DataLoader, diagnostic_classes: tuple[str]
) -> pd.DataFrame:
    """Оценка качества модели

    Args:
        model (nn.Module): модель
        dataloader (DataLoader): dev_dataloader or test_dataloader
        diagnostic_classes (tuple[str]): классы диагнозов meta["labels"] = {0: "MI", 1: "STTC", 2: "CD", 3: "HYP"}

    Returns:
        pd.DataFrame:
        |      | TP | TN | FP | FN | Sensitivity | Specificity | G-mean |
        |------|----|----|----|----|-------------|-------------|--------|
        | MI   |    |    |    |    |             |             |        |
        | STTC |    |    |    |    |             |             |        |
        | CD   |    |    |    |    |             |             |        |
        | HYP  |    |    |    |    |             |             |        |
        | all  |    |    |    |    |             |             |        |
    """
    # Перевод модели в режим оценки
    model.eval()
    # Отключение вычисления градиентов
    with torch.no_grad():
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

            quality_metrics = compute_tptnfpfn(y_predict, y_true, quality_metrics)

            # Очистка памяти
            del X
            del label
            torch.cuda.empty_cache()
    quality_metrics = compute_all_tptnfpfn(quality_metrics)
    quality_metrics = compute_metrics(quality_metrics)
    # Возврат модели в режим обучения
    model.train()

    return quality_metrics
