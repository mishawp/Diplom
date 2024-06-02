import os
import pickle
import torch
from torch import nn
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Callable

from .dataset import MyDataset
from .config_run import set_seed, set_device

project_path = os.getenv("project_root")


def start_training(
    model: nn.Module,
    dimension: int,
    model_name: str,
    dataset_name: str,
    params_nn: dict,
    params_other: dict,
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
        dataset_train, batch_size=params_other["batch_size"], shuffle=True
    )
    dataloader_dev = DataLoader(dataset_dev, batch_size=1, shuffle=False)
    dataloader_test = DataLoader(dataset_test, batch_size=1, shuffle=False)

    optimizer = (
        torch.optim.Adam if params_other["optimizer"] == "adam" else torch.optim.SGD
    )
    optimizer = optimizer(
        model.parameters(),
        lr=params_other["learning_rate"],
        weight_decay=params_other["l2_decay"],
    )
