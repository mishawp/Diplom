import numpy as np
import pandas as pd


def calculate_tptnfpfn(
    y_predict: np.ndarray, y_true: np.ndarray, quality_metrics: pd.DataFrame
):
    """Вычисление метрик TP | TN | FP | FN для каждого класса в метке.

    Args:
        y_predict (np.ndarray): прогноз модели (batch_size, num_classes)
        y_true (np.ndarray): метка (batch_size, num_classes)
        quality_metrics (pd.DataFrame): Таблица вида

        |      | TP | TN | FP | FN | Sensitivity | Specificity | G-mean |
        |------|----|----|----|----|-------------|-------------|--------|
        | MI   |    |    |    |    |             |             |        |
        | STTC |    |    |    |    |             |             |        |
        | CD   |    |    |    |    |             |             |        |
        | HYP  |    |    |    |    |             |             |        |
        | all  |    |    |    |    |             |             |        |
    Returns:
        pd.DataFrame: обновленный quality_metrics
    """
    for prediction, true in zip(y_predict, y_true):
        for i in range(len(quality_metrics)):
            quality_metrics.iloc[i]["TP"] += (
                prediction[i] == 1 and true[i] == 1
            ).item()
            quality_metrics.iloc[i]["TN"] += (
                prediction[i] == 0 and true[i] == 0
            ).item()
            quality_metrics.iloc[i]["FP"] += (
                prediction[i] == 1 and true[i] == 0
            ).item()
            quality_metrics.iloc[i]["FN"] += (
                prediction[i] == 0 and true[i] == 1
            ).item()

    return quality_metrics


def calculate_all_tptnfpfn(quality_metrics: pd.DataFrame):
    """Вычисление метрик общего кол-ва TP | TN | FP | FN (строка 'all').

    Args:
        quality_metrics (pd.DataFrame): Таблица вида
        |      | TP | TN | FP | FN | Sensitivity | Specificity | G-mean |
        |------|----|----|----|----|-------------|-------------|--------|
        | MI   |    |    |    |    |             |             |        |
        | STTC |    |    |    |    |             |             |        |
        | CD   |    |    |    |    |             |             |        |
        | HYP  |    |    |    |    |             |             |        |
        | all  |    |    |    |    |             |             |        |
    Returns:
        pd.DataFrame: обновленный quality_metrics
    """
    for column in quality_metrics.columns:
        quality_metrics["all"][column] = quality_metrics[column].sum()

    return quality_metrics


def calculate_metrics(quality_metrics: pd.DataFrame):
    """Вычисление метрик Sensitivity | Specificity | G-mean для каждого класса в метке.

    Args:
        quality_metrics (pd.DataFrame): Таблица вида

        |      | TP | TN | FP | FN | Sensitivity | Specificity | G-mean |
        |------|----|----|----|----|-------------|-------------|--------|
        | MI   |    |    |    |    |             |             |        |
        | STTC |    |    |    |    |             |             |        |
        | CD   |    |    |    |    |             |             |        |
        | HYP  |    |    |    |    |             |             |        |
        | all  |    |    |    |    |             |             |        |
    Returns:
        pd.DataFrame: обновленный quality_metrics
    """
    for row in quality_metrics:
        row["Sensitivity"] = row["TP"] / (row["TP"] + row["FN"])
        row["Specificity"] = row["TN"] / (row["TN"] + row["FP"])
        row["G-mean"] = np.sqrt((row["Sensitivity"] + row["Specificity"]))

    return quality_metrics
