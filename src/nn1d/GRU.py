import torch
from torch import nn


class GRU(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        dropout,
        device=None,
    ):
        """
        Define the layers of the model
        Args:
            input_size (int): Кол-во входных признаков (1 на кол-во отведений, участвующих в обучение)
            hidden_size (int): Кол-во скрытых нейронов
            num_layers (int): Кол-во слоев
            dropout (float): Вероятность отключения нейронов
        """
        super(GRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.device = device
        self.bidirectional = 1  # 1 - не двунаправленный, 2 - двунаправленный

        self.gru = nn.GRU(
            input_size,
            hidden_size,
            num_layers,
            dropout=dropout,
            batch_first=True,
            bidirectional=self.bidirectional - 1,
        )

        # 4 - на число классов
        self.fc = nn.Linear(hidden_size * self.bidirectional, 4)

    def forward(self, X):
        """
        Forward Propagation

        Args:
            X: batch of training examples with dimension (batch_size, 1000, 3)
        """
        # initial hidden state:
        h_0 = torch.zeros(
            self.num_layers * self.bidirectional, X.size(0), self.hidden_size
        ).to(self.device)

        out_gru, _ = self.gru(X.to(self.device), h_0)
        # out_rnn shape: (batch_size, seq_length, hidden_size*bidirectional) = (batch_size, 1000, hidden_size*bidirectional)

        # last timestep
        out_gru = out_gru[:, -1, :]

        # out_rnn shape: (batch_size, hidden_size*bidirectional) - ready to enter the fc layer
        out_fc = self.fc(out_gru)
        # out_fc shape: (batch_size, num_classes)

        return out_fc


class GRUCreator:
    """Просто интерфейс для передачи нейронный сети"""

    def __init__(self) -> None:
        self.params = {
            "input_size": int,
            "hidden_size": int,
            "num_layers": int,
            "dropout": float,
            "device": str,
        }

    def create(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        device: str = None,
    ):
        """Проверка корректности передаваемых параметров"""
        for param, param_type in self.params:
            if isinstance(param, param_type):
                return param
        if device not in ["cpu", "cuda", "mps"]:
            return "device"
        if dropout < 0 or dropout > 1:
            return "dropout"

        return GRU(input_size, hidden_size, num_layers, dropout, device)
