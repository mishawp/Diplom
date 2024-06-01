from torch import nn


class AlexNet(nn.Module):
    def __init__(self, input_size, nb_filters, dropout, device=None):
        """
        Define the layers of the model
        Args:
            input_size (int): Кол-во входных признаков (1 на кол-во отведений, участвующих в обучение)
            nb_filters (int): Кол-во фильтров в первом слое
            dropout (float): Коэффициент регуляризации
        """
        super(AlexNet, self).__init__()
        self.input_size = input_size
        self.nb_filters = nb_filters
        self.dropout = nn.Dropout2d(dropout)
        self.device = device

        # 9 input channels nn.Conv2d(in_channels, out_channels, kernel_size)
        self.conv2d_1 = nn.Conv2d(9, nb_filters, 11, stride=4)
        self.conv2d_2 = nn.Conv2d(nb_filters, nb_filters * 2, 5, padding=2)
        self.conv2d_3 = nn.Conv2d(nb_filters * 2, nb_filters * 4, 3, padding=1)
        self.conv2d_4 = nn.Conv2d(nb_filters * 4, nb_filters * 8, 3, padding=1)
        self.conv2d_5 = nn.Conv2d(nb_filters * 8, 256, 3, padding=1)
        self.linear_1 = nn.Linear(9216, 4096)
        self.linear_2 = nn.Linear(4096, 2048)
        # 4 - на число классов
        self.linear_3 = nn.Linear(2048, 4)
        # nn.MaxPool2d(kernel_size)
        self.maxpool2d = nn.MaxPool2d(3, stride=2)
        self.relu = nn.ReLU()

    def forward(self, X):
        """
        Forward Propagation
        Args:
            X: batch of training examples with dimension (batch_size, 9, 1000, 1000)
        """
        x1 = self.relu(self.conv2d_1(X))
        maxpool1 = self.maxpool2d(x1)
        maxpool1 = self.dropout(maxpool1)
        x2 = self.relu(self.conv2d_2(maxpool1))
        maxpool2 = self.maxpool2d(x2)
        maxpool2 = self.dropout(maxpool2)
        x3 = self.relu(self.conv2d_3(maxpool2))
        x4 = self.relu(self.conv2d_4(x3))
        x5 = self.relu(self.conv2d_5(x4))
        x6 = self.maxpool2d(x5)
        x6 = self.dropout(x6)
        x6 = x6.reshape(x6.shape[0], -1)  # flatten (batch_size,)
        x7 = self.relu(self.linear_1(x6))
        x8 = self.relu(self.linear_2(x7))
        x9 = self.linear_3(x8)
        return x9


class AlexNetCreator:
    """Просто интерфейс для передачи нейронный сети"""

    def __init__(self) -> None:
        self.params = {
            "input_size": int,
            "nb_filters": int,
            "dropout": float,
            "device": str,
        }

    def create(
        self,
        input_size: int,
        nb_filters: int,
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

        return AlexNet(input_size, nb_filters, device)
