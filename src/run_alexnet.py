from dotenv import load_dotenv

load_dotenv()
from utils import start_training
from nn2d import AlexNet

if __name__ == "__main__":
    model_parameters = {
        "input_size": 3,
        "hidden_size": 128,
        "num_layers": 2,
        "device": "cuda",
    }
    parameters = {
        "epochs": 1,
        "batch_size": 256,
        "learning_rate": 0.01,
        "l2_decay": 0.0,
        "optimizer": "adam",
        "device": "cuda",
    }
    start_training(AlexNet, 1, "gru", "my_data", parameters, model_parameters)
