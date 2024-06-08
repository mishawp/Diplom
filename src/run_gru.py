from dotenv import load_dotenv

load_dotenv()
from utils import start_training
from nn1d import GRU

if __name__ == "__main__":
    model_parameters = {
        "input_size": 3,
        "hidden_size": 128,
        "num_layers": 2,
        "dropout": 0.3,
        "device": "cuda",
    }
    parameters = {
        "epochs": 30,
        "patience_limit": 5,
        "batch_size": 256,
        "learning_rate": 0.01,
        "l2_decay": 0.0,
        "optimizer": "adam",
        "device": "cuda",
    }
    start_training(GRU, 1, "gru", "1d-data", parameters, model_parameters)
