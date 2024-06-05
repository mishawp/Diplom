import matplotlib.pyplot as plt
import pandas as pd
from dotenv import load_dotenv
from utils import start_training
from nn1d import GRU

if __name__ == "__main__":
    load_dotenv()
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
    start_training(GRU, 1, "gru", "my_data", parameters, model_parameters)
