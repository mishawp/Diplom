import torch
import pandas as pd
import numpy as np

torch.set_default_device("cuda")

tensor1 = torch.tensor([0])
tensor2 = torch.tensor([1])

arr = np.array([[0, 0], [0, 0]])
fr = pd.DataFrame(arr, index=["a", "b"], columns=["1", "2"])
fr.iloc[0]["1"] += ((tensor1[0] == 0) & (tensor2[0] == 1)).item()

print(fr)
