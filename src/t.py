import matplotlib.pyplot as plt
import pandas as pd


def plot(plottable, ylabel="", name=""):
    plt.clf()
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.plot(plottable)
    plt.savefig("%s.pdf" % (name), bbox_inches="tight")


fr = pd.DataFrame(data=[[0, 0], [1, 0], [0, 0]], index=[1, 2, 3], columns=["1", "2"])

fr["1"].plot(kind="bar")
