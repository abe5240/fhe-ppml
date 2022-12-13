import numpy as np
import matplotlib.pyplot as plt 
from matplotlib.pyplot import cm
import seaborn as sns


if __name__ == "__main__":

    polynomial_orders = np.arange(2,11)
    model_names = ["MLPNet", "ConvNet"]
    activations = ["LeakyRelu", "Sigmoid", "Softplus"]

    colors = cm.Set1(np.linspace(0, 1, len(polynomial_orders)))

    accs, perfs = [], []
    first = True
    for model_name in model_names:
        for activation in activations:
            for i, order in enumerate(polynomial_orders):

                with open(f"checkpoint/{model_name}/{order}/stats_{activation}.txt", "r") as f:
                    lines = f.readlines()


                for line in lines:
                    if "Accuracy" in line:
                        acc = float(line.strip().split(":")[1])
                    if "Time" in line:
                        perf = float(line.strip().split(":")[1])

                if activation == "LeakyRelu":
                    symbol = 'o' 
                elif activation == "Sigmoid":
                    symbol = 'P'
                elif activation == "Softplus":
                    symbol = 'D'

                if model_name == "ConvNet":
                    plt.scatter(perf/1000, acc, color=colors[i], marker=symbol, edgecolors='black')
                else:
                    if first:
                        plt.scatter(perf/1000, acc, color=colors[i], marker=symbol, label=f"{order}")
                    else:
                        plt.scatter(perf/1000, acc, color=colors[i], marker=symbol)

            first = False

    #plt.plot(accs, perfs, "o")
    plt.xlabel("Encrypted Inference Time (s)")
    plt.ylabel("Accuracy")
    #plt.legend(ncol=3)
    sns.despine()
    plt.tight_layout()
    #plt.show()
    plt.savefig("inference-results.png", dpi=200)

