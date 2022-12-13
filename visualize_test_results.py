import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


polynomial_orders = np.arange(2,11)
model_names = ["MLPNet", "ConvNet"]
activations = ["LeakyRelu", "Sigmoid", "Softplus"]

df = pd.DataFrame(columns=['Order', 'Accuracy', "Model"])
for model_name in model_names:
    for activation in activations:
        for order in polynomial_orders:

            with open(f"checkpoint/{model_name}/{order}/stats_{activation}.txt", "r") as f:
                lines = f.readlines()

            for line in lines:
                if "Accuracy" in line:
                    acc = float(line.strip().split(":")[1])

            mname = "MLP" if model_name == "MLPNet" else "CNN"

            tmp = pd.DataFrame(columns=list(df.columns))
            tmp.loc[0] = [order, acc, f"{mname}"]
            df = pd.concat([df, tmp], ignore_index=True)

sns.violinplot(data=df, x="Order", y="Accuracy", hue="Model", split=True)

plt.tight_layout()
plt.savefig('visualize_test_results.eps')
plt.show()