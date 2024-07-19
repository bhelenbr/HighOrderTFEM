import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import math
sns.set_theme()

data = pd.read_csv("accuracy_data_unfuzzed.csv")
data["num_points"] = data["Mesh"].map(lambda n: pow(pow(2, n-1)+1, 2))
data["num_elements"] = data["Mesh"].map(lambda n: 2 * pow(4, n - 1))
data["dx"] = data["num_points"].map(lambda n: 2.0 / (math.sqrt(n) - 1))

plt.figure(figsize=(3.5,6))
sns.lineplot(data=data, x="dx",y="RMSE")
plt.gca().set_aspect("equal")
plt.loglog()
plt.title("Error vs Mesh Size")
plt.ylabel("Root Mean Square Error")
plt.xlabel("Mesh Spacing (\"dx\")")
plt.tight_layout()
plt.savefig("Accuracy.svg")