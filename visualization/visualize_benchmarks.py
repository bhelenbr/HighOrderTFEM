import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
sns.set_theme()


data = pd.read_csv("benchmark_data.csv")
data["num_points"] = data["Mesh"].map(lambda n: pow(pow(2, n-1)+1, 2))
data["num_elements"] = data["Mesh"].map(lambda n: 2 * pow(4, n - 1))
data["element_throughput"] = 10000 * data["num_elements"] / data["Time"]
data = data[data["Mesh"] > 4]

print(data.groupby(["Device", "Algorithm"]).aggregate("max") / 6.201151e+07)


plt.figure()
sns.lineplot(data=data, x="num_elements",y="Time",hue="Device", style="Algorithm")
plt.loglog()
plt.xlabel("Mesh Size (Elements)")
plt.ylabel("Time (s)")
plt.suptitle("Runtimes to compute 10000 time steps")
plt.title("By Device and Algorithm Type")
plt.savefig("Runtimes.svg")
plt.close()

plt.figure()
sns.lineplot(data=data, x="num_elements",y="element_throughput",hue="Device", style = "Algorithm")
plt.loglog()
plt.xlabel("Mesh Size (Elements)")
plt.ylabel("Throughput (Elements / s)")
plt.suptitle("Element Throughput")
plt.title("By Device and Algorithm Type")
plt.savefig("Throughput.svg")
plt.close()