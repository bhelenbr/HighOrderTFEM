import matplotlib.pyplot as plt
import seaborn as sns
import json
import numpy as np
sns.set_theme()

file = "out/slices.json"

with open(file, "r") as data_file:
    data = json.load(data_file)

point_locs = np.array(data["points"]).T
print(point_locs.shape)

slice_values = np.array(data["slices"])
print(slice_values.shape)

for t in range(0, slice_values.shape[0]):
    sns.scatterplot(x = point_locs[0], y = point_locs[1], hue=slice_values[t], hue_norm=(-1, 1), palette=sns.color_palette("icefire", as_cmap=True))
    plt.savefig(f"viz_out/slice_{t}.png")
    plt.close()