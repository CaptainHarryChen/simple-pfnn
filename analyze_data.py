import numpy as np
import matplotlib.pyplot as plt

f = np.load("processed_data.npz")

data_ph, data_in, data_out, in_mean, in_std, out_mean, out_std = f["data_ph"], f["data_in"], f["data_out"], f["in_mean"], f["in_std"], f["out_mean"], f["out_std"]
# data_in = data_in * in_std + in_mean
# data_out = data_out * out_std + out_mean

plt.hist(data_in[:,10], bins=100)
# plt.hist(data_out[:, 2], bins=100)
# plt.scatter(data_in[:,10], data_out[:, 2], s=1)

# x = np.arctan2(data_in[:,10], data_in[:,11])
# y = np.arctan2(data_out[:, 2], data_out[:,3])
# plt.hist(x, bins=100)
# plt.hist(y, bins=100)
# plt.scatter(x, y, s=1)

plt.show()
