import numpy as np

file_path = "../resources/tcn_time_point.npy"

content = np.load(file_path, allow_pickle=True)

content = content.transpose(2, 0, 1).squeeze(axis=2)

print(content)
