import numpy as np

items = np.arange(10)
index = np.random.choice(np.arange(10000), 10, replace=False)
items_idx = np.vstack((items, index))
print(items_idx)
print(items_idx[1, :].argsort())
items_idx = items_idx[:, items_idx[1, :].argsort()]
print(items_idx)
