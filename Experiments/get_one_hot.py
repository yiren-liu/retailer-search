import numpy as np

index = np.load("descriptions_big_one_hot_index.npy")
one_hot = np.zeros((*index.shape,index.shape[-1]))