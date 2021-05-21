from skimage.util import view_as_windows
import numpy as np


def window_array_1d(array_1d, window_width, step):
    windowed = view_as_windows(array_1d, window_width, step)

    return windowed


# a = np.array([1,2,3,1,2,3])
#
# print(a)
#
# # b = np.pad(a.reshape(*a.shape[:-1]),((0,0)))
# b = window_array_1d(a, 3, 1)
# print(b)
#
# c = np.array([7,8,9,7,8,9])
# print(c)
# d = window_array_1d(c, 3, 1)
# print(d)
#
# b = np.expand_dims(b,axis=-1)
# # print(b)
# d = np.expand_dims(d,axis=-1)
# e = np.concatenate((b,d),axis=2)
# print(e)