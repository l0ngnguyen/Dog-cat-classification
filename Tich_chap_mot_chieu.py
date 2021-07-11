import numpy as np

tin_hieu = np.array([2, 4, -2, -2, -2, -1, 2 , 5, 4, 0, -1, 0])
bo_loc = np.array([1, 0, -1])

def tich_chap(tin_hieu, bo_loc, padding=bo_loc.size-1, stride=1):
    if tin_hieu.size < bo_loc.size: return 0
    res = np.zeros((tin_hieu.size))
    zeros = np.zeros((int(padding/2)))
    padding_array = np.hstack((zeros, tin_hieu, zeros))
    for i in range(tin_hieu.size):
        res[i] = (padding_array[i: i+bo_loc.size] * bo_loc).sum()
    return res

def tich_chap_2_chieu(input, filter, stride=1):
    input_size = input.shape
    filter_size = filter.shape

    if (input_size[0] < filter_size[0]) or (input_size[1] < filter_size[1]):
        return 0

    res = np.zeros((input_size[0] - filter_size[0] + 1, input_size[1] - filter_size[1] + 1))
    for i in range(res.shape[0]):
        for j in range(res.shape[1]):
            a = input[i: i+filter_size[0], j: j+filter_size[1]]
            res[i, j] = (a*filter).sum()
    return res

input = np.array([[1, 3, 5, 7, 9],
                  [2, 5, 0, 1, 2],
                  [3, 3, 2, 0, 7]])
a = np.array([[1, 2, 4, 5, 6, 7],
              [2, 4, 6, 7, 8, 9],
              [2, 0, 3, 2, 1, 1],
              [2, 4, 0, 2, 4, 7]])
filter = np.array([[2, -1, 0],
                   [1, 0, 1]])
print(tich_chap_2_chieu(a, filter))
        
