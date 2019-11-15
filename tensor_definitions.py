import numpy as np

#########################################################################
# 0D Tensor - Scalar - Value
# Shape = ()

tensor_0d = 3
tensor_0d = np.array(tensor_0d)
print(tensor_0d.shape)

#########################################################################
# 1D Tensor - Vector - Array
# Shape = (3)

tensor_1d = [1, 2, 3]
tensor_1d = np.array(tensor_1d)
print(tensor_1d.shape)

#########################################################################
# 2D Tensor - Matrix - 2D-array
# Shape = (3, 3) = (rows, columns)

tensor_2d = [
  [1, 2, 3],
  [1, 2, 3],
  [1, 2, 3]
]
tensor_2d = np.array(tensor_2d)
print(tensor_2d.shape)

#########################################################################
# 3D Tensor
# Shape = (3, 3, 3) = (axis 0, 1, 2)

tensor_3d = [
  [
    [1, 2, 3],
    [1, 2, 3],
    [1, 2, 3]
  ],
  [
    [1, 2, 3],
    [1, 2, 3],
    [1, 2, 3]
  ],
  [
    [1, 2, 3],
    [1, 2, 3],
    [1, 2, 3]
  ]
]
tensor_3d = np.array(tensor_3d)
print(tensor_3d.shape)

