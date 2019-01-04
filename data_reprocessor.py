import numpy as np
from sklearn import preprocessing

input_data = np.array([[5.1, -2.9, 3.3],
                       [-1.2, 7.8, -6.1],
                       [3.9, 0.4, 2.1],
                       [7.3, -9.9, -4.5]])
print(input_data)

data_binarized = preprocessing.Binarizer(threshold=2.1).transform(input_data)
print("\nBinarized data:\n", data_binarized)

# Binarized data:
#  [[1. 0. 1.]
#  [0. 1. 0.]
#  [1. 0. 0.]
#  [1. 0. 0.]]

print("\nBefore:")
print("Mean =", input_data.mean(axis=0))
print("Std deviation =", input_data.std(axis=0))

# Before:
# Mean = [ 3.775 -1.15  -1.3  ]
# Std deviation = [3.12039661 6.36651396 4.0620192 ]


data_scaled = preprocessing.scale(input_data)
print("\nAFTER:")
print("Mean =", data_scaled.mean(axis=0))
print("Std deviation =", data_scaled.std(axis=0))

# AFTER:
# Mean = [1.11022302e-16 0.00000000e+00 2.77555756e-17]
# Std deviation = [1. 1. 1.]

# scale data so max is 1 and other values are relative
data_scaler_minmax = preprocessing.MinMaxScaler(feature_range=(0, 1))
data_scaler_minmax = data_scaler_minmax.fit_transform(input_data)
print("\nMin Max scaled dada:\n", data_scaler_minmax)

# Min Max scaled dada:
#  [[0.74117647 0.39548023 1.        ]
#  [0.         1.         0.        ]
#  [0.6        0.5819209  0.87234043]
#  [1.         0.         0.17021277]]

# L1 for robust with no outliers
data_normalized_l1 = preprocessing.normalize(input_data, norm='l1')

# L2 for when outliers are important
data_normalized_l2 = preprocessing.normalize(input_data, norm='l2')
print("\nL1 normalized data:\n", data_normalized_l1)
print("\nL2 normalized data:\n", data_normalized_l2)

# L1 normalized data:
#  [[ 0.45132743 -0.25663717  0.2920354 ]
#  [-0.0794702   0.51655629 -0.40397351]
#  [ 0.609375    0.0625      0.328125  ]
#  [ 0.33640553 -0.4562212  -0.20737327]]

# L2 normalized data:
#  [[ 0.75765788 -0.43082507  0.49024922]
#  [-0.12030718  0.78199664 -0.61156148]
#  [ 0.87690281  0.08993875  0.47217844]
#  [ 0.55734935 -0.75585734 -0.34357152]]
