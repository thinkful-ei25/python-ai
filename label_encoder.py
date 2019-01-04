import numpy as np
from sklearn import preprocessing

input_labels = ['red', 'black', 'red', 'green',
                'black', 'yellow', 'white']

encoder = preprocessing.LabelEncoder()
encoder.fit(input_labels)

# map the labels
print("\nLabel mapping:")
for i, item in enumerate(encoder.classes_):
  print(item, '--->', i)

# Label mapping:
# black ---> 0
# green ---> 1
# red ---> 2
# white ---> 3
# yellow ---> 4

test_labels = ['green', 'red', 'black']
encoded_values = encoder.transform(test_labels)
print("\nLabels = ", test_labels)
print("Encoded values = ", list(encoded_values))

# Labels =  ['green', 'red', 'black']
# Encoded values =  [1, 2, 0]

encoded_values = [3,0,4,1]
decoded_list = encoder.inverse_transform(encoded_values)
print("\nEncoded values =", encoded_values)
print("Decoded labels =", list(decoded_list))

# Labels =  ['green', 'red', 'black']
# Encoded values =  [1, 2, 0]
