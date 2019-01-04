import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

true_labels = [2,0,0,2,4,4,1,0,3,3,3]
pred_labels = [2,1,0,2,4,3,1,0,1,3,3]

confusion_mat = confusion_matrix(true_labels, pred_labels)

plt.imshow(confusion_mat, interpolation='nearest', cmap=plt.cm.gray)
plt.title('Confusion matrix')
plt.colorbar()
ticks = np.arange(5)
plt.xticks(ticks, ticks)
plt.yticks(ticks, ticks)
plt.ylabel('True labels')
plt.xlabel('Predicted labels')
plt.show()

targets = ['Class-0', 'Class-1', 'Class-2', 'Class-3', 'Class-4']
print("\n", classification_report(true_labels, pred_labels, target_names=targets))

#      precision    recall  f1-score   support

#      Class-0       1.00      0.67      0.80         3
#      Class-1       0.33      1.00      0.50         1
#      Class-2       1.00      1.00      1.00         2
#      Class-3       0.67      0.67      0.67         3
#      Class-4       1.00      0.50      0.67         2

#    micro avg       0.73      0.73      0.73        11
#    macro avg       0.80      0.77      0.73        11
# weighted avg       0.85      0.73      0.75        11
