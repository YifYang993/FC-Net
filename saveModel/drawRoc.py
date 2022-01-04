import scikitplot as skplt
import matplotlib.pyplot as plt
from numpy import *
from sklearn import metrics
import numpy as np

# y_probas =  [[1.],[0.],[0.],[1.],[0.],[0.],[0.],[0.],[0.]]
# y_true= [1, 0, 0, 1, 0, 0, 0, 0, 0] # predicted probabilities generated by sklearn classifier

outputs = [array([[0.01045937],
       [0.03820571]], dtype=float32), array([[0.00703975],
       [0.00671291]], dtype=float32), array([[0.4516132 ],
       [0.00123392]], dtype=float32), array([[0.50106909],
       [0.04493943]], dtype=float32), array([[0.9304484]], dtype=float32)]
labels = [array([0., 0.], dtype=float32), array([0., 0.], dtype=float32), array([1., 0.], dtype=float32), array([0., 0.], dtype=float32), array([1.], dtype=float32)]

if isinstance(outputs, list):
    pred = np.concatenate(outputs, axis=0)
    y = np.concatenate(labels, axis=0)
else:
    pred = outputs
    y = labels
fpr, tpr, thresholds = metrics.roc_curve(y, pred, pos_label=1)
print(fpr, tpr ,thresholds)
roc_auc = metrics.auc(fpr, tpr)
if np.isnan(roc_auc):
    roc_auc = 0
# roc_auc = metrics.auc(fpr, tpr)


plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.savefig("epoch_RocOf.png")

# skplt.metrics.plot_roc_curve(y_true, y_probas)


# plt.show()

# plt.savefig("ROC.jpg")
# plt.show()

# tn, fp, fn, tp = metrics.confusion_matrix (y, pred).ravel ()