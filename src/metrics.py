import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

def summarize(y_true, y_pred):
    print(classification_report(y_true, y_pred, digits=3))
    print("Confusion matrix:\n", confusion_matrix(y_true, y_pred))
