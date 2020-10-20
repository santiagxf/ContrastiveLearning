import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (confusion_matrix, precision_recall_curve, auc,
                             roc_curve, recall_score, classification_report, f1_score,
                             precision_recall_fscore_support)
import pandas as pd

def plot_loss_by_epoch(trainer):
    plt.figure(figsize=(14, 6))
    plt.plot(trainer.history['loss'])
    plt.plot(trainer.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right');

    plt.show()

def plot_reconstructuion_err_by_class(X_test, y_test, model, anomaly_class = 1, plot_th = False, threshold = 0):
    predictions = model.predict(X_test)
    mse = np.mean(np.power(X_test - predictions, 2), axis=1)
    error_df = pd.DataFrame({'reconstruction_error': mse,
                            'true_class': y_test})
    
    #print reconstruction mse by class 
    print(error_df.groupby('true_class').mean())
    
    groups = error_df.groupby('true_class')
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # small hack. The function reverse the order of the group by to ensure fraudulent instances are ploted on top
    # of normal ones. Otherwise they will overlap
    if anomaly_class == 1:
        group_fn = list
    else:
        group_fn = reversed

    for name, group in group_fn(list(groups)):
        ax.plot(group.index, group.reconstruction_error, marker='o', ms=3.5, linestyle='',
                label= "Fraud" if name == anomaly_class else "Normal")
    
    if plot_th:
        ax.hlines(threshold, ax.get_xlim()[0], ax.get_xlim()[1], colors="r", zorder=100, label='Threshold')
    ax.legend()
    plt.title("Reconstruction error for different classes")
    plt.ylabel("Reconstruction error")
    plt.xlabel("Data point index")
    plt.show();
    

def plot_precision_recall_curves(X_test, y_test, model, anomaly_class = 1):
    predictions = model.predict(X_test)
    mse = np.mean(np.power(X_test - predictions, 2), axis=1)
    error_df = pd.DataFrame({'reconstruction_error': mse,
                            'true_class': y_test})
    
    # ROC
    fpr, tpr, thresholds = roc_curve(error_df.true_class, error_df.reconstruction_error, pos_label = anomaly_class)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(14, 6))
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, label='AUC = %0.4f'% roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0,1],[0,1],'r--')
    plt.xlim([-0.001, 1])
    plt.ylim([0, 1.001])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show();
    
    # Precision Recall
    precision, recall, th = precision_recall_curve(error_df.true_class, 
                                                   error_df.reconstruction_error, 
                                                   pos_label = anomaly_class)

    plt.figure(figsize=(14, 6))
    plt.plot(th, precision[1:], label="Precision",linewidth=3)
    plt.plot(th, recall[1:], label="Recall",linewidth=3)
    plt.title('Precision and recall for different threshold values')
    plt.xlabel('Threshold')
    plt.ylabel('Precision/Recall')
    plt.legend()
    plt.show()

def plot_confusion_matrix(threshold):
    predictions = model.predict(X_test)
    mse = np.mean(np.power(X_test - predictions, 2), axis=1)
    error_df = pd.DataFrame({'reconstruction_error': mse,
                            'true_class': y_test})
    
    y_pred = [1 if e > threshold else 0 for e in error_df.reconstruction_error.values]
    conf_matrix = confusion_matrix(error_df.true_class, y_pred)
    plt.figure(figsize=(12, 12))
    sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d");
    plt.title("Confusion matrix")
    plt.ylabel('True class')
    plt.xlabel('Predicted class')
    plt.show()
