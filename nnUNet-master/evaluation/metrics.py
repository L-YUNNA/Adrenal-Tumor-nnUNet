import seaborn as sns
from sklearn.metrics import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def cal_perf(true, pred, eps=1e-10):  # only binary
    acc, pre, rec, f1 = accuracy_score(true, pred), precision_score(true, pred), recall_score(true, pred), f1_score(true, pred)
    tn, fp, fn, tp = confusion_matrix(true, pred).flatten()
    spec = (tn + eps) / ((fp + tn) + eps)
    ppv = (tp + eps) / ((tp + fp) + eps)
    npv = (tn + eps) / ((tn + fn) + eps)
    perf = [acc, pre, rec, spec, f1, ppv, npv]
    return perf


def get_cm(y_true, y_pred, save_path, label_name: list):   # labels = ['LM', 'GM', 'SSc']
    num_cls = len(np.unique(y_true))
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(5.5, 5))

    group_counts = ["{0:0.0f}".format(value) for value in cm.flatten()]
    group_percent = ["{0:.2%}".format(value) for value in (cm.flatten() / np.sum(cm))]
    labels = [f"{v1}\n\n({v2})" for v1, v2 in zip(group_counts, group_percent)]
    labels = np.asarray(labels).reshape(num_cls, num_cls)

    # cm/np.sum(cm))*70 여기서 곱하는 값을 바꿔서 색 조절
    f = sns.heatmap((cm / np.sum(cm)) * 70, annot=labels, fmt='',
                    cmap='Blues', vmin=0, vmax=25, linewidths=0.1,
                    annot_kws={'size': '12'}, cbar_kws={'label': '(%)'})  # annot=True, fmt='.2%'

    fig = f.figure
    cbar = fig.get_children()[-1]
    cbar.yaxis.set_ticks([0, 25])

    labels = label_name
    f.set_xticklabels(labels, fontdict={'size': '12'})
    f.set_yticklabels(labels, fontdict={'size': '12'})

    f.set(xlabel='Predicted label', ylabel='True label')
    f.axhline(y=0, color='k', linewidth=1)
    f.axhline(y=num_cls, color='k', linewidth=2)
    f.axvline(x=0, color='k', linewidth=1)
    f.axvline(x=num_cls, color='k', linewidth=2)

    plt.title("Confusion Matrix", fontsize=18, y=1.02)
    plt.xlabel('Predicted label', fontsize=12, labelpad=15)
    plt.ylabel('True label', fontsize=12, labelpad=14)

    plt.tight_layout()
    plt.savefig(save_path)