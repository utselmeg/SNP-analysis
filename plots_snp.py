
import numpy as np
from sklearn.metrics import precision_recall_curve, f1_score
from tensorflow.keras.utils import to_categorical
from matplotlib import pyplot as plt

num_s="top250-8-tile-lr0.00001-optimized-r20-020123-multi-less-augumentation-more-shift"

def plot_learning_rate(a=0.00001, steps=20000, rate_decay=0.4):
    y = []
    x = range(steps)
    for i in range(steps):
        y.append(a * pow(rate_decay, i / 10000))

    plt.figure(figsize=(8.5, 8))
    plt.style.use("classic")
    plt.plot(x,y)
    plt.show()

def plot_accuracy(history, current_path):
    plt.figure(figsize=(8.5, 8))
    plt.style.use("classic")
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(current_path / "accuracy.png", bbox_inches="tight")
    plt.show()

def plot_categorical_crossentropy_loss(history, current_path):
    plt.figure(figsize=(8.5, 8))
    plt.style.use("classic")
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('categorical crossentropy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(current_path / "categorical_crossentropy_loss.png", bbox_inches="tight")
    plt.show()

def plot_f1_scores(y_true, y_pred, current_path, labels=None):
    if len(y_true.shape) == 1:  # if y_true is integer encoded
        y_true = to_categorical(y_true)
    if len(y_pred.shape) == 1:  # if y_pred is integer encoded
        y_pred = to_categorical(y_pred)

    f1_scores = f1_score(y_true.argmax(axis=1), y_pred.argmax(axis=1), average=None)
    plt.bar(range(len(f1_scores)), f1_scores)
    plt.xlabel('Class')
    plt.ylabel('F1 Score')
    plt.title('F1 Scores for Each Class')
    if labels:
        plt.xticks(range(len(f1_scores)), labels)
    plt.savefig(current_path / "f1_scores.png", bbox_inches="tight")
    plt.show()

def plot_precision_recall(y_true, y_pred_proba, current_path, labels=None):
    for i in range(y_pred_proba.shape[1]): # loop over each class
        precision, recall, _ = precision_recall_curve(y_true[:, i], y_pred_proba[:, i])
        label = f'Class {i}' if labels is None else labels[i]
        plt.plot(recall, precision, lw=2, label=label)

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='best')
    plt.grid()
    plt.savefig(current_path / "precision_recall_curve.png", bbox_inches="tight")
    plt.show()

def all_plots(history, y_pred, test_label, current_path, num_s, y_pred_proba=None):
    plot_learning_rate()
    plt.show()
    
    plot_accuracy(history, current_path)
    
    plot_categorical_crossentropy_loss(history, current_path)
    
    if y_pred_proba is not None:
        plot_f1_scores(test_label, y_pred, current_path)
        plot_precision_recall(test_label, y_pred_proba, current_path)
