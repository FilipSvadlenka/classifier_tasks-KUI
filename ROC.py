import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score


# draw final ROC curve
def draw_ROC(true_labels, best_predictions, abs_best_auc):
    fpr, tpr, thresholds = roc_curve(true_labels, best_predictions)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % abs_best_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()


# load classifiers in np.array
def load_data(c_fileset, size):
    data = [[[] for _ in range(size)] for b in range(len(c_fileset))]
    for c, classifier in enumerate(c_fileset):
        f = open(classifier, "rt")
        for line in f:
            i = 0
            for char in line:
                if char == '1' or char == "0":
                    data[c][i].append(int(char))
                    i += 1
        f.close()
    return np.array(data)


def load_labels(data_sample):
    true_labels = []
    f = open(data_sample, "rt")
    for i, line in enumerate(f):
        for char in line:
            if char == '1' or char == "0":
                true_labels.append(int(char))
    f.close()
    return np.array(true_labels)

if __name__ == '__main__':
    # individual file names with trained classifier
    c_fileset = ('C1.dsv', 'C2.dsv', 'C3.dsv', 'C4.dsv', 'C5.dsv')
    # representing picture for example
    data_sample = 'GT.dsv'
    # number of parameters / columns
    size = 50
    data = load_data(c_fileset, size)
    true_labels = load_labels(data_sample)
    # finding best classifier
    abs_best_auc = 0
    abs_best_column = 0
    best_c = ' '
    best_predictions = []
    best_classifier = None
    for n, classifier in enumerate(c_fileset):
        predictions = data[n].T
        best_auc = 0
        best_column = 0
        for i in range(predictions.shape[1]):
            auc = roc_auc_score(true_labels, predictions[:, i])
            if auc > best_auc:
                best_auc = auc
                best_column = i
        if best_auc > abs_best_auc:
            abs_best_auc = best_auc
            abs_best_column = best_column
            best_predictions = predictions[:, abs_best_column]
            best_classifier = classifier

    print(f'Nejlepší klasifikátor je: {best_classifier} s AUC: {abs_best_auc} a nejlepší parametr A{abs_best_column}')
    draw_ROC(true_labels, best_predictions, abs_best_auc)