import numpy as np
from sklearn.metrics import roc_curve


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
    best_tpr = 1.0
    best_fpr = 1.0
    best_classifier = None
    c_n = None
    best_aplha = None
    for n, classifier in enumerate(c_fileset):
        predictions = data[n].T
        for i in range(predictions.shape[1]):
            fpr, tpr, thresholds = roc_curve(true_labels, predictions[:, i])
            # have to secure the best (highest) tpr even in case of multiple cases of fpr == 0
            if fpr[1] == 0.0 and tpr[1] > best_tpr:
                best_fpr = fpr[1]
                best_tpr = tpr[1]
                best_classifier = classifier
                best_aplha = i
            # just finding lowest fpr for non-zero tpr
            elif best_fpr > fpr[1] and tpr[1] > 0:
                best_fpr = fpr[1]
                best_tpr = tpr[1]
                best_classifier = classifier
                best_aplha = i

    print(f'Nejlepší klasifikátor pro zamčení  dat je: {best_classifier} s parametrem {best_aplha}. '
          f'TPR = {best_tpr} pro FPR = {best_fpr}')

