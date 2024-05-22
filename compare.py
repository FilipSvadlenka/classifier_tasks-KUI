import numpy as np
from sklearn.metrics import roc_curve


# load classifiers in np.array
def load_data(c_file, size):
    data = [[] for _ in range(size)]
    f = open(c_file, "rt")
    for line in f:
        i = 0
        for char in line:
            if char == '1' or char == "0":
                data[i].append(int(char))
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
    # unknown classifier
    c_file = 'C6.dsv'
    # representing picture for example
    data_sample = 'GT.dsv'
    # number of parameters / columns
    size = 50
    # known values for my best classifier
    my_tpr = 0.46
    my_fpr = 0.0
    my_classifier = "Mine"
    data = load_data(c_file, size)
    true_labels = load_labels(data_sample)

    # finding best classifier
    best_fpr = 1.0
    best_tpr = 1.0
    predictions = data.T
    for i in range(predictions.shape[1]):
        fpr, tpr, thresholds = roc_curve(true_labels, predictions[:, i])
        # have to secure the best (highest) tpr even in case of multiple cases of fpr == 0
        if fpr[1] == 0.0 and tpr[1] > best_tpr:
            best_fpr = fpr[1]
            best_tpr = tpr[1]
        # In our case, this can never happen, but if (my_fpr > 0.0), it can.
        elif best_fpr > fpr[1] and tpr[1] > 0:
            best_fpr = fpr[1]
            best_tpr = tpr[1]
    if (best_fpr < my_fpr) or (best_fpr == my_fpr and best_tpr > my_tpr):
        best_classifier = c_file
        print(f'Agent nám dal lepší klasifikátor pro zamčení dat: {best_classifier} s '
              f'TPR: {best_tpr} pro FPR: {best_fpr}! To vypadá důvěryhodně.')
    else:
        print(f'Můj klasifikátor je lepší, kolega agent mi nejspíš něco tají. Jeho klasifikátor měl '
              f'pouze TPR={best_tpr} pro FPR={best_fpr}')
