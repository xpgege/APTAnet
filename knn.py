import os.path
from typing import Iterable, List

import numpy as np
from Levenshtein import distance as levenshtein_distance


def knn(
    train_epitopes: List[str],
    train_tcrs: List[str],
    train_labels: List[int],
    test_epitopes: List[str],
    test_tcrs: List[str],
    k: int = 1,
    return_knn_labels: bool = False,
):
    """Baseline model for epitope-TCR binding prediction. Applies KNN classification
    using as similarity the length-normalized Levensthein distance between epitopes and
    TCRs. Predictions conceptually correspond to the predict_proba method of
    sklearn.neighbors.KNeighborsClassifier.

    Args:
        train_epitopes (List[str]): List of AA sequences of training epitope samples.
            Length should be identical to train_tcrs and train_labels.
        train_tcrs (List[str]): List of AA sequences of training TCR samples. Length
            should be identical to train_epitopes and train_labels.
        train_labels (List[int]): List of training labels. Length should be identical
            to train_tcrs and train_labels.
        test_epitopes (List[str]): List of AA sequences of test epitope samples. Length
            should be identical to test_tcrs.
        test_tcrs (List[str]): List of AA sequences of test TCR samples. Length should
            be identical to test_epitopes
        k (int, optional): Hyperparameter for KNN classification. Defaults to 1.
        return_knn_labels (bool, optional): If set, the labels of the K nearest
            neighbors are also returned.
    """
    assert isinstance(train_epitopes, Iterable)
    assert isinstance(train_tcrs, Iterable)
    assert isinstance(train_labels, Iterable)
    assert isinstance(test_epitopes, Iterable)
    assert isinstance(test_tcrs, Iterable)

    assert len(test_tcrs) == len(test_epitopes), 'Test data lengths dont match'
    assert len(train_epitopes
               ) == len(train_tcrs), 'Test data lengths dont match'
    assert len(train_epitopes
               ) == len(train_labels), 'Test data lengths dont match'

    predictions, knn_labels = [], []
    for epitope, tcr in zip(test_epitopes, test_tcrs):

        el = len(epitope)
        tl = len(tcr)
        epitope_dists = [
            levenshtein_distance(epitope, e) / el for e in train_epitopes
        ]
        tcr_dists = [levenshtein_distance(tcr, t) / tl for t in train_tcrs]

        knns = np.argsort(np.array(epitope_dists) + np.array(tcr_dists))[:k]
        _knn_labels = np.array(train_labels)[knns]
        predictions.append(np.mean(_knn_labels))
        knn_labels.append(_knn_labels)

    return (predictions, knn_labels) if return_knn_labels else predictions

if __name__=='__main__':
    import pandas as pd
    from sklearn.metrics import (
        auc, average_precision_score, precision_recall_curve, roc_curve
    )
    basedir="/home/xpgege/Documents/TCR/ATPnet/data" #"/home/xp/ATPnet/data/"
    train=pd.read_csv(os.path.join(basedir,"tcr_split/fold0/train+covid.csv"),index_col=0)
    test=pd.read_csv(os.path.join(basedir,"tcr_split/fold0/test+covid.csv"),index_col=0)
    epitpde=pd.read_csv(os.path.join(basedir,"epitopes.csv"), sep="\t", header=None, index_col=1)
    epitpde=epitpde.to_dict()[0]
    tcr=pd.read_csv(os.path.join(basedir,"tcr_cdr3.csv"), sep="\t", header=None, index_col=1)
    tcr=tcr.to_dict()[0]
    train_epitopes=train["ligand_name"].map(epitpde)
    train_tcrs=train["sequence_id"].map(tcr)
    train_labels=train["label"]
    test_epitopes=test["ligand_name"].map(epitpde)
    test_tcrs=test["sequence_id"].map(tcr)
    test_labels=test["label"]
    test_labels=test_labels.to_list()

    auc_list={}
    import seaborn as sns
    import matplotlib.pyplot as plt
    sns.set()
    for k in range(3,20,2):
        predictions=knn(train_epitopes,train_tcrs,train_labels,test_epitopes,test_tcrs,k,False)
        fpr, tpr, _ = roc_curve(test_labels, predictions)
        test_roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label='k='+str(k) + '_AUC = %0.4f' % test_roc_auc)
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.legend()
        auc_list[k]=test_roc_auc

    plt.show()