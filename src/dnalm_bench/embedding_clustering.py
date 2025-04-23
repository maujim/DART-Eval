import os

import h5py
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from sklearn.cluster import *

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

# np.random.seed(0)


class EmbeddingCluster:
    def __init__(self, cluster_obj, embeddings, labels):
        self.cluster_obj = cluster_obj
        self.embeddings = embeddings
        self.true_labels = labels
        self.cluster_obj.fit(self.embeddings)

    def get_cluster_labels(self):
        return self.cluster_obj.labels_

    def get_clustering_score(self, metric):
        cluster_labels = self.get_cluster_labels()
        return metric(self.true_labels, cluster_labels)

    def plot_embeddings(self, reduction_obj, out_path, categories):
        embeddings_2d = reduction_obj.fit_transform(self.embeddings)
        first_dim, second_dim = embeddings_2d[:, 0], embeddings_2d[:, 1]
        plot_labels = [categories[x] for x in self.true_labels]
        plt.figure(dpi=300)
        # colors = {0:'red', 1:'blue', 2:'green', 3:'orange', 4:'purple'}
        cmap = ListedColormap(
            ["tab:red", "tab:blue", "tab:green", "tab:orange", "tab:purple"]
        )
        scatter = plt.scatter(
            first_dim, second_dim, c=self.true_labels, s=0.5, cmap=cmap
        )
        plt.xticks([])
        plt.yticks([])
        plt.title("Model Embeddings Colored by Cell Type")
        # print(scatter.legend_elements()[0])
        # plt.legend(handles=scatter.legend_elements()[0], labels=categories)
        plt.savefig(out_path, format="png")
        plt.show()

    def save_model(self, out_path):
        joblib.dump(self.cluster_obj, out_path)


def load_embeddings_and_labels(embedding_file, label_file):
    """
    Assumes embedding_h5 embeddings for all peaks
    """
    arrays = []
    file = h5py.File(embedding_file, "r")
    cat_list = list(pd.read_csv(label_file, sep="\t")["label"].values)
    cat_set = sorted(list(set(cat_list)))
    labels = [cat_set.index(x) for x in cat_list]
    for key in list(file["seq"].keys()):
        if "idx" in key:
            continue
        split = key.split("_")
        ind_start, ind_end = int(split[-2]), int(split[-1])
        h5_array = file["seq"][key][:]
        if "idx_var" in file["seq"].keys():
            idx_vars = file["seq"]["idx_var"][ind_start:ind_end]
            mins, maxes = idx_vars.min(1), idx_vars.max(1) + 1
            indices = [np.arange(mi, ma) for mi, ma in zip(mins, maxes)]
            curr_means = np.array(
                [
                    np.mean(h5_array[i, indices[i], :], axis=0)
                    for i in range(h5_array.shape[0])
                ]
            )
        elif "idx_fix" in file["seq"].keys():
            idx_fix = file["seq"]["idx_fix"][:]
            indices = np.arange(idx_fix.min(), idx_fix.max() + 1)
            curr_means = np.mean(h5_array[:, indices, :], axis=1)
            # Calculate mean over specified slices for each row
        arrays.append(np.vstack(curr_means))
    stacked_arrays = np.vstack(arrays)
    assert len(stacked_arrays) == len(labels)
    return stacked_arrays, labels, cat_set


def load_embeddings_and_labels_subset(embedding_file, label_file, index_file):
    """
    Assumes embedding_h5 embeddings for all peaks
    """
    arrays = []
    file = h5py.File(embedding_file, "r")
    cat_list = list(pd.read_csv(label_file, sep="\t")["label"].values)
    idx_arr = np.array(pd.read_csv(index_file, index_col=0).index).astype(int)
    cat_set = sorted(list(set(cat_list)))
    labels = [cat_set.index(x) for i, x in enumerate(cat_list) if i in idx_arr]
    for key in list(file["seq"].keys()):
        if "idx" in key:
            continue
        split = key.split("_")
        ind_start, ind_end = int(split[-2]), int(split[-1])
        h5_array = file["seq"][key][:]
        if "idx_var" in file["seq"].keys():
            idx_vars = file["seq"]["idx_var"][ind_start:ind_end]
            mins, maxes = idx_vars.min(1), idx_vars.max(1) + 1
            indices = [np.arange(mi, ma) for mi, ma in zip(mins, maxes)]
            curr_means = np.array(
                [
                    np.mean(h5_array[i, indices[i], :], axis=0)
                    for i in range(h5_array.shape[0])
                ]
            )
        elif "idx_fix" in file["seq"].keys():
            idx_fix = file["seq"]["idx_fix"][:]
            indices = np.arange(idx_fix.min(), idx_fix.max() + 1)
            curr_means = np.mean(h5_array[:, indices, :], axis=1)
            # Calculate mean over specified slices for each row
        arrays.append(np.vstack(curr_means))
    stacked_arrays = np.vstack(arrays)[idx_arr]
    assert len(stacked_arrays) == len(labels)
    return stacked_arrays, labels, cat_set
