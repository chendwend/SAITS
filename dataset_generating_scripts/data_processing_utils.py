import os

import h5py
import numpy as np
import pandas as pd


def verify_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def return_seq_indexes(df, seq_len=96, win_len=96, res=15, verbose=0):
    """locates sequences in the data and retuns list of indexes where each sequence starts"""

    index_arr = []
    count = 0

    diff = df.dropna().index.to_series().diff()[1:]
    diff_minutes = diff / np.timedelta64(1, "m")

    while count < len(diff_minutes):
        try:
            if (diff_minutes[count : count + seq_len - 1] == res).all():
                index_arr.append(diff_minutes.index[count])
                count += win_len
            else:
                count += 1
        except IndexError:
            break

    if verbose == 1:
        print(f"Found {len(index_arr)} sequences.")
    elif verbose == 2:
        print(
            f"Found {len(index_arr)} sequences with sliding window = {win_len}"
            f" and sequence length = {seq_len}."
        )

    return index_arr


def construct_seq_from_index(df, seq_index_arr, seq_len=96, res=15):

    sample_collector = []

    for idx in seq_index_arr:
        idx_set = pd.date_range(start=idx, periods=seq_len, freq=f"{res}T")
        sample_collector.append(df[df.index.isin(idx_set)])
    df_sef = pd.concat(sample_collector)

    return df_sef


def add_const_mask(X, rate, set_name, mode="COD"):

    sample_num, seq_len, feature_num = X.shape
    if set_name == "train":
        # if this is train set, we don't need add artificial missing values right now.
        # If we want to apply MIT during training, dataloader will randomly mask out some values to generate X_hat

        # calculate empirical mean for model GRU-D, refer to paper
        mask = (~np.isnan(X)).astype(np.float32)
        X_filledWith0 = np.nan_to_num(X)
        empirical_mean_for_GRUD = np.sum(mask * X_filledWith0, axis=(0, 1)) / np.sum(
            mask, axis=(0, 1)
        )
        data_dict = {
            "X": X[:1024,],
            "empirical_mean_for_GRUD": empirical_mean_for_GRUD,
        }
    else:
        # if this is val/test set, then we need to add artificial missing values right now,
        # because we need they are fixed
        indices_for_holdout = return_constant_mask_indices(X.shape[1], rate)
        # X = X.reshape(-1)
        X_hat = np.copy(X)  # X_hat contains artificial missing values
        if mode == "COD":
            X_hat[:, indices_for_holdout, 0] = np.nan
        else:
            X_hat[:, indices_for_holdout, :] = np.nan
        missing_mask = (~np.isnan(X_hat)).astype(np.float32)
        # indicating_mask contains masks indicating artificial missing values
        indicating_mask = ((~np.isnan(X_hat)) ^ (~np.isnan(X))).astype(np.float32)

        data_dict = {
            "X": X[:256,],
            "X_hat": X_hat[:256,],
            "missing_mask": missing_mask[:256,],
            "indicating_mask": indicating_mask[:256,],
        }

    return data_dict


def window_truncate(feature_vectors, seq_len):
    """ Generate time series samples, truncating windows from time-series data with a given sequence length.
    Parameters
    ----------
    feature_vectors: time series data, len(shape)=2, [total_length, feature_num]
    seq_len: sequence length
    """
    start_indices = np.asarray(range(feature_vectors.shape[0] // seq_len)) * seq_len
    sample_collector = []
    for idx in start_indices:
        sample_collector.append(feature_vectors[idx : idx + seq_len])

    return np.asarray(sample_collector).astype("float32")


def random_mask(vector, artificial_missing_rate):
    """generate indices for random mask"""
    assert len(vector.shape) == 1
    indices = np.where(~np.isnan(vector))[0].tolist()
    indices = np.random.choice(indices, int(len(indices) * artificial_missing_rate))
    return indices


def return_constant_mask_indices(length, rate):
    observed_indices = list(range(0, length, rate))
    missing_indices = list(set(range(length)) - set(observed_indices))
    return missing_indices


def add_artificial_mask(X, artificial_missing_rate, set_name):
    """ Add artificial missing values.
    Parameters
    ----------
    X: feature vectors
    artificial_missing_rate: rate of artificial missing values that are going to be create
    set_name: dataset name, train/val/test
    """
    sample_num, seq_len, feature_num = X.shape
    if set_name == "train":
        # if this is train set, we don't need add artificial missing values right now.
        # If we want to apply MIT during training, dataloader will randomly mask out some values to generate X_hat

        # calculate empirical mean for model GRU-D, refer to paper
        mask = (~np.isnan(X)).astype(np.float32)
        X_filledWith0 = np.nan_to_num(X)
        empirical_mean_for_GRUD = np.sum(mask * X_filledWith0, axis=(0, 1)) / np.sum(
            mask, axis=(0, 1)
        )
        data_dict = {
            "X": X[:1024,],
            "empirical_mean_for_GRUD": empirical_mean_for_GRUD,
        }
    else:
        # if this is val/test set, then we need to add artificial missing values right now,
        # because we need they are fixed
        X = X.reshape(-1)
        indices_for_holdout = random_mask(X, artificial_missing_rate)
        X_hat = np.copy(X)
        X_hat[indices_for_holdout] = np.nan  # X_hat contains artificial missing values
        missing_mask = (~np.isnan(X_hat)).astype(np.float32)
        # indicating_mask contains masks indicating artificial missing values
        indicating_mask = ((~np.isnan(X_hat)) ^ (~np.isnan(X))).astype(np.float32)

        data_dict = {
            "X": X.reshape([sample_num, seq_len, feature_num]),
            "X_hat": X_hat.reshape([sample_num, seq_len, feature_num]),
            "missing_mask": missing_mask.reshape([sample_num, seq_len, feature_num]),
            "indicating_mask": indicating_mask.reshape(
                [sample_num, seq_len, feature_num]
            ),
        }

    return data_dict


def saving_into_h5(saving_dir, data_dict, classification_dataset):
    """ Save data into h5 file.
    Parameters
    ----------
    saving_dir: path of saving dir
    data_dict: data dictionary containing train/val/test sets
    classification_dataset: boolean, if this is a classification dataset
    """

    def save_each_set(handle, name, data):
        single_set = handle.create_group(name)
        if classification_dataset:
            single_set.create_dataset("labels", data=data["labels"].astype(int))
        single_set.create_dataset("X", data=data["X"].astype(np.float32))
        if name in ["val", "test"]:
            single_set.create_dataset("X_hat", data=data["X_hat"].astype(np.float32))
            single_set.create_dataset(
                "missing_mask", data=data["missing_mask"].astype(np.float32)
            )
            single_set.create_dataset(
                "indicating_mask", data=data["indicating_mask"].astype(np.float32)
            )

    saving_path = os.path.join(saving_dir, "datasets.h5")
    with h5py.File(saving_path, "w") as hf:
        hf.create_dataset(
            "empirical_mean_for_GRUD",
            data=data_dict["train"]["empirical_mean_for_GRUD"],
        )
        save_each_set(hf, "train", data_dict["train"])
        save_each_set(hf, "val", data_dict["val"])
        save_each_set(hf, "test", data_dict["test"])
