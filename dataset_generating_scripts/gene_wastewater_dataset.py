# %%
import argparse
import json
import logging
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

%pylab inline
sys.path.append("..")
from modeling.unified_dataloader import (fill_with_future_observation,
                                         fill_with_last_observation)
from modeling.utils import masked_mse_cal
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from data_processing_utils import (add_const_mask, construct_seq_from_index,
                                   return_constant_mask_indices,
                                   return_seq_indexes, window_truncate)
from kando_utils import remove_faulty_records

# from modeling.utils import setup_logger


DATASET_PATH = "./RawData/Wastewater/883--01_01_18-13_03_21"
FAULTY_READS_PATH = "./RawData/Wastewater/dict_bh_info.json"
TRAIN_SIZE, VAL_SIZE, TEST_SIZE = (0.8, 0.2, 0.2)
RATE = 2
UNIT_DICT = {"EC": "mS", "COD": "$\\frac{mg}{{L}}$", "TEMPERATURE": "$^oC$"}

# if __name__ == "__main__":
# parser = argparse.ArgumentParser(description="Generate wastewater dataset")
# parser.add_argument("--file_path", help="path of dataset file", type=str)
# parser.add_argument(
#     "--rate", help="upsampling rate", type=int, default=2,
# )
# parser.add_argument("--seq_len", help="sequence length", type=int, default=96)
# parser.add_argument(
#     "--dataset_name",
#     help="name of generated dataset, will be the name of saving dir",
#     type=str,
#     default="test",
# )
# parser.add_argument(
#     "--saving_path", type=str, help="parent dir of generated dataset", default="."
# )
# args = parser.parse_args()

# dataset_saving_dir = os.path.join(args.saving_path, args.dataset_name)
# data_list = kando_utils.retrieve_data(POINT_IDS, secret, start_date, end_date)

# logger = setup_logger(
#     os.path.join(dataset_saving_dir + "/dataset_generating.log"),
#     "Generate UCI electricity dataset",
#     mode="w",
# )
# logger.info(args)

df = pd.read_csv(DATASET_PATH, parse_dates=True, index_col="DateTime")

with open(FAULTY_READS_PATH) as f:
    faulty_dict = json.load(f)


df_no_nan = df.dropna()
df_clean = remove_faulty_records(df_no_nan, faulty_dict, 883, 1)

seq_index_arr = return_seq_indexes(df_clean, seq_len=96, win_len=20, verbose=2)

train_set_indexes, test_set_indexes = train_test_split(
    seq_index_arr, test_size=TEST_SIZE, random_state=42
)
train_set_indexes, val_set_indexes = train_test_split(
    train_set_indexes, test_size=VAL_SIZE
)

print(
    "Train set size:",
    round(len(train_set_indexes) / len(seq_index_arr), 2) * 100,
    "%" "\nVal set size:",
    round(len(val_set_indexes) / len(seq_index_arr), 2) * 100,
    "%" "\nTest set size:",
    round(len(test_set_indexes) / len(seq_index_arr), 2) * 100,
    "%",
)

scaler = StandardScaler()
df_all = construct_seq_from_index(df_clean, seq_index_arr, seq_len=96, res=15)
df_train = construct_seq_from_index(df_clean, train_set_indexes, seq_len=96, res=15)
df_val = construct_seq_from_index(df_clean, val_set_indexes, seq_len=96, res=15)
df_test = construct_seq_from_index(df_clean, test_set_indexes, seq_len=96, res=15)

train_set_X_full = scaler.fit_transform(df_train)
val_set_X_full = scaler.transform(df_val)
test_set_X_full = scaler.transform(df_test)
train_set_X = window_truncate(train_set_X_full, seq_len=96)
val_set_X = window_truncate(val_set_X_full, seq_len=96)
test_set_X = window_truncate(test_set_X_full, seq_len=96)

# for deterministic approaches 
test_set_X_det = window_truncate(df_test.to_numpy(), seq_len=96)
X = test_set_X_det
data_dict=add_const_mask(X, 2, 'test', mode="COD")
# %%

# indices_for_holdout = return_constant_mask_indices(y.shape[0], 2)
# X_hat_flat = np.copy(y_flat)
# y_hat[indices_for_holdout] = np.nan  # X_hat contains artificial missing values
# missing_mask = (~np.isnan(y_hat)).astype(np.float32)
# indicating_mask contains masks indicating artificial missing values
# indicating_mask = ((~np.isnan(X_hat_flat)) ^ (~np.isnan(y_flat))).astype(
    # np.float32
# )

# X_hat = X_hat_flat.reshape(y.shape)
# %%
y_hat = fill_with_future_observation(np.transpose(data_dict['X_hat'],axes=[0,2,1]))
# %%
y_hat = np.transpose(y_hat,axes=[0,2,1])
# indicating_mask = indicating_mask_flat.reshape(y.shape)


# %%
input = torch.from_numpy(y_hat.copy())
target = torch.from_numpy(X.copy())
mask = torch.from_numpy(data_dict['indicating_mask'].copy())
mse = masked_mse_cal(input, target, mask)

# %%
df_reconstructed = df = pd.DataFrame(
    y_hat, columns=df_all.columns, index=df_clean.index
)

# %%
fig, ax = plt.subplots(4, figsize=(15, 20))
fig.suptitle(f"Reconstructed plot - future fill", fontsize=25)
for index, feature in enumerate(df.columns):
    ax[index].set_title(f"{feature}(t)", fontsize=15)
    ax[index].set_xlabel("t")
    if feature in UNIT_DICT:
        ax[index].set_ylabel(r"{}".format(UNIT_DICT[feature]))
    sns.lineplot(
        ax=ax[index],
        x=df_reconstructed.index,
        y=df_reconstructed.columns[index],
        data=df_reconstructed,
    )
    fig.tight_layout()
    fig.subplots_adjust(top=0.94)

# %%