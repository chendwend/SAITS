# %%
import argparse
import json
import logging
import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

# %pylab inline
sys.path.append("..")
from modeling.unified_dataloader import (fill_with_future_observation,
                                         fill_with_last_observation)
from modeling.utils import masked_mse_cal
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from data_processing_utils import (add_const_mask, construct_seq_from_index,
                                   return_constant_mask_indices,
                                   return_seq_indexes, saving_into_h5,
                                   verify_dir, window_truncate)
from kando_utils import remove_faulty_records

# from modeling.utils import setup_logger

# TODO: move all configuration to .ini file
DATASET_PATH = "./RawData/Wastewater/883--01_01_18-13_03_21"
FAULTY_READS_PATH = "./RawData/Wastewater/dict_bh_info.json"
TRAIN_SIZE, VAL_SIZE, TEST_SIZE = (0.8, 0.2, 0.2)
RATE = 2
SEQ_LEN = 96
RESOLUTION = 15
UPSAMPLING_MODE = "COD"  # "COD" / "full"
SLIDING_WINDOW_SIZE = 20
UNIT_DICT = {"EC": "mS", "COD": "$\\frac{mg}{{L}}$", "TEMPERATURE": "$^oC$"}
WASTEWATER_HOME_DIR = "../generated_datasets/Wastewater_datasets/"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate wastewater dataset")
    parser.add_argument("--file_path", help="path of dataset file", type=str)
    parser.add_argument(
        "--rate", help="upsampling rate", type=int, default=2,
    )
    parser.add_argument("--seq_len", help="sequence length", type=int, default=SEQ_LEN)
    parser.add_argument(
        "--point_ids",
        help="sites identified by point ids as data sources",
        type=str,
        default="883",
    )
    parser.add_argument(
        "--win_len", help="sliding window length", type=int, default=SLIDING_WINDOW_SIZE
    )
    args = parser.parse_args()

    point_id_dir = os.path.join(WASTEWATER_HOME_DIR, args.point_ids)
    dataset_saving_dir = os.path.join(
        point_id_dir, f"rate{args.rate}_seqlen{args.seq_len}_winlen{args.win_len}"
    )

    verify_dir(WASTEWATER_HOME_DIR)
    verify_dir(point_id_dir)
    verify_dir(dataset_saving_dir)

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

    seq_index_arr = return_seq_indexes(
        df_clean, seq_len=SEQ_LEN, win_len=SLIDING_WINDOW_SIZE, verbose=2
    )

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

    # construct dataframes from set indexes
    df_all = construct_seq_from_index(
        df_clean, seq_index_arr, seq_len=SEQ_LEN, res=RESOLUTION
    )
    df_train = construct_seq_from_index(
        df_clean, train_set_indexes, seq_len=SEQ_LEN, res=RESOLUTION
    )
    df_val = construct_seq_from_index(
        df_clean, val_set_indexes, seq_len=SEQ_LEN, res=RESOLUTION
    )
    df_test = construct_seq_from_index(
        df_clean, test_set_indexes, seq_len=SEQ_LEN, res=RESOLUTION
    )

    # standartize wrt train set
    scaler = StandardScaler()
    train_set_X_full = scaler.fit_transform(df_train)
    val_set_X_full = scaler.transform(df_val)
    test_set_X_full = scaler.transform(df_test)

    # construct numpy arrays (#datapoints x SEQ_LEN x #features)
    train_set_X = window_truncate(train_set_X_full, seq_len=SEQ_LEN)
    val_set_X = window_truncate(val_set_X_full, seq_len=SEQ_LEN)
    test_set_X = window_truncate(test_set_X_full, seq_len=SEQ_LEN)

    # for deterministic approaches, use unscaled version
    test_set_X_det = window_truncate(df_test.to_numpy(), seq_len=SEQ_LEN)
    X = test_set_X_det
    test_set_dict_det = add_const_mask(X, RATE, "test", mode=UPSAMPLING_MODE)
    y_hat = fill_with_future_observation(test_set_dict_det["X_hat"])

    input = torch.from_numpy(y_hat.copy())
    target = torch.from_numpy(X.copy())
    mask = torch.from_numpy(test_set_dict_det["indicating_mask"].copy())

    mse_ffill = masked_mse_cal(input, target, mask)
    # %%
    # for SAITS
    train_set_dict = add_const_mask(train_set_X, RATE, "train", mode=UPSAMPLING_MODE)
    val_set_dict = add_const_mask(val_set_X, RATE, "val", mode=UPSAMPLING_MODE)
    test_set_dict = add_const_mask(test_set_X, RATE, "test", mode=UPSAMPLING_MODE)

    processed_data = {
        "train": train_set_dict,
        "val": val_set_dict,
        "test": test_set_dict,
    }
    # %%
    # dataset_saving_dir = (
    #     f"../generated_datasets/Wastewater_seqlen{SEQ_LEN}_win{SLIDING_WINDOW_SIZE}/"
    # )
    # if not os.path.exists(dataset_saving_dir):
    #     os.makedirs(dataset_saving_dir)
    # %%
    saving_into_h5(dataset_saving_dir, processed_data, classification_dataset=False)
    # %%
    # df_reconstructed = df = pd.DataFrame(
    #     y_hat, columns=df_all.columns, index=df_clean.index
    # )

    # %%
    # fig, ax = plt.subplots(4, figsize=(15, 20))
    # fig.suptitle(f"Reconstructed plot - future fill", fontsize=25)
    # for index, feature in enumerate(df.columns):
    #     ax[index].set_title(f"{feature}(t)", fontsize=15)
    #     ax[index].set_xlabel("t")
    #     if feature in UNIT_DICT:
    #         ax[index].set_ylabel(r"{}".format(UNIT_DICT[feature]))
    #     sns.lineplot(
    #         ax=ax[index],
    #         x=df_reconstructed.index,
    #         y=df_reconstructed.columns[index],
    #         data=df_reconstructed,
    #     )
    #     fig.tight_layout()
    #     fig.subplots_adjust(top=0.94)

    # %%
    # TODO: Check all data from pkl
    # with open('serialized.pkl', 'rb') as f:
    #     data = pickle.load(f)
