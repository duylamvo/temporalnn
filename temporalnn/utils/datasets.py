"""Datasets Tools"""
import os
import tqdm
import numpy as np
import multiprocessing as mp


def test_gen_data():
    import pandas as pd
    ds = pd.read_csv("data/climate/preprocessed_climate.csv")
    ds.measure_date = pd.to_datetime(ds.measure_date)
    ds = ds.set_index("measure_date")
    # sub_ds = ds[ds.index.year > 2016]
    sub_ds = ds
    x_train, y_train = gen_train(sub_ds, "tmk", "tmk", "stations_id",
                                 x_steps=32,
                                 save=True, save_dir="data/train/uts")
