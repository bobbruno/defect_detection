
import sys
import logging
import dask.dataframe as dd
import pandas as pd
import numpy as np
import cv2
from pathlib import Path
from dask.distributed import Client


def hot_encode(img_arr):
    new_arr = np.zeros((676, 3))
    for x in range(676):
        new_arr[x, img_arr[x]] = 1
    return new_arr.reshape((26, 26, 3))


if __name__=='__main__':
    scheduler_ip = sys.argv[-1]
    root_path = Path('/opt/ml/processing')
    input_path = root_path / 'input'
    output_path = root_path / 'train'
    # Start the Dask cluster client
    try:
        client = Client(f"tcp://{scheduler_ip}:8786")
        logging.info(f"Printing cluster information: {client}")
    except Exception as err:
        logging.exception(err)
        
    logging.info("Loading data")
    input_df = dd.from_pandas(
        pd.read_pickle(
            input_path / "LSWMD.pkl"
        ).astype(
            {"waferIndex": "int32"}
        ),
        npartitions=100)

    logging.info("Cleaning data")
    clean_df = input_df.drop('waferIndex', axis=1)
    clean_df['waferMapDim'] = clean_df.waferMap.apply(lambda x: x.shape, meta=pd.Series({'waferMapDim': [(0, 0)]}))
    clean_df = clean_df[clean_df.waferMapDim.apply(lambda x: x[0] == x[1], meta=pd.Series({'x': True}))]
    clean_df['label'] = clean_df.failureType.apply(lambda x: x[0, 0] if (isinstance(x, np.ndarray) and x.shape[0] > 0) else 'unknown', meta=pd.Series({"x": "none"}))
    clean_df = clean_df[clean_df.label != "unknown"]
    
    label_dist = clean_df.groupby('label').size().compute()
    inv_prob_label = {k: v for (k, v) in ((1/(label_dist/label_dist.sum()) )/((1/(label_dist/label_dist.sum()) ).sum())).iteritems()}
    logging.info(f"Labels: {inv_prob_label}")
    
    x = np.stack(
        clean_df.apply(
            lambda x: x.waferMap.reshape(
                (26, 26, 1)
            ) if x.waferMapDim[0] == 26 else cv2.resize(
                x.waferMap.reshape(x.waferMapDim[0], x.waferMapDim[1]), (26, 26)
            ).reshape(26, 26, 1), axis=1, meta=pd.Series({'x': [np.zeros((26, 26, 1))]})).compute().values
    )
    x = np.apply_along_axis(hot_encode, axis=1, arr=x.reshape(-1, 26 * 26))
    y = clean_df.label.compute().values

    logging.info(f"x: {x.shape}")
    logging.info(f"y: {y.shape}")
    np.savez_compressed(output_path / "data.npz", x, y)
    