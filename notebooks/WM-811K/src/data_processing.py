import sys
import os
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


def load_data(client, input_path: Path, file_name: str="wafers.pkl.gz", n_partitions=32):
    logging.info("Loading data")
#     with open(input_path / file_name, 'rb') as f:
#         temp_df = pickle.load(f, encoding="bytes") 

    temp_df = pd.read_pickle(
                input_path / file_name
            ).astype(
                {"waferIndex": "int32"}
            )
    return client.persist(
        dd.from_pandas(temp_df, npartitions=n_partitions)
    )


def process_data(input_df: dd.DataFrame):
    logging.info("Cleaning data")
    clean_df = input_df.drop('waferIndex', axis=1)
    clean_df['waferMapDim'] = clean_df.waferMap.apply(lambda x: x.shape, meta=pd.Series({'waferMapDim': [(0, 0)]}))
    clean_df = clean_df[clean_df.waferMapDim.apply(lambda x: x[0] == x[1], meta=pd.Series({'x': True}))]
    clean_df['label'] = clean_df.failureType.apply(lambda x: x[0, 0] if (isinstance(x, np.ndarray) and x.shape[0] > 0) else 'unknown', meta=pd.Series({"x": "none"}))
    clean_df = clean_df[clean_df.label != "unknown"]
    return clean_df


def calc_x_y(clean_df: dd.DataFrame):
    logging.info("Calculating x and y")
    x = np.stack(
        clean_df.apply(
            lambda x: x.waferMap.reshape(
                (26, 26, 1)
            ) if x.waferMapDim[0] == 26 else cv2.resize(
                x.waferMap.reshape(x.waferMapDim[0], x.waferMapDim[1]), (26, 26)
            ).reshape(26, 26, 1),
            axis=1,
            meta=pd.Series({'x': [np.zeros((26, 26, 1))]})
        ).compute().values
    )
    x = np.apply_along_axis(hot_encode, axis=1, arr=x.reshape(-1, 26 * 26))
    y = clean_df.label.compute().values
    return x, y


def calc_label_info(clean_df: dd.DataFrame):
    logging.info("Calculating labels and their distribution")
    label_dist = clean_df.groupby('label').size().compute()
    inv_prob_label = {k: v for (k, v) in ((1/(label_dist/label_dist.sum()) )/((1/(label_dist/label_dist.sum()) ).sum())).iteritems()}
    logging.info(f"Labels with their inverse frequency: {inv_prob_label}")
    label_classes = np.array({l: i for (i, l) in enumerate(label_dist.index.values)})
    return inv_prob_label, label_classes


if __name__=='__main__':
    logging.basicConfig(level=logging.INFO)
    logging.info(f"python version {sys.version}")
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
        client = None
        
    input_df = load_data(client, input_path)
    clean_df = process_data(input_df)    
    x, y = calc_x_y(clean_df)
    inv_prob_label, label_classes = calc_label_info(clean_df)

    logging.info(f"x: {x.shape}")
    logging.info(f"y: {y.shape}")
    logging.info(f"Label classes: {label_classes}")
    compressed_file_name = output_path / "data.npz"
    np.savez_compressed(compressed_file_name, x=x, y=y, label_classes=label_classes)
    
    for attempt in range(10):
        if compressed_file_name.exists() and compressed_file_name.is_file():
            logging.info(f"Data saved to {compressed_file_name}")
            break
        else:
            time.sleep(2)
        if attemp >= 5:
            logging.error(f"File {compressed_file_name} not created after {2 * attempt}. Aborting...")
            raise Exception(f"Output file {compressed_file_name} couldn't be found")
    
    if client:
        client.close()
    sys.exit(os.EX_OK)