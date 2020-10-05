import os
import sagemaker
import numpy as np
import argparse
from tensorflow.keras import layers, Input, models
from pathlib import Path
import logging


def parse_arguments():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--num-epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--model_dir", type=str, default=None)
    parser.add_argument('--model-save-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
#    parser.add_argument('--test', type=str, default=os.environ['SM_CHANNEL_TEST'])
    return parser.parse_args()

    
def create_autoencoder(data, epochs, batch_size):
    input_shape = (26, 26, 3)
    input_tensor = Input(input_shape)
    encode = layers.Conv2D(64, (3,3), padding='same', activation='relu')(input_tensor)

    latent_vector = layers.MaxPool2D()(encode)

    # Decoder
    decode_layer_1 = layers.Conv2DTranspose(64, (3,3), padding='same', activation='relu')
    decode_layer_2 = layers.UpSampling2D()
    output_tensor = layers.Conv2DTranspose(3, (3,3), padding='same', activation='sigmoid')

    # connect decoder layers
    decode = decode_layer_1(latent_vector)
    decode = decode_layer_2(decode)

    ae = models.Model(input_tensor, output_tensor(decode))
    ae.compile(
        optimizer = 'Adam',
        loss = 'mse',
    )
    ae.fit(data, data,
        batch_size=batch_size,
        epochs=epochs,
        verbose=2)
    
    decoder_input = Input((13, 13, 64))
    decoder = decode_layer_1(decoder_input)
    decoder = decode_layer_2(decoder)
    
    return ae, models.Model(input_tensor, latent_vector), models.Model(decoder_input, output_tensor(decoder))


def load_data(path: str, file_name: str="data.npz", limit: int=None):
    file_path = Path(path) / file_name
    with np.load(str(file_path), allow_pickle=True) as data:
        x = data['x']
        y = data['y']
    print(np.unique(y, return_counts=True))
    if limit:
        return (x[:limit], y[:limit])
    else:
        return (x, y)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = parse_arguments()
    x, y = load_data(args.train, "data.npz", limit=args.max_rows)
    ae, encoder, decoder = create_autoencoder(x, epochs=args.num_epochs, batch_size=args.batch_size)
    logging.info(f"Autoencoder: {ae.summary()}")
    logging.info(f"Encoder: {encoder.summary()}")
    logging.info(f"Dncoder: {decoder.summary()}")
    ae.save(str(Path(args.model_save_dir) / "ae.h5" ))
    encoder.save(str(Path(args.model_save_dir) / "encoder.h5" ))
    decoder.save(str(Path(args.model_save_dir) / "decoder.h5" ))
