import numpy as np
import tarfile
import argparse
import logging
from pathlib import Path
from tensorflow.keras.models import load_model


def parse_arguments():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--augmented-size", type=int, default=2000)
    return parser.parse_args()


def load_models(model_path: str):
    # Add tar decompression here
    model_tar = Path(model_path) / "model.tar.gz"
    logging.info(f"Loading models from {model_tar}")
    tf_file = tarfile.open(str(model_tar), mode="r:gz")
    dest_dir = Path("/tmp/models")
    dest_dir.mkdir()
    tf_file.extractall(path=str(dest_dir))
    tf_file.close()
    encoder = load_model(Path(dest_dir) / "encoder.h5", compile=False)
    logging.info(f"Encoder loaded: {encoder.summary()}")
    decoder = load_model(Path(dest_dir) / "decoder.h5", compile=False)
    logging.info(f"Decoder loaded: {decoder.summary()}")
    return encoder, decoder


def load_data(path: str, file_name: str="data.npz"):
    file_path = Path(path) / file_name
    with np.load(str(file_path), allow_pickle=True) as data:
        x = data['x']
        y = data['y']
        label_classes = data['label_classes'].item(0)
    logging.info(f"x shape: {x.shape}")
    logging.info(f"y shape: {y.shape}")
    return (x, y, label_classes)
    
    
def generate_augmented_data(wafers, label, encoder, decoder, n_examples):
    # Encode input wafer
    logging.info(f"There are {len(wafers)} examples for {label}")
    encoded_x = encoder.predict(wafers)
    logging.info(f"Shape of encoded_x: {encoded_x.shape}")
    
    additional_example_batches = n_examples // wafers.shape[0] + 1
    logging.info(f"Generating {additional_example_batches} batches of examples for label {label}")
    temp_noised = []
    for i in range(additional_example_batches):
        temp_noised.append(encoded_x + np.random.normal(loc=0, scale=0.1, size = (len(encoded_x), 13, 13, 64)))
    noised_encoded_x = np.vstack(temp_noised)
    gen_x = decoder.predict(noised_encoded_x)
    # also make label vector with same length
    gen_y = np.full((len(gen_x), ), label)

    logging.info(f"Returning {n_examples - len(wafers)} new examples") 
    return gen_x, gen_y


def augment(x, y, labels, encoder, decoder, n_examples):
    aug_x = x.copy()
    aug_y = y.copy()
    for l in labels: 
        # skip none case
        if l in {'none', 'unknown'} : 
            continue
        else:
            logging.info(f'Generating {l}')

        gen_x, gen_y = generate_augmented_data(x[np.where(y==l)], l, encoder, decoder, n_examples)
        aug_x = np.concatenate((aug_x, gen_x), axis=0)
        aug_y = np.concatenate((aug_y, gen_y))
    return aug_x, aug_y


def save_augmented(x, y, label_classes, output_path):
    np.savez_compressed(output_path / "data.npz", x=x, y=y, label_classes=label_classes)

                        
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = parse_arguments()
    root_path = Path('/opt/ml/processing')
    model_path = root_path / "models"
    data_path = root_path / "data"
    augmented_path = root_path / "augmented"
    x, y, label_classes = load_data(str(data_path))
    encoder, decoder = load_models(str(model_path))
    x, y = augment(x, y, label_classes, encoder, decoder, args.augmented_size)
    save_augmented(x, y, label_classes, augmented_path)
