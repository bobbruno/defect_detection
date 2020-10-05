import numpy as np
import logging
import argparse
import os
from pathlib import Path
from tensorflow.keras import layers, Input, models
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier 
from sklearn.model_selection import KFold 
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split


def parse_arguments():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--num-epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--model_dir", type=str, default=None)
    parser.add_argument('--model-save-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    return parser.parse_args()


def create_classifier(x_train, x_test, y_train, y_test, epochs, batch_size):
    def create_model():
        input_shape = (26, 26, 3)
        input_tensor = Input(input_shape)

        conv_1 = layers.Conv2D(16, (3,3), activation='relu', padding='same')(input_tensor)
        conv_2 = layers.Conv2D(64, (3,3), activation='relu', padding='same')(conv_1)
        conv_3 = layers.Conv2D(128, (3,3), activation='relu', padding='same')(conv_2)

        flat = layers.Flatten()(conv_3)

        dense_1 = layers.Dense(512, activation='relu')(flat)
        dense_2 = layers.Dense(128, activation='relu')(dense_1)
        output_tensor = layers.Dense(9, activation='softmax')(dense_2)

        model = models.Model(input_tensor, output_tensor)
        model.compile(optimizer='Adam',
                     loss='categorical_crossentropy',
                     metrics=['accuracy'])
        return model
    logging.info("Running Cross-Validation")
    model = KerasClassifier(build_fn=create_model, epochs=10, batch_size=1024, verbose=2)
    kfold = KFold(n_splits=3, shuffle=True, random_state=42)
    results = cross_val_score(model, x_train, y_train, cv=kfold)
    logging.info(f"CV Score: {np.mean(results)}")
    logging.info("Training model")
    model = create_model()
    model.fit(x_train, y_train,
        validation_data=[x_test, y_test],
        epochs=epochs,
        batch_size=batch_size,
    )
    return model


def save_model(model, path, file_name):
    dest_path = str(Path(path) / file_name)
    logging.info(f"Saving model at {dest_path}")
    model.save(dest_path)


def load_data(path, file_name='data.npz'):
    file_path = Path(path) / file_name
    with np.load(str(file_path), allow_pickle=True) as data:
        x = data['x']
        y = data['y']
        label_classes = data['label_classes'].item(0)
    logging.info(f"x shape: {x.shape}")
    logging.info(f"y shape: {y.shape}")
    aug_y = to_categorical([label_classes[label] for label in y])
    x_train, x_test, y_train, y_test = train_test_split(x, aug_y, test_size=0.33, random_state=42)
    logging.info(f"x train shape: {x_train.shape}")
    logging.info(f"y train shape: {y_train.shape}")
    logging.info(f"x test shape: {x_test.shape}")
    logging.info(f"y test shape: {y_test.shape}")
    return x_train, x_test, y_train, y_test


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = parse_arguments()
    x_train, x_test, y_train, y_test = load_data(args.train, "data.npz")
    classifier = create_classifier(x_train, x_test, y_train, y_test, args.num_epochs, args.batch_size)
    save_model(classifier, args.model_save_dir, '1')
