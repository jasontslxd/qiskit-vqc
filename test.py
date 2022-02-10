import os
import numpy as np
import idx2numpy
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from vqc import VQC
from optimiser import Adam

def load_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    print("starting to load data")
    
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    x_train = np.array(idx2numpy.convert_from_file("mnist/train-images.idx3-ubyte"))
    x_test = np.array(idx2numpy.convert_from_file("mnist/t10k-images.idx3-ubyte"))
    y_train = np.array(idx2numpy.convert_from_file("mnist/train-labels.idx1-ubyte"))
    y_test = np.array(idx2numpy.convert_from_file("mnist/t10k-labels.idx1-ubyte"))

    print("loaded data")

    digits_two_category = np.array([1,9])
    train_indices = np.flatnonzero(np.isin(y_train, digits_two_category))
    test_indices = np.flatnonzero(np.isin(y_test, digits_two_category))

    x_train_two_category = x_train[train_indices]
    y_train_two_category = y_train[train_indices]
    vf = np.vectorize(lambda x : 0 if x == 1 else 1)
    y_train_mapped = vf(y_train_two_category)
    x_test_two_category = x_test[test_indices]
    y_test_two_category = y_test[test_indices]
    y_test_mapped = vf(y_test_two_category)

    print("tensorflow now")

    x_train_tensor = tf.reshape(x_train_two_category, [x_train_two_category.shape[0], 28, 28, 1])
    x_train_resized = tf.image.resize(x_train_tensor, [16,16]).numpy()
    x_train_flat = np.reshape(x_train_resized, (x_train_resized.shape[0], 
                              x_train_resized.shape[1] * x_train_resized.shape[2]))
    x_test_tensor = tf.reshape(x_test_two_category, [x_test_two_category.shape[0], 28, 28, 1])
    x_test_resized = tf.image.resize(x_test_tensor, [16,16]).numpy()
    x_test_flat = np.reshape(x_test_resized, (x_test_resized.shape[0], 
                             x_test_resized.shape[1] * x_test_resized.shape[2]))

    x_train_normalised = normalize(x_train_flat)
    x_test_normalised = normalize(x_test_flat)

    x_train_final, x_val_final, y_train_final, y_val_final = train_test_split(x_train_normalised, 
    y_train_mapped, test_size=0.0833)

    print("preprocessed data")
    
    return x_train_final, x_val_final, x_test_normalised, y_train_final, y_val_final, y_test_mapped

def main():
    x_train, x_val, x_test, y_train, y_val, y_test = load_data()
    vqc = VQC(Adam(), n_layers=10)
    vqc.fit(x_train[:4], y_train[:4], epochs=5)

if __name__ == "__main__":
    main()