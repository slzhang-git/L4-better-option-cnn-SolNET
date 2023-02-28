import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dense, Flatten, BatchNormalization, Dropout
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.utils import plot_model
from tensorflow.keras.losses import binary_crossentropy
from sklearn.model_selection import cross_val_score
from model import SolNet
import os  # added
import matplotlib.pyplot as plt  # added
from tensorflow.keras.callbacks import CSVLogger   # added
import time


def train():
    batch_size = 32
    location = os.path.realpath(os.getcwd() + '/../dataset/')  # location = 'dataset/'
    label_mode = 'binary'
    seed = 10  # changed for each fold made manually
    epochs = 600  # 30
    class_names = ['clean', 'dirty']
    in_size = [227, 227, 3]  # added

    tr_dataset = image_dataset_from_directory(directory=location, label_mode=label_mode, class_names=class_names,
                                              seed=seed, labels='inferred', image_size=in_size[:-1],
                                              subset='training', batch_size=batch_size, validation_split=.2)

    val_dataset = image_dataset_from_directory(directory=location, label_mode=label_mode, class_names=class_names,
                                               seed=seed, labels='inferred', image_size=in_size[:-1],
                                               subset='validation', batch_size=batch_size, validation_split=.2)

    #in_size = [227, 227, 3]
    SolNet1 = SolNet(in_size)
    csv_logger = CSVLogger(str(epochs) + '-epoch-training.log', separator=',', append=False) # added
    #history = SolNet1.fit(tr_dataset, validation_data=val_dataset, epochs=epochs, batch_size=batch_size)
    history = SolNet1.fit(tr_dataset, validation_data=val_dataset, epochs=epochs, batch_size=batch_size, callbacks=[csv_logger])
    return history


if __name__ == "__main__":
    start_time = time.time()        # added
    history = train()  # revised #   train()
    run_time = time.time() - start_time
    plt.plot(history.history['acc'])  # added
    plt.plot(history.history['val_acc'])  # added
    plt.title('accuracy vs epoch. Running time: ' + str(run_time) + ' seconds')  # added
    plt.xlabel('epoch')  # added
    plt.legend(['train accuracy', 'validation accuracy'], loc='upper left')  # added
    plt.show()  # added
