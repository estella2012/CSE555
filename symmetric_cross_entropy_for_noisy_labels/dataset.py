"""
Reference: "Symmetric cross entropy for robust learning with noisy labels" 
Updated: Lipisha Chaudhary (lipishan@buffalo.edu)
Presented: Enjamamul Hoq (ehoq@buffalo.edu)
"""

import os
import numpy as np
from keras.datasets import mnist, cifar10, cifar100
from keras.utils import np_utils
from util import other_class
from numpy.testing import assert_array_almost_equal

# Set random seed
np.random.seed(123)

NUM_CLASSES = {'mnist': 10, 'svhn': 10, 'cifar-10': 10, 'cifar-100': 100}


def get_data(dataset='mnist', noise_ratio=0, asym=False, random_shuffle=False):
    """
    Get training images with specified ratio of syn/ayn label noise
    """
    if dataset == 'mnist':
        (X_train, y_train), (X_test, y_test) = mnist.load_data()

        X_train = X_train.reshape(-1, 28, 28, 1)
        X_test = X_test.reshape(-1, 28, 28, 1)

        X_train = X_train / 255.0
        X_test = X_test / 255.0
    else:
        return None, None, None, None


    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    y_train_clean = np.copy(y_train)

    ROOT_PATH = "D:\Drive F\PHD\Research\Code\symmetric_cross_entropy_for_noisy_labels\data1"
    if not os.path.exists(ROOT_PATH):
        os.makedirs(ROOT_PATH)
    # generate random noisy labels
    if noise_ratio > 0:
        if asym:
            data_file = "asym_%s_train_labels_%s.npy" % (dataset, noise_ratio)
            if dataset == 'cifar-100':
                P_file = "asym_%s_P_value_%s.npy" % (dataset, noise_ratio)
        else:
            data_file = "%s_train_labels_%s.npy" % (dataset, noise_ratio)
        #################################################################
        if os.path.isfile(data_file):
            y_train = np.load(data_file)
            if dataset == 'cifar-100' and asym:
                P = np.load(P_file)
        else:
            if asym:
                if dataset == 'mnist':
                    # 1 < - 7, 2 -> 7, 3 -> 8, 5 <-> 6
                    source_class = [7, 2, 3, 5, 6]
                    target_class = [1, 7, 8, 6, 5]
                else:
                    print('Asymmetric noise is not supported now for dataset: %s' % dataset)
                    return

                if dataset == 'mnist':
                    for s, t in zip(source_class, target_class):
                        cls_idx = np.where(y_train_clean == s)[0]
                        # print('cls_idx',cls_idx)
                        n_noisy = int(noise_ratio * cls_idx.shape[0] / 100)
                        # print('n_noisy',n_noisy)
                        noisy_sample_index = np.random.choice(cls_idx, n_noisy, replace=False)
                        y_train[noisy_sample_index] = t
                        # print(y_train[noisy_sample_index])
                        # print(y_train_clean[noisy_sample_index])
            else:
                n_samples = y_train.shape[0]
                n_noisy = int(noise_ratio * n_samples / 100)
                class_index = [np.where(y_train_clean == i)[0] for i in range(NUM_CLASSES[dataset])]
                class_noisy = int(n_noisy / NUM_CLASSES[dataset])

                noisy_idx = []
                for d in range(NUM_CLASSES[dataset]):
                    noisy_class_index = np.random.choice(class_index[d], class_noisy, replace=False)
                    noisy_idx.extend(noisy_class_index)
                    

                for i in noisy_idx:
                    y_train[i] = other_class(n_classes=NUM_CLASSES[dataset], current_class=y_train[i])
                    # print(y_train[noisy_class_index])
                    # print(y_train_clean[noisy_class_index])
                
            PATH = os.path.join(ROOT_PATH, data_file)
            np.save(PATH, y_train)


        # print statistics
        print("Print noisy label generation statistics:")
        for i in range(NUM_CLASSES[dataset]):
            n_noisy = np.sum(y_train == i)
            print("Noisy class %s, has %s samples." % (i, n_noisy))

    if random_shuffle:
        # random shuffle
        idx_perm = np.random.permutation(X_train.shape[0])
        X_train, y_train, y_train_clean = X_train[idx_perm], y_train[idx_perm], y_train_clean[idx_perm]

    # one-hot-encode the labels
    y_train_clean = np_utils.to_categorical(y_train_clean, NUM_CLASSES[dataset])
    y_train = np_utils.to_categorical(y_train, NUM_CLASSES[dataset])
    y_test = np_utils.to_categorical(y_test, NUM_CLASSES[dataset])

    print("X_train:", X_train.shape)
    print("y_train:", y_train.shape)
    print("X_test:", X_test.shape)
    print("y_test", y_test.shape)
    return X_train, y_train, X_test, y_test


if __name__ == "__main__":
    X_train, Y_train, X_test, Y_test = get_data(dataset='mnist', noise_ratio=40, asym=True)
    Y_train = np.argmax(Y_train, axis=1)
    (_, Y_clean_train), (_, Y_clean_test) = mnist.load_data()
    clean_selected = np.argwhere(Y_train == Y_clean_train).reshape((-1,))
    noisy_selected = np.argwhere(Y_train != Y_clean_train).reshape((-1,))
    print("#correct labels: %s, #incorrect labels: %s" % (len(clean_selected), len(noisy_selected)))