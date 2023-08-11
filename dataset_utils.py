import os
import img_tools
import csv
import numpy as np
import static_values as sv
import pandas as pd
import shutil
import matplotlib.pyplot as plt
import itertools
from tqdm import tqdm
from random import shuffle
import operator
from multiprocessing.dummy import Pool as ThreadPool


class DatasetOrganizer:
    def __init__(self, train_df, test_df, images_per_file=sv.STATIC_VALUES.images_per_file, use_tif=False):
        self.use_tif = use_tif

        self.train_df = train_df
        self.test_df = test_df

        self.images_per_file = images_per_file

        self.current_train_npy = None
        self.current_train_npy_idx = 0

        self.current_test_npy = None
        self.current_test_npy_idx = 0

    def get_train_image(self, index):
        if (self.current_train_npy is None) | (index // self.images_per_file != self.current_train_npy_idx):
            self.current_train_npy = np.load(
                sv.STATIC_VALUES.local_dir + 'images-npy/train/data-{}.npy'.format(index // self.images_per_file))

            self.current_train_npy_idx = index // self.images_per_file
        return self.current_train_npy[index % self.images_per_file]

    def get_test_image(self, index):
        if (self.current_test_npy is None) | (index // self.images_per_file != self.current_test_npy_idx):
            self.current_test_npy = np.load(
                sv.STATIC_VALUES.local_dir + 'images-npy/test/data-{}.npy'.format(index // self.images_per_file))
            self.current_test_npy_idx = index // self.images_per_file
        return self.current_test_npy[index % self.images_per_file]

    def create_npy_dataset(self):
        # create npy files according to images_per_file
        print('Create images npy files...')
        """"""
        print('Train...')
        file_iter = 0
        for start in range(0, len(self.train_df), self.images_per_file):
            end = min(start + self.images_per_file, len(self.train_df))
            files_df = self.train_df[start:end]
            print("'{}' - '{}'".format(self.train_df.values[start, 0], self.train_df.values[end - 1, 0]))
            files = files_df.values[:, 0]
            pool = ThreadPool()
            images = pool.map(self.read_train_images, files)
            pool.close()
            pool.join()
            # save to file
            print('Saving data-{}.npy file...'.format(file_iter))
            np.save(sv.STATIC_VALUES.local_dir + 'images-npy/train/data-{}.npy'.format(file_iter), images)
            file_iter += 1

        print('Test...')
        file_iter = 0
        for start in range(0, len(self.test_df), self.images_per_file):
            end = min(start + self.images_per_file, len(self.test_df))
            files_df = self.test_df[start:end]
            print("'{}' - '{}'".format(self.test_df.values[start, 0], self.test_df.values[end - 1, 0]))
            files = files_df.values[:, 0]
            pool2 = ThreadPool()
            images = pool2.map(self.read_test_images, files)
            pool2.close()
            pool2.join()
            # save to file
            print('Saving data-{}.npy file...'.format(file_iter))
            np.save(sv.STATIC_VALUES.local_dir + 'images-npy/test/data-{}.npy'.format(file_iter), images)
            file_iter += 1

    # parallelism
    def read_train_images(self, img_name):
        if self.use_tif:
            img = img_tools.read_tif(sv.STATIC_VALUES.base_dir + 'train-tif-v2/{}.tif'.format(img_name))
        else:
            img = img_tools.read_jpg(sv.STATIC_VALUES.base_dir + 'train-jpg/{}.jpg'.format(img_name))
        return img

    def read_test_images(self, img_name):
        if self.use_tif:
            img = img_tools.read_tif(sv.STATIC_VALUES.base_dir + 'test-tif-v2/{}.tif'.format(img_name))
        else:
            img = img_tools.read_jpg(sv.STATIC_VALUES.base_dir + 'test-jpg/{}.jpg'.format(img_name))
        return img


if __name__ == '__main__':
    df_train_data = pd.read_csv(sv.STATIC_VALUES.base_dir + 'labels/train_v2.csv')
    df_test_data = pd.read_csv(sv.STATIC_VALUES.base_dir + 'labels/sample_submission_v2.csv')
    dt = DatasetOrganizer(df_train_data, df_test_data)
    dt.create_npy_dataset()
