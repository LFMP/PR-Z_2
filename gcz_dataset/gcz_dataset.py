import numpy as np
import cv2, os
from tensorflow.keras.utils import to_categorical
'''
    Import the Glial Cells Dataset as NumPy arrays

    @Date: 03/17/2019
    @Author: Gustavo Zanoni Felipe
'''

class GlialCellsDataset():

    def __init__(self, 
            folds_file="./gcz_dataset/kFoldsDeclar/glia_cells/glia_cells.10.folds.txt", 
            images_path='../Dataset_Variations/resized_04/',
            read_mode=-1
        ):
        self.diseases = ['01_Diabetes', '02_Tumor', '03_Artrite']
        self.folds_files =  self.get_folds(folds_file)
        self.dataset = self.read_images(images_path, read_mode)
        

        self.batch_holder={'train_imgs':[], 'train_labels':[], 'i':0}
    '''
    read_images: read the images present in the dataset

    Parameters:
        dir_path = directory path containing the dataset's images
        read_mode = load the image in:
            * color mode (1) 
            * grayscale mode (0)
            * unchanged (-1) [DEFAULT]

    Return:
        a dictionary cointaining the 3 diseases as keys and a list of numpy arrays
        (where each array represents an image sample) as value
    '''
    def read_images(self, dir_path, read_mode):
        print("Reading images from the dataset...")

        dataset = {i : {j:[] for j in self.diseases} for i in range(10)}
        count = 0
        for img_sample in os.listdir(dir_path):
            if '$' in img_sample:
                img_name = img_sample.split('$')[0]
            else:
                img_name = img_sample.split('.')[0]

            fold_num = self.folds_files[img_name]
            img_disease, img_class = img_name.split('&')[:2]

            img_array = cv2.imread(os.path.join(dir_path, img_sample), read_mode)
            
            if read_mode == 0:
                img_array = np.reshape(img_array, (img_array.shape[0], img_array.shape[1], 1))

            #if resize_to:
            #    img_array = cv2.resize(img_array, resize_to)

            # int(img_class == "Unhealthy") --> 0: Healthy / 1: Unhealthy
            dataset[fold_num][img_disease].append([img_array,  int(img_class == "Unhealthy")])
            count += 1

        print("Dataset suceffuly read! # of Samples read: " + str(count))

        return dataset

    def get_train_test(self, disease, fold):
        X_train = [] # training data
        y_train = [] # training labels
        X_test = [] # test data
        y_test = [] # test labels

        folds = list(range(10))
        folds.remove(fold)

        # get all training samples
        for i in folds:
            for x, y in self.dataset[i][disease]:
                X_train.append(x)
                y_train.append(y)

        # get all testing samples
        for x, y in self.dataset[fold][disease]:
            X_test.append(x)
            y_test.append(y)

        return np.array(X_train), np.array(X_test), to_categorical(y_train), to_categorical(y_test)

    def get_folds(self, input_file):
        output = {}
        with open(input_file, 'r') as folds_file:
            for line in folds_file:
                fold, dir_file, file_name = line.split(' ')
                dir_file = dir_file.split('/')[2:]

                output["{}&{}&{}&{}".format(dir_file[0], dir_file[1], dir_file[2], file_name.split('.')[0])] = int(fold)
        
        print("Folds file suceffuly read!")

        return output

    # return the diseases labels
    def get_deseases(self):
        return self.diseases

    # return the shape of an image sample to a certain disease
    def get_shape(self, disease):
        return self.dataset[0][disease][0][0].shape
    