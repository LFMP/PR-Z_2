import json
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import estimator 

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, LeakyReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers.normalization import BatchNormalization


def plot_history(history, dir_name, save_json=True):
    #taken from: https://machinelearningmastery.com/display-deep-learning-model-training-history-in-keras/

    # summarize history for accuracy
    plt.plot(history['acc'])
    plt.plot(history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(dir_name + "_acc.png")
    plt.clf()

    # summarize history for loss
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(dir_name + "_loss.png")
    plt.clf()


    if save_json:
        with open(dir_name+'_hist.json', 'w') as json_file:
            json.dump(history, json_file, indent=2)
            json_file.write("\n")

class GeneralClassifier():
    def __init__(self, num_classes, img_height, img_width, num_color_channels):
    # ------------------------------------- Constants
        self.FEATURES_FOLDER = ""
        self.NUMBER_FEATURES = 0
        self.N_CLASSES = num_classes

        self.LEARNING_RATE = 0.0001
        self.BATCH_SIZE = 32
        self.NUM_EPOCHS = 64
        self.STEPS = 500

        self.IMG_HEIGHT = img_height
        self.IMG_WIDTH = img_width
        self.NUM_COLOR_CHANNELS = num_color_channels
    '''
    # ---------------------------------------- Helper Functions
    # Normalizes a numPy array 
    # (values between 0 and 1)
    def array_normalization(self, np_array):
        normalization = lambda i : (i - np_array.min()) / ( np_array.max() - np_array.min())
        return np.vectorize(normalization)(np_array)

    # Get an array of percentages from the array's values
    # (Sum of the new array is 1)
    def array_percentage(self, np_array):
        percentage = lambda i : i/np.sum(np_array)
        return np.vectorize(percentage)(np_array)
    
    def to_percentage(self, np_matrix):
        for i in range(np_matrix.shape[0]):
            np_matrix[i] = self.array_normalization(np_matrix[i])
            np_matrix[i] = self.array_percentage(np_matrix[i])
        return np_matrix
    '''
    # ------------------------------------- DNN
    def DNN_Classifier(self, X_train, X_test, y_train, y_test):
        scaler = MinMaxScaler()
        scaled_x_train = scaler.fit_transform(X_train)
        scaled_x_test = scaler.transform(X_test)
        feat_cols = [tf.feature_column.numeric_column("x", shape=[self.NUMBER_FEATURES])]

        deep_model = estimator.DNNClassifier(hidden_units=[self.NUMBER_FEATURES, self.NUMBER_FEATURES, self.NUMBER_FEATURES],
                                            feature_columns=feat_cols,
                                            n_classes=self.N_CLASSES,
                                            optimizer=tf.train.GradientDescentOptimizer(learning_rate=self.LEARNING_RATE))

        input_fn = estimator.inputs.numpy_input_fn(x={'x':scaled_x_train},
                                                y=y_train,
                                                shuffle=True,
                                                batch_size=self.BATCH_SIZE,
                                                num_epochs=self.NUM_EPOCHS)
                        
        deep_model.train(input_fn=input_fn,steps=self.STEPS)

        input_fn_eval = estimator.inputs.numpy_input_fn(x={'x':scaled_x_test},shuffle=False)
        
        #return list(deep_model.predict(input_fn=input_fn_eval))
        return deep_model.predict(input_fn=input_fn_eval, predict_keys='probabilities')

    # ------------------------------------- CNN
    def CNN_Classifier(self, X_train, X_test, X_pred, y_train, y_test, model_name='MaxNet'):
        # INPUT_SHAPE = (self.IMG_HEIGHT, self.IMG_WIDTH, self.NUM_COLOR_CHANNELS)
        
        model = self.model_selector(model_name)

        # train
        history = model.fit(X_train, y_train, validation_data=(X_test, y_test), verbose=1, epochs=self.NUM_EPOCHS, batch_size=self.BATCH_SIZE)

        # evaluate and test
        print("Evaluating")
        score = model.evaluate(X_test, y_test, verbose=0)
        print("Testing")
        predictions = model.predict_proba(X_pred)

        del model

        return predictions, score, history

    def model_selector(self, model_name):
        if model_name == 'MaxNet':
            model = self.MaxNet(self.IMG_HEIGHT, self.IMG_WIDTH, self.NUM_COLOR_CHANNELS)
        elif model_name == 'LeNet5':
            model = self.LeNet5(self.IMG_HEIGHT, self.IMG_WIDTH, self.NUM_COLOR_CHANNELS)
        elif model_name == 'AlexNet':
            model = self.AlexNet(self.IMG_HEIGHT, self.IMG_WIDTH, self.NUM_COLOR_CHANNELS)
        return model

    def MaxNet(self, height, width, num_chan):
        # model created according to the 'Roecker et al. (2018)' architecture
        model = Sequential()
        
        model.add(Conv2D(32, kernel_size=(3,3), strides=(1,1), activation=LeakyReLU(alpha=0.01), input_shape=(height, width, num_chan)))
        model.add(Conv2D(32, kernel_size=(3,3), strides=(1,1), activation=LeakyReLU(alpha=0.01)))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
        
        model.add(Conv2D(32, kernel_size=(3,3), strides=(1,1), activation=LeakyReLU(alpha=0.01)))
        model.add(Conv2D(32, kernel_size=(3,3), strides=(1,1), activation=LeakyReLU(alpha=0.01)))
        model.add((MaxPooling2D(pool_size=(2, 2), strides=(2,2))))

        model.add(Flatten())

        model.add(Dense(4096, activation=LeakyReLU(alpha=0.01)))
        model.add(Dropout(0.5))
        model.add(Dense(512, activation=LeakyReLU(alpha=0.01)))
        model.add(Dropout(0.5))
        model.add(Dense(512, activation=LeakyReLU(alpha=0.01)))
        model.add(Dense(self.N_CLASSES, activation='softmax'))

        model.compile(optimizer=Adam(lr=self.LEARNING_RATE, beta_1=0.9, beta_2=0.999, decay=0.0005), loss='categorical_crossentropy', metrics=['accuracy'])

        return model
    
    def LeNet5(self, height, width, num_chan):
        # model created according to the 'LeCun et al., (1998)' architecture
        model = Sequential()
        
        model.add(Conv2D(filters = 6, kernel_size = 5, strides = 1, activation = 'relu', input_shape = (height, width, num_chan))) 
        model.add(MaxPooling2D(pool_size= 2, strides = 2))

        model.add(Conv2D(filters = 16, kernel_size = 5, strides = 1, activation = 'relu'))
        model.add(MaxPooling2D(pool_size = 2, strides = 2))
        model.add(Flatten())

        model.add(Dense(units = 120, activation = 'relu'))
        model.add(Dense(units = 84, activation = 'relu'))
        model.add(Dense(units = self.N_CLASSES, activation = 'softmax'))

        model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

        return model

    def AlexNet(self, height, width, num_chan):
        # model created according to the 'Krizhevsky et al. (2012)' architecture
        # keras version originally from https://www.mydatahack.com/building-alexnet-with-keras/
        model = Sequential()

        # 1st Convolutional Layer
        model.add(Conv2D(filters=96, input_shape=(height, width, num_chan), activation = 'relu', kernel_size=(11,11), strides=(4,4), padding='valid'))
        model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
        model.add(BatchNormalization())

        # 2nd Convolutional Layer
        model.add(Conv2D(filters=256, kernel_size=(11,11), strides=(1,1), activation = 'relu', padding='valid'))
        model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
        model.add(BatchNormalization())

        # 3rd Convolutional Layer
        model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation = 'relu', padding='valid'))
        model.add(BatchNormalization())

        # 4th Convolutional Layer
        model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation = 'relu', padding='valid'))
        model.add(BatchNormalization())

        # 5th Convolutional Layer
        model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), activation = 'relu', padding='valid'))
        model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
        model.add(BatchNormalization())

        # Passing it to a dense layer
        model.add(Flatten())
        # 1st Dense Layer
        model.add(Dense(4096, activation = 'relu', input_shape=(height, width, num_chan,)))
        model.add(Dropout(0.4))
        model.add(BatchNormalization())

        # 2nd Dense Layer
        model.add(Dense(4096, activation = 'relu'))
        model.add(Dropout(0.4))
        model.add(BatchNormalization())

        # 3rd Dense Layer
        model.add(Dense(1000, activation = 'relu'))
        model.add(Dropout(0.4))
        model.add(BatchNormalization())

        # Output Layer
        model.add(Dense(self.N_CLASSES, activation = 'softmax'))

        model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

        return model

    def save_model_to_file(self, file_path, model_name="MaxNet"):
        model = self.model_selector(model_name)

        with open(file_path + model_name+'_model.json', 'w') as json_file:
            json.dump(model.to_json(), json_file, indent=2)
            json_file.write("\n")
    
