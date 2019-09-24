from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, MaxPooling2D, LeakyReLU, Dropout
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input
from sklearn.feature_selection import SelectKBest, chi2

import numpy as np
import cv2, os

def select_model(model_name='VGG16'):
    if model_name == 'VGG16':
        input_format = (224, 224)
        from tensorflow.keras.applications.vgg16 import VGG16
        base_mod = VGG16
    elif model_name == 'InceptionV3':
        input_format = (299, 299)
        from tensorflow.keras.applications.inception_v3 import InceptionV3
        base_mod = InceptionV3
    elif model_name == 'InceptionResNetV2':
        input_format = (299, 299)
        from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
        base_mod = InceptionResNetV2

    base_model = base_mod(weights='imagenet', include_top=False)

    return base_model, input_format

def extract_features(img, base_model):
    features = base_model.predict(img)
    return np.reshape(features, (np.prod(np.shape(features))))

def feature_learning_to_file(input_dir, output_dir, tag, n_features=512, model='InceptionV3'):
    base_model, inp_format = select_model(model_name=model)
    out_file = open("{}/{}.{}_{}.features".format(output_dir, tag, model, n_features), 'w')
    
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    dataset = {
        '01_Diabetes' : {'img_name':[], 'X_data':[], 'y_labels':[]}, 
        '02_Tumor'    : {'img_name':[], 'X_data':[], 'y_labels':[]}, 
        '03_Artrite'  : {'img_name':[], 'X_data':[], 'y_labels':[]}
    }

    for sample in os.listdir(input_dir):
        img = cv2.imread(input_dir + sample, 1)
        img = np.array([cv2.resize(img, inp_format)])

        img_disease, img_class = sample.split('&')[:2]

        dataset[img_disease]['img_name'].append(sample) 
        dataset[img_disease]['X_data'].append(extract_features(img, base_model))
        dataset[img_disease]['y_labels'].append(int(img_class == "Unhealthy"))

    selector = SelectKBest(chi2, k=n_features)

    count = 0
    for disease in dataset.keys():
        print(disease)
        dataset[disease]['X_data'] = selector.fit_transform(dataset[disease]['X_data'], dataset[disease]['y_labels'])

        for i in range(len(dataset[disease]['X_data'])):
            count += 1
            
            for feature in dataset[disease]['X_data'][i]:
                out_file.write(str(feature) + " ")
            print(dataset[disease]['img_name'][i] + " #" + str(count))
            out_file.write(dataset[disease]['img_name'][i] + " \n")


    del dataset, selector
    out_file.close()

#INPUT = "../../GC_Images/"
INPUT = "../../Dataset_Variations/02_Glia_Images_BW/"
OUTPUT = "../../Features&Predicoes/features/glia_cells_FL/"

for i in [4096]:
	feature_learning_to_file(INPUT, OUTPUT, "glia_cells", n_features=i, model="VGG16")
    print(i)
    feature_learning_to_file(INPUT, OUTPUT, 'glia_cells_gs', n_features=i, model='InceptionV3')
    #feature_learning_to_file(INPUT, OUTPUT, 'glia_cells', n_features=i, model='VGG16')
    #feature_learning_to_file(INPUT, OUTPUT, 'glia_cells', n_features=i, model='InceptionResNetV2')
