import numpy as np
import pandas as pd
import os, time

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
#from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import GridSearchCV
from afterClassify.classifAnalysis import getDataFromSvmFiles, getClasificationReportFromCsv

# ---------------------------------------------------------------------------------------- Parametros
tag = 'glia_cells_BW'
text_desc = 'lpq_11'
N_FOLDS = 10

diseases = ['03_Artrite']#['01_Diabetes', '02_Tumor', '03_Artrite']
OUTPUT_DIR = './gcz_dataset/predictions/glia_cells/nb/'
file_path = './gcz_dataset/features/{}/svm/10_folds/{}/'.format(tag, text_desc)
# ----------------------------------------------------------------------------------------- classifiers

def get_knn_clf():
    parameters = {'n_neighbors':[3, 5, 7, 11, 13, 15], 'weights':('uniform', 'distance')}
    return GridSearchCV(KNeighborsClassifier(), parameters, scoring='accuracy', cv=4, n_jobs=-1), parameters.keys()

def get_rf_clf():
    parameters = {'n_estimators':(50, 75, 100, 125, 150)}
    return GridSearchCV(RandomForestClassifier(), parameters, scoring='accuracy', cv=4, n_jobs=-1), parameters.keys()

def get_lr_clf():
    parameters = {'C':(0.01, 0.1, 1.0, 10.0, 100.0)}
    return GridSearchCV(LogisticRegression(solver='liblinear'), parameters, scoring='accuracy', cv=4, n_jobs=-1), parameters.keys()

# ------------------------------------------------------------------------------------------

if not os.path.isdir(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)

OUTPUT_DIR += text_desc + '_gs/'

if not os.path.isdir(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)

beg_time = time.time()

for disease in diseases:
    full_path = file_path + disease + '/'
    #log = open(OUTPUT_DIR + 'log_'+disease+'.txt', 'w')
    output = []
    for i in range(N_FOLDS):
        classifier, params = GaussianNB(), []

        print(disease, i)
        file_name = 'glia_cells.{}.{}.{}'.format(disease, i, text_desc)

        X_train, y_train = getDataFromSvmFiles(full_path + file_name + '.train.svm')
        X_test , y_test  = getDataFromSvmFiles(full_path + file_name + '.test.svm')
        
        print("Training...")
        classifier.fit(X_train, y_train)
        print("Testing...")
        preds = classifier.predict_proba(X_test)
        
        #log.write("Fold: {}\n".format(i))
        #for param in params:
        #    log.write("\t{}: {}\n".format(param, classifier.get_params()['estimator__' + param]))
        #log.write("\n")

        print()

        for j in range(len(preds)):
            pred = preds[j]
            output.append([np.argmax(pred), y_test[j], pred[0], pred[1]])

        del classifier, preds, file_name, X_train, y_train, X_test , y_test

    pd.DataFrame(output, columns=['Pred', 'Label', 'Healthy', 'Unhealthy']).to_csv(OUTPUT_DIR + disease + '.csv', sep=" ")
    #log.close()

    del output, full_path#, log

for disease in diseases: 
    print("\nGetting classification report: " + disease)
    getClasificationReportFromCsv(OUTPUT_DIR + disease + ".csv")

print("Execution time (seconds): ", time.time() - beg_time)

