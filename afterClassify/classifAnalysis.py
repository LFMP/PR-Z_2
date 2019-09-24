from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
import numpy as np


def accFromConfusionMatrix(matrixConf):
    length = range(len(matrixConf))
    return float(sum([matrixConf[i][i] for i in length])) / float(sum([matrixConf[i][j] for i in length for j in length]))

def accFromLabelsAndPreds(y_label, y_pred):
    return np.mean(np.equal(y_label, y_pred, dtype=np.int32))

def labelsAndConfMatrixCalc(matrixConf):
    length = range(len(matrixConf))
    labels = []
    predicts = []

    for clas in length:
        for pred in length:
            count = matrixConf[clas][pred]
            labels.extend([clas for i in range(count)])
            predicts.extend([pred for i in range(count)])

    return labels, predicts

def getClasificationReport(matrixConf):
    labels, predicts = labelsAndConfMatrixCalc(matrixConf)

    print(classification_report(labels, predicts, digits=4))
    print("Accuracy: " + str(accFromConfusionMatrix(matrixConf)))
    print(confusion_matrix(labels, predicts))

def getClasificationReportFromCsv(csv_file):
    data = pd.read_csv(csv_file, sep=" ")
    y_pred = data['Pred']
    y_label = data['Label']

    print(classification_report(y_label, y_pred, digits=4))
    print("Accuracy: " + str(accFromLabelsAndPreds(y_label, y_pred)))
    print(confusion_matrix(y_label, y_pred))


def getClassificationReportFromSvmFiles(labelsFiles, predictsFiles, index, separator):
    labels = []
    predicts = []

    for labelsFile in labelsFiles:
        for line in open(labelsFile, 'r'):
            labels.append(line.split(separator)[index])

    for predictsFile in predictsFiles:
        for line in open(predictsFile, 'r'):
            if not line.startswith('labels'):
                predicts.append(line.split(separator)[index])
        
    matrixConf = confusion_matrix(labels, predicts)

    print(classification_report(labels, predicts, digits=4))
    print("Accuracy: " + str(accFromConfusionMatrix(matrixConf)))
    print(matrixConf)

def getPredsFromSvmFiles(predictFile):
    y_pred = []
    predictions = []
    separator = ' '

    for line in open(predictFile, 'r'):
        # se for a primeira linha, a pula
        if line.startswith("labels"):
            continue

        info = line.split(separator)
        y_pred.append(int(info[0]))
        predictions.append([float(i) for i in info[1:]])

    return predictions, y_pred
        

def getDataFromSvmFiles(featuresFiles):
    y_labels = []
    X_data = []
    separator = ' '

    for line in open(featuresFiles, 'r'):
        info = line.split(separator)
        y_labels.append(int(info[0]))
        X_data.append([float(i.split(':')[1]) for i in info[1:]])

    return X_data, y_labels
        
