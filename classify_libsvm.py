from beforeClassify.foldFileCreation import readFolds, formatFoldsFeatures
from svm.svm import svmClassification
from beforeClassify.featureManip import FeatureManip
from afterClassify.classifAnalysis import getClassificationReportFromSvmFiles

# PARAMETERS
NUM_FOLDS = 10
TAG = "glia_cells"
TEXTURE_DESCRIPTOR = 'VGG16_4096'#"lpq_11"
CLASSIFIER = "svm"
COLOR_VARIATION = "_FL" # "_BW" = grayscale; "" = none

# FOLDERS
FEATURES_FILE = "../Features_Predicoes/features/{0}{1}/{0}_hsv.{2}.features".format(TAG, COLOR_VARIATION, TEXTURE_DESCRIPTOR)
FOLDS_FILE = "./gcz_dataset/kFoldsDeclar/{0}/{0}.{1}.folds.txt".format(TAG, NUM_FOLDS)
SVM_FOLDER = '../Features_Predicoes/features/{0}{1}/{2}/'.format(TAG, COLOR_VARIATION, CLASSIFIER)
PREDICS_FOLDER = "../Features_Predicoes/predictions/{0}{1}/{2}/".format(TAG, COLOR_VARIATION, CLASSIFIER)
FEATURES_FOLDER =  SVM_FOLDER + "{0}_folds/{1}/".format(NUM_FOLDS, TEXTURE_DESCRIPTOR)

# create files of features based on a previously created 'folds file'
y = FeatureManip().readFeatures(FEATURES_FILE, "last")
formatFoldsFeatures(FOLDS_FILE, " ", y, SVM_FOLDER, TEXTURE_DESCRIPTOR)

# SVM Classification
svmClassification(NUM_FOLDS, FEATURES_FOLDER, PREDICS_FOLDER, TAG, TEXTURE_DESCRIPTOR)


# Analisys
for disease in ['01_Diabetes', '02_Tumor', '03_Artrite']:
    print("\nClassification Analysis for: ", disease)
    label_files = [FEATURES_FOLDER + '{2}/{0}.{2}.{3}.{1}.test.svm'.format(TAG, TEXTURE_DESCRIPTOR, disease, i) for i in range (NUM_FOLDS)]
    prediction_files = [PREDICS_FOLDER + '{4}_folds/{1}/{2}/{0}.{2}.{3}.{1}.test.svm.predict'.format(TAG, TEXTURE_DESCRIPTOR, disease, i, NUM_FOLDS) for i in range (NUM_FOLDS)]
    getClassificationReportFromSvmFiles(label_files, prediction_files, 0, " ")


