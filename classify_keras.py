from neuralNets.nns import GeneralClassifier, plot_history
from gcz_dataset.gcz_dataset import GlialCellsDataset
import pandas as pd
import numpy as np
from afterClassify.classifAnalysis import getClasificationReportFromCsv
import os, time

dataset = GlialCellsDataset(images_path='../Dataset_Variations/DA_16_resized_64x48/', read_mode=-1)
original_dataset =  GlialCellsDataset(images_path='../Dataset_Variations/resized_64x48/', read_mode=-1)
OUTPUT_DIR = './gcz_dataset/predictions/glia_cells/cnn/DA_16_resized_64x48/'
diseases = dataset.get_deseases()[1:]
# 0 = HSV, 1 = RGB, 2 = GS

# classifier
shape = dataset.get_shape(diseases[0])
classif = GeneralClassifier(2, shape[0], shape[1], shape[2])
classif.LEARNING_RATE = 0.00001
classif.NUM_EPOCHS = 64
classif.BATCH_SIZE = 128

if not os.path.isdir(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)
    os.mkdir(OUTPUT_DIR + "graphics/")
    os.mkdir(OUTPUT_DIR + "models/")

beg_time = time.time()

for disease in diseases:
    histories = []
    output = []

    for i in range(10):
        print("\nClassifying Disease: {} / Folds: {}\n".format(disease, i))

        # train with DA and test with original
        X_train, _, y_train, _ = dataset.get_train_test(disease, i)
        _, X_test, _, y_test = original_dataset.get_train_test(disease, i)
        X_hold, y_hold = X_test, y_test

        print("\nTest: {} / Train: {}\n".format(len(y_test), len(y_train)))

        preds, score, hist  = classif.CNN_Classifier(X_train, X_test, X_hold, 
                                                    y_train, y_test, 
                                                    disease + "_" + str(i),
                                                    OUTPUT_DIR + "models/", 
                                                    model_name='AlexNet')
        print(score)
        #histories.append(hist)
        plot_history(hist.history,"{}graphics/{}".format(OUTPUT_DIR, disease+"_"+str(i)))

        for j in range(len(preds)):
            pred = preds[j]
            output.append([np.argmax(pred), np.argmax(y_hold[j]), pred[0], pred[1]])

        # clear memory
        del X_train, X_test, X_hold, y_train, y_test, y_hold, preds, score, hist
        K.clear_session()
    
    #hist = {
    #    'acc':      np.mean(np.array([i.history['acc'] for i in histories]), axis=0      ).tolist(),
    #    'val_acc':  np.mean(np.array([i.history['val_acc'] for i in histories]), axis=0  ).tolist(),
    #    'loss':     np.mean(np.array([i.history['loss'] for i in histories]), axis=0     ).tolist(),
    #    'val_loss': np.mean(np.array([i.history['val_loss'] for i in histories]), axis=0 ).tolist()
    #}
    pd.DataFrame(output, columns=['Pred', 'Label', 'Healthy', 'Unhealthy']).to_csv(OUTPUT_DIR + disease + '.csv', sep=" ")
    #plot_history(hist,"{}graphics/{}".format(OUTPUT_DIR, disease))

    # clear memory
    del histories, output#, hist

for disease in diseases: 
    print("\nGetting classification report: " + disease)
    getClasificationReportFromCsv(OUTPUT_DIR + disease + ".csv")

print("Execution time (seconds): ", time.time() - beg_time)
