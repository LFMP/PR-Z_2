from afterClassify.classifCombin import max_rule, sum_rule, product_rule
from afterClassify.classifAnalysis import getPredsFromSvmFiles, getDataFromSvmFiles, accFromLabelsAndPreds


import pandas as pd
import numpy as np
import os
from itertools import combinations 

# --------------------------------------------------------------------------------------- GLOBAIS
n_folds = 10
diseases = ['01_Diabetes', '02_Tumor', '03_Artrite']
# --------------------------------------------------------------------------------------- FUNCOES
def get_subsets(set_keys):
    subset = []
    for i in range(2, len(set_keys)+1):
        subset.extend(combinations(set_keys, i))

    return subset
# ---------------------------------------------------------------------------------------- Convert .svm to .csv
'''
DESC = 'InceptionV3_4096'

preds_path = "../Features_Predicoes/predictions/glia_cells_FL/svm/10_folds/" + DESC + '/'
labels_path = "../Features_Predicoes/features/glia_cells_FL/svm/10_folds/" + DESC + '/'

for disease in diseases:
    print(disease)
    
    cur_preds_path = preds_path + disease + '/'
    cur_labels_path = labels_path + disease + '/'
    cur_output = preds_path + disease + '.csv'

    output = []

    for i in range(n_folds):
        labels_file = cur_labels_path + 'glia_cells.{0}.{1}.{2}.test.svm'.format(disease, i, DESC)
        preds_file = cur_preds_path + 'glia_cells.{0}.{1}.{2}.test.svm.predict'.format(disease, i, DESC)

        _, y_labels = getDataFromSvmFiles(labels_file)
        predicts, y_pred = getPredsFromSvmFiles(preds_file)

        for j in range(len(y_labels)):
            output.append([y_pred[j], y_labels[j]] + predicts[j])

    print('Writing output...')
    pd.DataFrame(output, columns=['Pred', 'Label', 'Healthy', 'Unhealthy']).to_csv(cur_output, sep=" ")
'''
# -------------------------------------------------------------------------------------- CLASSIFIER COMBINATION
input_path = '../Features_Predicoes/combinations/input/'
output_path = '../Features_Predicoes/combinations/output/'


for disease in diseases:
    cur_input = input_path + disease + '/'
    cur_output = input_path + disease + '/'
    file_name = disease + '.csv'

    data = {}
    for sub_folder in os.listdir(cur_input):
        data[sub_folder] = pd.read_csv(cur_input + sub_folder + '/' + file_name, sep=' ')
        #print(sub_folder)
        #print(data[sub_folder].values[:, 3:])

    y_labels = data[list(data.keys())[0]]['Label']
    subsets = get_subsets(set(data.keys()))
    
    i = 0
    # para cada subconjunto dos possiveis
    output = output_path + disease + '.csv'
    output_list = []
    for subset in subsets:
        names = []
        predicts = []

        # pega todas as predicoes e coloca em uma lista
        for predict in subset:
            names.append(predict)
            predicts.append(data[predict].values[:, 3:])

        # pegar as novas predicoes com os 3 metodos
        max_preds, _ = max_rule(predicts)
        sum_preds, _ = sum_rule(predicts)
        product_preds, _ = product_rule(predicts)

        names += [0 for i in range(len(data.keys()) - len(names))]
        output_list.append(names + [accFromLabelsAndPreds(y_labels, max_preds), 
                                    accFromLabelsAndPreds(y_labels, sum_preds), 
                                    accFromLabelsAndPreds(y_labels, product_preds)]
                            )

        del names, max_preds, product_preds, sum_preds, predicts
        print(disease, i)
        i += 1

    pd.DataFrame(output_list, columns=[str(i) for i in range(len(data.keys()))] + ['Max', 'Sum', 'Prod']).to_csv(output, sep=' ')

    del output_list, output, subsets, y_labels, data, file_name, cur_input, cur_output

            