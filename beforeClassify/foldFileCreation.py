from beforeClassify.featureManip import FeatureManip
import os

def readFolds(foldsFile, separator, output):
    folds = {}

    print("Reading folds...")
    for line in open(foldsFile, 'r'):
        fold, direc, fileName = line.split(separator)
        disease, clas, group = direc.split('/')[2:5]

        # Example: 01_Diabetes&Healthy&00&11_c1.jpg
        fileName = "{}&{}&{}&{}".format(disease, clas, group, fileName[:-1])

        if not disease in folds.keys():
            folds[disease] = {
                'Healthy' : {},
                'Unhealthy' : {}
            }

        folds[disease][clas][fileName] = int(fold)
    print("Done!")

    return folds

def formatFoldsFeatures(foldsFile, separator, features, output, descriptor):
    info = foldsFile.split('/')
    tag, numFolds, _, _ = info[len(info)-1].split('.')
    output += numFolds + "_folds/" + descriptor + "/"
    folds = readFolds(foldsFile, separator, output)
    clas = {
        'Healthy' : 0,
        'Unhealthy' : 1
    }

    numFolds = int(numFolds)

    if not os.path.exists(output):
        os.makedirs(output)

    for disease in folds.keys():
        currentOutput = output + disease + '/'

        if not os.path.exists(currentOutput):
            os.mkdir(currentOutput)

        # example: output/glia_cells.0.lbp.svm
        featuresFiles = {
            'test' : [open('{}{}.{}.{}.{}.test.svm'.format(currentOutput, tag, disease,i, descriptor), 'w') for i in range(numFolds)],
            'train' : [open('{}{}.{}.{}.{}.train.svm'.format(currentOutput, tag, disease,i, descriptor), 'w') for i in range(numFolds)],
        }

        print("Disease: ", disease)
        for claz in folds[disease]:
            count = 0
            for files in folds[disease][claz]:
                sampleFold = folds[disease][claz][files]
                classFeature = "{}{}\n".format(clas[claz], features[files])
                featuresFiles['test'][sampleFold].write(classFeature)

                for j in [i for i in range(numFolds) if i != sampleFold]:
                    featuresFiles['train'][j].write(classFeature)

                count += 1

            print("\t Samples from ",claz,': ',count)

        print()

        for i in range(numFolds):
            featuresFiles['test'][i].close()
            featuresFiles['train'][i].close()
    
    print("Done!")
