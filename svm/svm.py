import os, shutil

def svmClassification(numFolds, inputDir, outputDir, tag, descriptor):
    libsvmDir =  "./svm/libsvm-3.21/"
    libsvm = libsvmDir + 'tools/easyZ.py'
    outputDir += "{}_folds/{}/".format(numFolds, descriptor)

    if not os.path.exists(outputDir):
        os.makedirs(outputDir)

    for disease in os.listdir(inputDir):
        currentDir = inputDir + disease + '/'

        if not os.path.exists(currentDir):
            os.makedirs(currentDir)

        for i in range(numFolds):
            fileName = "{}.{}.{}.{}".format(tag, disease, i, descriptor)
            test = "{}/{}/{}.test.svm".format(inputDir, disease, fileName)
            train = "{}/{}/{}.train.svm".format(inputDir, disease, fileName )
            
            cmd = "python {} {} {} {}".format(libsvm, train, test, libsvmDir)
            print("Disease: {} / Fold: {}\n".format(disease, i))
            os.system(cmd)

            for file in os.listdir("./"):
                if file.startswith(fileName):
                    currentOutDir = outputDir + disease + '/' 
                    if not os.path.exists(currentOutDir):
                        os.makedirs(currentOutDir)
                    shutil.move("./" + file, currentOutDir + file)
