import os, shutil, random
'''
    Author: Gustavo Zanoni Felipe
    Date: 11/20/2018

    DataManip: manipulate a dataset
        Params:
            * nFlds = number of folds to cross-validation
            * outName = output label used for file names
            * targetExt = extension of the dataset files (.png, .jpg, .mp3 e etc)
            * outDir = output directory, where the new generated files will be saved 
            * dtFile = file with the mapped dataset (relation between directory and filename)
            * labelPosi = position of the datafile which has the classe' tag
            # divSym = separator symbol from the datafile

'''
class DataManip:
    def __init__(self, nFlds, outName, targetExt, 
                outDir="./gcz_dataset/kFoldsDeclar", 
                dtFile="", 
                labelPosi=-1, 
                separator=" "):
        # Number of folds
        self.numFolds = nFlds
        # Output directory
        self.outDir = outDir
        # Output file label
        self.outName = outName
        # Data file directory
        self.dtFile = dtFile
        # Position of the dataFile which has the class' tag
        self.labelPosi = labelPosi
        # Separator symbol from the dataFile
        self.divSym = separator
        # Extention from the target files
        self.targetExt = targetExt.upper()

    # divide the dataset in "k" folds 
    def kFoldsGen(self):
        if self.dtFile == "":
            print("Error! No data file informed. Plese run the 'mapDataset' function and try again!")
        else:
            foldsFileName = "{0}.{1}.folds.txt".format(self.dtFile, self.numFolds)
            foldFile = open(foldsFileName, 'w')
            samplesDict = {}

            # sort the samples by class
            for sample in open(self.dtFile+'.map.txt'):
                # picks up the sample's class
                sampleTag = sample.split(self.divSym)[self.labelPosi]

                if sampleTag.endswith('\n'):
                    # remove the '\n'
                    sampleTag = sampleTag[:-1]
                

                if sampleTag not in samplesDict.keys():
                    samplesDict[sampleTag] = []
                
                # add in the dictionary
                samplesDict[sampleTag].append(sample)

            # folds division
            print("Creating folds")
            for clas in samplesDict.keys():
                i = 0
                for fold in self.__kPartsDiv(samplesDict[clas]):
                    for element in fold:
                        foldFile.write("{0} {1}".format(i, element))
                    i += 1
            foldFile.close()
            print("Done!\n")

            self.__checkSum(foldsFileName)

    # Divide a list in "k" balanced lists    
    def __kPartsDiv(self, vector):
        return random.shuffle([vector[i::self.numFolds] for i in range(self.numFolds)])

    # create test and train files
    def makeTestTrainFiles(self):
        foldsFolder = "{0}/{1}/{2}_folds/".format(self.outDir, self.outName, self.numFolds)
        foldsFileName = "{0}.{1}.folds.txt".format(self.dtFile, self.numFolds)

        if not os.path.exists(foldsFolder):
            os.mkdir(foldsFolder)

        filesTest = [open("{0}/{1}.{2}.test.txt".format(foldsFolder, self.outName, i), 'w') for i in range(self.numFolds)]
        filesTrain = [open("{0}/{1}.{2}.train.txt".format(foldsFolder, self.outName, i), 'w') for i in range(self.numFolds)]



        print("Creating test/train files...")
        for line in open(foldsFileName, 'r'):
            fold, info, sampleName = line.split(" ")
            sample = "{}{}".format(info, sampleName)

            filesTest[int(fold)].write(sample)

            for i in [j for j in range(self.numFolds) if j != int(fold)]:
                filesTrain[i].write(sample)
        
        for i in range(self.numFolds):
            filesTest[i].close()
            filesTrain[i].close()

        print("Done!\n")
    
    # create a dataset map
    def mapDataset(self, dirData):
        dirs = []
        dirCount = {}

        if not dirData.endswith('/'):
            dirData += "/"

        # define class values
        defOutDir  = "{0}/{1}/".format(self.outDir, self.outName)
        self.dtFile = defOutDir +  self.outName
        self.separator = " "
        self.labelPosi = 1

        if not os.path.exists(defOutDir):
            os.mkdir(defOutDir)
        
        outFile = open(self.dtFile+'.map.txt', 'w')
        outCheckSum = open(defOutDir + self.outName +  '.checksum.txt' , 'w')

        dirs.extend([dirData+i+'/' for i in os.listdir(dirData)])

        print("Creating map...")
        while(len(dirs) > 0):
            # picks a new directory
            newDir = dirs.pop()
            # for all his members
            for member in os.listdir(newDir):
                # if the member of the path is another directory
                # this one is added to the 'dirs' list
                if os.path.isdir(newDir + member):
                    dirs.append("{0}{1}/".format(newDir, member))
                # if it's a target file
                elif member.upper().endswith(self.targetExt):
                    outFile.write("{0} {1}\n".format(newDir, member))

                    if newDir not in dirCount.keys():
                        dirCount[newDir] = 0

                    dirCount[newDir] += 1

        print("Creating check sum file...")
        for key in dirCount.keys():
            outCheckSum.write("{0} {1}\n".format(key, dirCount[key]))

        outCheckSum.close()
        outFile.close()

        print("Done!")

        self.__checkSum("{0}/{1}/{1}.map.txt".format(self.outDir, self.outName))

    def __checkSum(self, targetFile):
        fileSum = 0
        checkSum = 0

        print("Checking for errors...")

        for line in open("{0}/{1}/{1}.checksum.txt".format(self.outDir, self.outName)):
            checkSum += int(line.split(self.separator)[1][:-1])

        for line in open(targetFile):
            fileSum += 1

        if fileSum == checkSum:
            print("No errors reported!")
        else:
             print("Error! Check Sum failed.")
