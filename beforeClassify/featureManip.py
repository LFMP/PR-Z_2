class FeatureManip:

    '''
    labelPosition = "last" or "first"
    '''
    def readFeatures(self, featureFile, labelPosition):
        features = {}

        print("Reading Features...")
        for line in open(featureFile, 'r'):
            info = [i for i in line.split(' ') if (i != "" and i != " " and i != "\n")]
            leng = len(info)

            if labelPosition == "last":
                tag = info[leng-1]
            else:
                tag = info[0]

            features[tag] = ""

            count = 1
            for feature in info[:-1]:
                features[tag] += " {0}:{1}".format(count, feature)
                count += 1
            
        print("Done! Number of readen files: ", len(features))

        return features
