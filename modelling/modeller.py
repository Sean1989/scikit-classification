
import sys
import numpy as np
from sklearn.ensemble import RandomForestClassifier

class Modeller:
    
    def __init__ (self, trainingInput, trainingLabels, testInput):
        self.loadData(trainingInput, trainingLabels, testInput)
        
    def loadData(self, trainInput, trainLabels, testInput):
        self.xTrain = np.loadtxt(open(trainInput, 'rb'), delimiter = ',')
        self.xTest = np.loadtxt(open(testInput, 'rb'), delimiter = ',')
        self.yTrain = np.loadtxt(open(trainLabels, 'rb'))
        
    def runModeller(self):
        numberOfRows, numberOfCols = self.xTrain.shape
        
        trainRows = int(0.8 * numberOfRows)
        valRows = trainRows+1
        
        subXTrain = self.xTrain[0:trainRows,:]
        subYTrain = self.yTrain[0:trainRows]
        
        subXVal = self.xTrain[valRows::, :]
        subYVal = self.yTrain[valRows::]
        
        rfClassifier = RandomForestClassifier(n_estimators = 1000, max_depth = 2)
        
        rfClassifier.fit(subXTrain, subYTrain)
        yValPred = rfClassifier.predict(subXVal)
        
        print self.find_accuracy(yValPred, subYVal)
        
        yTestPred = rfClassifier.predict(self.xTest)
        
        self.write_output("../output/testLabels.csv", yTestPred)
        
    def write_output(self, outputFile, yPred):
        outputWriter = open(outputFile, 'w+')
        for i in range(0, len(yPred)):
            if i == len(yPred) -1 :
                outputWriter.write(str(int(yPred[i])))
            else:
                outputWriter.write(str(int(yPred[i]))+'\n')
        outputWriter.close()
        
    
    def find_accuracy(self, yPred, yTrue):
        return np.mean(yPred - yTrue)
        
        
        
if __name__ == "__main__":
    if len(sys.argv) < 4:
        print "Please give the train.csv, test.csv and trainLabels.csv"
        print "python modeller.py <train.csv> <trainLabels.csv> <test.csv>"
        sys.exit(1)
    trainInput = sys.argv[1]
    trainLabels = sys.argv[2]
    testInput = sys.argv[3]
    model = Modeller(trainInput, trainLabels, testInput)
    model.runModeller()