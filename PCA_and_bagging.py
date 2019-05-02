import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

from sklearn import metrics
from sklearn.svm import SVC

from util import plot_confusion_matrix
from sklearn.ensemble import BaggingClassifier
from sklearn.decomposition import PCA

def dataLoad(data_url, num_class):
    trainX = np.load(data_url + '/trainX_new.npy')
    trainY = np.load(data_url + '/trainY_new.npy')
    testX = np.load(data_url + '/testX_new.npy')
    testY = np.load(data_url + '/testY_new.npy')

    pca=PCA(21)

    pca.fit(trainX)

    trainX=pca.transform(trainX)
    testX=pca.transform(testX)


    trainX = np.array(trainX, dtype='float64')
    testX = np.array(testX, dtype='float64')

    return trainX, trainY, testX, testY


# Evaluation
def PRAF(test, pred, num_class, algo_name, reorder=True):
    if reorder == True:
        for index in range(0, len(pred)):
            if pred[index] < 1:
                pred[index] = 1
            if pred[index] > 5:
                pred[index] = 5
            pred[index] = (int)(round(pred[index]))

    pred = map(int, pred)
    pred = list(pred)
    precision = metrics.precision_score(y_true=test, y_pred=pred, average='macro')
    recall = metrics.recall_score(y_true=test, y_pred=pred, average='macro')
    f1 = metrics.f1_score(y_true=test, y_pred=pred, average='macro')
    accuracy = metrics.accuracy_score(y_true=test, y_pred=pred)

    r2score = metrics.r2_score(y_true=test, y_pred=pred)
    mse = metrics.mean_squared_error(y_true=test, y_pred=pred)
    x = np.linspace(1, 300, 300)
    #
    print()
    print("*" + algo_name + "*")
    print ("R-squared: " + str(r2score))
    print ("MSE: " + str(mse))
    print ("Precision: " + str(precision))
    print ("Recall: " + str(recall))
    print ("accuracy: " + str(accuracy))
    print ("f1: " + str(f1))
    print("----------------------------------------------")

    plot_confusion_matrix(pred, test, algo_name + ' - confusion matrix')

    plt.figure(algo_name)
    plt.subplot(2, 1, 1)
    plt.plot(x, test)
    plt.xlabel("events")
    plt.ylabel("actual class")
    plt.yticks([1, 2, 3, 4, 5])

    # As above but select 2nd plot
    plt.subplot(2, 1, 2)
    plt.plot(x, pred)
    plt.xlabel("events")
    plt.ylabel("predicted class")
    plt.yticks([1, 2, 3, 4, 5])
    # show only once for all plots
    plt.show()

    sio.savemat('pred.mat', mdict={'pred': pred})
    sio.savemat('test.mat', mdict={'test': test})



    return accuracy




# Algorithms
def Bagging_Support_Vector_Machine(X_train, y_train, X_test, y_test, num_class,kernal_info,C_info,clw):
    print(" ")
    print("----------------------------------------------")
    algo_name = 'Support Vector Machine'
    bagging_svm_model_linear = BaggingClassifier(SVC(kernel=kernal_info, C=C_info,class_weight=clw),max_features=21,max_samples=300).fit(X_train, y_train)
    bagging_svm_predictions = bagging_svm_model_linear.predict(X_test)
    accuracy=PRAF(y_test, bagging_svm_predictions, num_class, algo_name)
    return accuracy

def main():
    data_url = "/home/samin/Downloads/ML6"
    num_class = 5

    # trainX,trainY,testX,testY = DataPrepare(data_url);
    trainX, trainY, testX, testY = dataLoad(data_url, num_class);
    print("Data Loaded..")
    print("Data Information: ")
    print("Train X: " + str(trainX.shape))
    print("Train Y: " + str(trainY.shape))
    print("Test X: " + str(testX.shape))
    print("Test Y: " + str(testY.shape))


    accuracy = Bagging_Support_Vector_Machine(trainX, trainY, testX, testY, num_class, 'linear', .9, {3:.5})


    print()
    print("Done...")

    check = 0


if __name__ == '__main__':
    main()
