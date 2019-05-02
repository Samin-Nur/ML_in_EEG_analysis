
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio


from sklearn import metrics

from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestRegressor
from util import plot_confusion_matrix
from sklearn.decomposition import PCA



def dataLoad(data_url,num_class):
    trainX = np.load(data_url + '/trainX_new.npy')
    trainY = np.load(data_url + '/trainY_new.npy')
    testX = np.load(data_url + '/testX_new.npy')
    testY = np.load(data_url + '/testY_new.npy')


    trainX = np.array(trainX, dtype='float64')
    testX = np.array(testX, dtype='float64')
    


    return trainX,trainY,testX,testY

# Evaluation
def PRAF(test, pred, num_class, algo_name, reorder=True):
    if reorder == True:
        for index in range(0,len(pred)):
            if pred[index] < 1:
                pred[index] = 1
            if pred[index] > 5:
                pred[index] = 5
            pred[index] = (int)(round(pred[index]))

    pred = map(int,pred)
    pred = list(pred)
    precision = metrics.precision_score(y_true=test, y_pred=pred,average='macro')
    recall = metrics.recall_score(y_true=test, y_pred=pred,average='macro')
    f1 = metrics.f1_score(y_true=test, y_pred=pred,average='macro')
    accuracy = metrics.accuracy_score(y_true=test, y_pred=pred)

    r2score=metrics.r2_score(y_true=test, y_pred=pred)
    mse=metrics.mean_squared_error(y_true=test, y_pred=pred)
    x = np.linspace(1,300,300)
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
    plt.subplot(2,1,1)
    plt.plot(x,test)
    plt.xlabel("events")
    plt.ylabel("actual class")
    plt.yticks([1,2,3,4,5])


     #As above but select 2nd plot
    plt.subplot(2,1,2)
    plt.plot(x,pred)
    plt.xlabel("events")
    plt.ylabel("predicted class")
    plt.yticks([1, 2, 3, 4, 5])
    #show only once for all plots
    plt.show()

    
    sio.savemat('pred.mat', mdict={'pred':pred})
    sio.savemat('test.mat', mdict={'test':test})




   
# Algorithms
def Linear_Regression(trainX,trainY,testX,testY,num_class):
    print(" ")
    print("----------------------------------------------")
    algo_name = 'Linear Regression'
    lm = LinearRegression()
    model = lm.fit(trainX,trainY)
    predictions = lm.predict(testX)
    PRAF(testY, predictions,num_class,algo_name)

def Decision_Tree_Classifier(X_train,y_train,X_test,y_test,num_class):
    print(" ")
    print("----------------------------------------------")
    algo_name = 'Decision Tree Classifier'
    clf = DecisionTreeClassifier()
    clf = clf.fit(X_train,y_train)
    y_pred_dtc=clf.predict(X_test)
#    kappascore=metrics.cohen_kappa_score(y_test, y_pred_dtc) #NOT PRINTED
    PRAF(y_test, y_pred_dtc,num_class,algo_name, reorder=False)

def SGD_Classification(X_train,y_train,X_test,y_test,num_class):
    print(" ")
    print("----------------------------------------------")
    algo_name = "SGD_Classification"
    sgd = SGDClassifier()
    sgd.fit(X_train, y_train)
    pred = sgd.predict(X_test)
    PRAF(y_test, pred,num_class,algo_name, reorder=False)

def Support_Vector_Machine(X_train,y_train,X_test,y_test,num_class):
    print(" ")
    print("----------------------------------------------")
    algo_name = 'Support Vector Machine'
    svm_model_linear = SVC(kernel='linear',C=1).fit(X_train,y_train)
    svm_predictions = svm_model_linear.predict(X_test)
    PRAF(y_test, svm_predictions,num_class,algo_name)

def Random_Forest_Regression(X_train,y_train,X_test,y_test,num_class):
    algo_name = 'Random Forest Regression'
    rf_model = RandomForestRegressor(max_depth=4,n_estimators=100,max_features='sqrt',verbose=1,random_state=1)
    rf_model.fit(X_train,y_train)
    y_pred_rf = rf_model.predict(X_test)
    PRAF(y_test, y_pred_rf,num_class,algo_name)


def main():
    data_url = "/home/samin/Downloads/ML6"
    num_class = 5
    #trainX,trainY,testX,testY = DataPrepare(data_url);
    trainX,trainY,testX,testY = dataLoad(data_url,num_class);
    print("Data Loaded..")
    print("Data Information: ")
    print("Train X: " + str(trainX.shape))
    print("Train Y: " + str(trainY.shape))
    print("Test X: " + str(testX.shape))
    print("Test Y: " + str(testY.shape))

    # Algorithm Calls
    Linear_Regression(trainX,trainY,testX,testY,num_class)
    Decision_Tree_Classifier(trainX,trainY,testX,testY,num_class)
    SGD_Classification(trainX,trainY,testX,testY,num_class)
    Support_Vector_Machine(trainX,trainY,testX,testY,num_class)
    Random_Forest_Regression(trainX,trainY,testX,testY,num_class)

    print()
    print("Done...")

    check = 0

if __name__ == '__main__':
  main()
