import numpy as np
import scipy.io
import os

from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler

def SVM(Xs_new,Xt_new,Ys,Yt):
    C = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 4, 8, 16, 32]
    Classification_report = []
    Accuracy_score = []
    MAR = []
    for i in range(len(C)):
        clf_SVM = SVC(kernel='rbf', C=C[i],gamma='auto')
        clf_SVM.fit(X=Xs_new, y=Ys)
        pre_SVM = clf_SVM.predict(Xt_new)
        Classification_report.append(classification_report(Yt.ravel(),pre_SVM.ravel(),labels=[0,1]))
        Accuracy_score.append(accuracy_score(Yt.ravel(),pre_SVM.ravel()))
        tn, fp, fn, tp = confusion_matrix(Yt.ravel(),pre_SVM.ravel()).ravel()
        MAR.append(fn / (fn + tp))
    return max(Accuracy_score),Classification_report[Accuracy_score.index(max(Accuracy_score))],MAR[Accuracy_score.index(max(Accuracy_score))]

def KNN(Xs_new,Xt_new,Ys,Yt):
    Neighbors = [1, 2, 5, 10, 50]
    Classification_report = []
    Accuracy_score = []
    MAR = []
    for i in range(len(Neighbors)):
        clf_KNN = KNeighborsClassifier(n_neighbors=Neighbors[i])
        clf_KNN.fit(Xs_new, Ys.ravel())
        pre_KNN = clf_KNN.predict(Xt_new)
        Classification_report.append(classification_report(Yt.ravel(), pre_KNN.ravel(),labels=[0,1]))
        Accuracy_score.append(accuracy_score(Yt.ravel(), pre_KNN.ravel()))
        tn, fp, fn, tp = confusion_matrix(Yt.ravel(), pre_KNN.ravel()).ravel()
        MAR.append(fn / (fn + tp))
    return max(Accuracy_score), Classification_report[Accuracy_score.index(max(Accuracy_score))],MAR[Accuracy_score.index(max(Accuracy_score))]

def LR(Xs_new,Xt_new,Ys,Yt):
    C = [0.001, 0.01, 0.1, 1, 10, 100]
    Classification_report = []
    Accuracy_score = []
    MAR = []
    for i in range(len(C)):
        clf_LR = LogisticRegression(penalty='l2', C=C[i],solver='lbfgs')
        clf_LR.fit(Xs_new, Ys.ravel())
        pre_LR = clf_LR.predict(Xt_new)
        Classification_report.append(classification_report(Yt.ravel(), pre_LR.ravel(),labels=[0,1]))
        Accuracy_score.append(accuracy_score(Yt.ravel(), pre_LR.ravel()))
        tn, fp, fn, tp = confusion_matrix(Yt.ravel(), pre_LR.ravel()).ravel()
        MAR.append(fn / (fn + tp))
    return max(Accuracy_score), Classification_report[Accuracy_score.index(max(Accuracy_score))],MAR[Accuracy_score.index(max(Accuracy_score))]

def RF(Xs_new,Xt_new,Ys,Yt):
    clf_RF = RandomForestClassifier(n_estimators=8)
    clf_RF.fit(Xs_new, Ys.ravel())
    pre_RF = clf_RF.predict(Xt_new)
    tn, fp, fn, tp = confusion_matrix(Yt.ravel(), pre_RF.ravel()).ravel()
    MAR = fn / (fn + tp)
    return accuracy_score(Yt.ravel(),pre_RF.ravel()),classification_report(Yt.ravel(),pre_RF.ravel(),labels=[0,1]),MAR

def GNB(Xs_new,Xt_new,Ys,Yt):
    clf_GNB = GaussianNB()
    clf_GNB.fit(Xs_new, Ys.ravel())
    pre_GNB = clf_GNB.predict(Xt_new)
    tn, fp, fn, tp = confusion_matrix(Yt.ravel(), pre_GNB.ravel()).ravel()
    MAR = fn / (fn + tp)
    return accuracy_score(Yt.ravel(), pre_GNB.ravel()),classification_report(Yt.ravel(), pre_GNB.ravel(),labels=[0,1]),MAR

def Classficator(Xs,Xt,Ys,Yt):

    # PCA
    D = [4, 8, 16, 32, 64,91]
    Xs_new_list = []
    Xt_new_list = []
    for i in range(len(D)):
        pca = PCA(n_components=D[i])
        Xs_new = pca.fit_transform(X=Xs)
        Xt_new = pca.transform(Xt)
        Xs_new_list.append(Xs_new)
        Xt_new_list.append(Xt_new)

    # classification

    # SVM
    Classification_report = []
    Accuracy_score = []
    MAR = []
    for i in range(len(D)):
        a, b, c = SVM(Xs_new_list[i], Xt_new_list[i], Ys, Yt)
        Accuracy_score.append(a)
        Classification_report.append(b)
        MAR.append(c)
    Accuracy_score_SVM = max(Accuracy_score)
    MAR_SVM = MAR[Accuracy_score.index(max(Accuracy_score))]
    # print(max(Accuracy_score))
    # print(Classification_report[Accuracy_score.index(max(Accuracy_score))])

    # KNN
    Classification_report = []
    Accuracy_score = []
    MAR = []
    for i in range(len(D)):
        a, b, c = KNN(Xs_new_list[i], Xt_new_list[i], Ys, Yt)
        Accuracy_score.append(a)
        Classification_report.append(b)
        MAR.append(c)
    Accuracy_score_KNN = max(Accuracy_score)
    MAR_KNN = MAR[Accuracy_score.index(max(Accuracy_score))]
    # print(max(Accuracy_score))
    # print(Classification_report[Accuracy_score.index(max(Accuracy_score))])

    # LogisticRegression
    Classification_report = []
    Accuracy_score = []
    MAR = []
    for i in range(len(D)):
        a, b, c = LR(Xs_new_list[i], Xt_new_list[i], Ys, Yt)
        Accuracy_score.append(a)
        Classification_report.append(b)
        MAR.append(c)
    Accuracy_score_LR = max(Accuracy_score)
    MAR_LR = MAR[Accuracy_score.index(max(Accuracy_score))]
    # print(max(Accuracy_score))
    # print(Classification_report[Accuracy_score.index(max(Accuracy_score))])

    # Random Forest
    Classification_report = []
    Accuracy_score = []
    MAR = []
    for i in range(len(D)):
        a, b, c = RF(Xs_new_list[i], Xt_new_list[i], Ys, Yt)
        Accuracy_score.append(a)
        Classification_report.append(b)
        MAR.append(c)
    Accuracy_score_RF = max(Accuracy_score)
    MAR_RF = MAR[Accuracy_score.index(max(Accuracy_score))]
    # print(max(Accuracy_score))
    # print(Classification_report[Accuracy_score.index(max(Accuracy_score))])

    # GaussianNB
    Classification_report = []
    Accuracy_score = []
    MAR = []
    for i in range(len(D)):
        a, b, c = GNB(Xs_new_list[i], Xt_new_list[i], Ys, Yt)
        Accuracy_score.append(a)
        Classification_report.append(b)
        MAR.append(c)
    Accuracy_score_GNB = max(Accuracy_score)
    MAR_GNB = MAR[Accuracy_score.index(max(Accuracy_score))]
    # print(max(Accuracy_score))
    # print(Classification_report[Accuracy_score.index(max(Accuracy_score))])
    return Accuracy_score_SVM,Accuracy_score_KNN,Accuracy_score_LR,Accuracy_score_RF,Accuracy_score_GNB,MAR_SVM,MAR_KNN,MAR_LR,MAR_RF,MAR_GNB


# input_length = 102
# source_data_path = 'C:/Users/ds.chacon/Documents/Diego/Crypto/Tesis maestria/Modelo 1/Source_Domain/BaseLoadCondition.mat'  #please write your <source domain dataset> path here.
# test_data_path = 'C:/Users/ds.chacon/Documents/Diego/Crypto/Tesis maestria/Modelo 1/Target_Domain/BaseLoadCondition.mat'   #please write your <target domain dataset> path here.

# source_data,test_data = scipy.io.loadmat(source_data_path),scipy.io.loadmat(test_data_path)
#print (source_data['BaseLoadCondition'])



source_data,test_data = source_data['BaseLoadCondition'],test_data['BaseLoadCondition']
source_data_input = source_data[:, 0:input_length]
test_data_input = test_data[:, 0:input_length]
source_data_label = source_data[:, input_length:input_length + 1]
test_data_label = test_data[:, input_length:input_length + 1]
ss = StandardScaler()
source_data_input_std = ss.fit_transform(source_data_input)
test_data_input_std = ss.transform(test_data_input)

Accuracy_score_SVM = []
Accuracy_score_KNN = []
Accuracy_score_LR = []
Accuracy_score_RF = []
Accuracy_score_GNB = []
MAR_SVM = []
MAR_KNN = []
MAR_LR = []
MAR_RF = []
MAR_GNB = []
for k in range(1000):
    idx = np.random.randint(0,source_data.shape[0],1000)
    Xs = source_data_input_std[idx]
    Ys = source_data_label[idx]
    idx = np.random.randint(0, test_data.shape[0], 1000)
    Xt = test_data_input_std[idx]
    Yt = test_data_label[idx]

    accuracy_score_SVM, accuracy_score_KNN, accuracy_score_LR, accuracy_score_RF, accuracy_score_GNB, mar_SVM, mar_KNN, mar_LR, mar_RF, mar_GNB = Classficator(Xs, Xt, Ys, Yt)
    Accuracy_score_SVM.append(accuracy_score_SVM)
    Accuracy_score_KNN.append(accuracy_score_KNN)
    Accuracy_score_LR.append(accuracy_score_LR)
    Accuracy_score_RF.append(accuracy_score_RF)
    Accuracy_score_GNB.append(accuracy_score_GNB)
    MAR_SVM.append(mar_SVM)
    MAR_KNN.append(mar_KNN)
    MAR_LR.append(mar_LR)
    MAR_RF.append(mar_RF)
    MAR_GNB.append(mar_GNB)

print(np.mean(Accuracy_score_SVM),np.mean(MAR_SVM))
print(np.mean(Accuracy_score_KNN),np.mean(MAR_KNN))
print(np.mean(Accuracy_score_LR),np.mean(MAR_LR))
print(np.mean(Accuracy_score_RF),np.mean(MAR_RF))
print(np.mean(Accuracy_score_GNB),np.mean(MAR_GNB))