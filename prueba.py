import pandas as pd
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import numpy as np

def CalcularResults(conf_matrix):


    #print(conf_matrix)

    # Obtener el número total de clases
    num_classes = conf_matrix.shape[0]

    # Inicializar listas para almacenar los valores de TP, FP, TN y FN para cada clase
    TP_list = []
    FP_list = []
    TN_list = []
    FN_list = []

    # Calcular TP, FP, TN y FN para cada clase
    for i in range(num_classes):
        TP = conf_matrix[i, i]  # Verdaderos Positivos
        FP = sum(conf_matrix[:, i]) - TP  # Falsos Positivos
        FN = sum(conf_matrix[i, :]) - TP  # Falsos Negativos
        
        # Calcular TN (Verdaderos Negativos) excluyendo la fila y columna de la clase actual
        TN = np.sum(conf_matrix) - TP - FP - FN
        
        # Agregar los valores a las listas
        TP_list.append(TP)
        FP_list.append(FP)
        TN_list.append(TN)
        FN_list.append(FN)
        
    # Imprimir los valores para cada clase
    for i in range(num_classes):
        print(f"Clase {i}:")
        print("Verdaderos Positivos (TP):", TP_list[i])
        print("Falsos Positivos (FP):", FP_list[i])
        print("Verdaderos Negativos (TN):", TN_list[i])
        print("Falsos Negativos (FN):", FN_list[i])

def GB(Xs_new,Xt_new,Ys,Yt):
    clf_RF = GradientBoostingClassifier(n_estimators=500, learning_rate=1.0, max_depth=1, random_state=0)
    clf_RF.fit(Xs_new, Ys.ravel())
    pre_RF = clf_RF.predict(Xt_new)
    tn, fp, fn, tp = confusion_matrix(Yt.ravel(), pre_RF.ravel()).ravel()
    MAR = fn / (fn + tp)
    return accuracy_score(Yt.ravel(),pre_RF.ravel()),classification_report(Yt.ravel(),pre_RF.ravel(),labels=[0,1]),MAR

def RF(Xs_new,Xt_new,Ys,Yt):
    clf_RF = RandomForestClassifier(n_estimators=8)
    clf_RF.fit(Xs_new, Ys.ravel())
    pre_RF = clf_RF.predict(Xt_new)
    tn, fp, fn, tp = confusion_matrix(Yt.ravel(), pre_RF.ravel()).ravel()
    MAR = fn / (fn + tp)
    return accuracy_score(Yt.ravel(),pre_RF.ravel()),classification_report(Yt.ravel(),pre_RF.ravel(),labels=[0,1]),MAR

def SVM(Xs_new,Xt_new,Ys,Yt):
    C = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 4, 8, 16, 32]
    Classification_report = []
    Accuracy_score = []
    MAR = []
    for i in range(len(C)):
        clf_SVM = SVC(kernel='rbf', C=C[i],gamma='auto')
        clf_SVM.fit(X=Xs_new, y=Ys)
        pre_SVM = clf_SVM.predict(Xt_new)
        class_report = classification_report(Yt.ravel(),pre_SVM.ravel(), zero_division=1)
        #print(class_report)
        Classification_report.append(class_report)
        Accuracy_score.append(accuracy_score(Yt.ravel(),pre_SVM.ravel()))
        conf_matrix = confusion_matrix(Yt.ravel(), pre_SVM.ravel())
        # Desempaqueta los valores de la matriz de confusión
        tn, fp, fn, tp = conf_matrix.ravel()
        MAR.append(fn / (fn + tp))
        print("Termino parametro:"+str(i))
    return max(Accuracy_score),Classification_report[Accuracy_score.index(max(Accuracy_score))],MAR[Accuracy_score.index(max(Accuracy_score))]


def KNN(Xs_new,Xt_new,Ys,Yt):
    Neighbors = [1, 2, 5, 10, 50]
    Classification_report = []
    Accuracy_score = []
    conf_matrix_list=[]
    for i in range(len(Neighbors)):
        clf_KNN = KNeighborsClassifier(n_neighbors=Neighbors[i])
        clf_KNN.fit(Xs_new, Ys.ravel())
        pre_KNN = clf_KNN.predict(Xt_new)
        Classification_report.append(classification_report(Yt.ravel(), pre_KNN.ravel(),zero_division=1))
        Accuracy_score.append(accuracy_score(Yt.ravel(), pre_KNN.ravel()))
        conf_matrix = confusion_matrix(Yt.ravel(), pre_KNN.ravel())
        conf_matrix_list.append( conf_matrix )
    index_max=Accuracy_score.index(max(Accuracy_score))
    ci = conf_matrix_list[index_max]
    fn=ci[1][0]
    tp=ci[0][0]
    MAR = fn / (fn + tp)
    #CalcularResults(conf_matrix_list[index_max])
    return max(Accuracy_score), Classification_report[index_max],MAR

def GNB(Xs_new,Xt_new,Ys,Yt):
    #print("Comienza GNB:")
    clf_GNB = GaussianNB()
    clf_GNB.fit(Xs_new, Ys.ravel())
    pre_GNB = clf_GNB.predict(Xt_new)
    class_report = classification_report(Yt.ravel(),pre_GNB.ravel(), zero_division=1)
    conf_matrix = confusion_matrix(Yt.ravel(), pre_GNB.ravel())
    tn, fp, fn, tp = conf_matrix.ravel()
    MAR = fn / (fn + tp)
    return accuracy_score(Yt.ravel(), pre_GNB.ravel()),class_report,MAR

def Classficator(Xs,Xt,Ys,Yt):

    # classification

    Classification_report = []
    Accuracy_score = []
    mar_score = []

    # GNB
    print("Comienza classification GNB")
    a, b , m = GNB(Xs, Xt, Ys, Yt)
    print("GNB accuracy: "+ str (a))
    Accuracy_score.append(a)
    print(b)
    Classification_report.append(b)
    mar_score.append(m)
    print("GNB MAR: "+ str (m) + "\n")
 
    # KNN
    print("Comienza classification KNN")
    a, b , m  = KNN(Xs,Xt, Ys, Yt)
    print("KNN accuracy: "+ str (a))
    Accuracy_score.append(a)
    print(b)
    Classification_report.append(b)
    mar_score.append(m)
    print("KNN MAR: "+ str (m) + "\n")

    #SVM
    print("Comienza classification Randomforest")
    a, b , m  = RF(Xs,Xt, Ys, Yt)
    print("RF accuracy: "+ str (a))
    Accuracy_score.append(a)
    print(b)
    Classification_report.append(b)
    mar_score.append(m)
    print("RF MAR: "+ str (m) + "\n" )

    # #GB
    # print("Comienza classification Gradientboost")
    # a, b , m  = GB(Xs,Xt, Ys, Yt)
    # print("RF accuracy: "+ str (a))
    # Accuracy_score.append(a)
    # print(b)
    # Classification_report.append(b)
    # mar_score.append(m)
    # print("RF MAR: "+ str (m))

    #print(max(Accuracy_score))
    return 0 


# Nombre del archivo de texto
archivo_txt = 'raw_data_transform_1.txt'

# Leer el archivo de texto y crear un DataFrame
df = pd.read_csv(archivo_txt, delimiter=',', header=None)

# Asignar nombres de columnas a las primeras 8 columnas
columnas = ['macAddress', 'date', 'logid', 'funcid', 'time', 'energy', 'Cpu', 'Label', 'LabelString']
df.columns = columnas

# Eliminar la columna original de la etiqueta
#df = df.drop(columns=[8])

# Dividir el DataFrame en conjuntos de entrenamiento y prueba (70% - 30%)
X = df.drop(columns=['macAddress','Label','LabelString'])  # Características (todas las columnas excepto 'Label')
y = df['Label']  # Etiquetas (columna 'Label')

# Dividir en 70% para entrenamiento y 30% para prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Verificar distribución inicial de las clases
unique, counts = np.unique(y_train, return_counts=True)
print("Distribución original:", dict(zip(unique, counts)))

# Aplicamos SMOTE al conjunto de datos de entrenamiento
sm = SMOTE(random_state=0, sampling_strategy=0.1)
X_train, y_train = sm.fit_resample(X_train, y_train)

# Verificar distribución de las clases después de SMOTE
unique, counts = np.unique(y_train, return_counts=True)
print("Distribución después de SMOTE:", dict(zip(unique, counts)))

# Aplicamos RandomUnderSampler
rus = RandomUnderSampler(random_state=0, sampling_strategy=0.12)
X_train, y_train = rus.fit_resample(X_train, y_train)

# Verificar distribución de las clases después de RandomUnderSampler
unique, counts = np.unique(y_train, return_counts=True)
print("Distribución después de RandomUnderSampler:", dict(zip(unique, counts)))

# Normalizar datos
ss = StandardScaler()
source_data_input_std = ss.fit_transform(X_train)
test_data_input_std = ss.transform(X_test)

Accuracy_score_SVM = []

Classficator(source_data_input_std, test_data_input_std, y_train, y_test)

#print("Resultados")
