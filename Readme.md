###Clasificación de Ataques en Smart Grids
##Descripción
Este proyecto tiene como objetivo clasificar y detectar ataques en redes de Smart Grids. Se utiliza una variedad de modelos de aprendizaje automático para realizar clasificaciones binarias o multiclases de diferentes tipos de ataques, basándose en patrones de consumo de energía, registros de dispositivos y otros datos relevantes.

##Modelos Implementados
KNN (K-Nearest Neighbors): Un algoritmo basado en la distancia que clasifica una entrada en función de cómo sus vecinos más cercanos están clasificados.
SVM (Support Vector Machines): Un modelo que busca encontrar el hiperplano que mejor divide el espacio de características para clasificar las entradas.
Random Forest (RF): Un ensamble de árboles de decisión que busca mejorar la precisión y robustez de las predicciones.
Gradient Boosting (GB): Un modelo que construye árboles de decisión de manera iterativa, corrigiendo errores de árboles previos.
Gaussian Naive Bayes (GNB): Un algoritmo basado en el teorema de Bayes que asume independencia entre las características.
##Dataset
El conjunto de datos consiste en registros de diversas redes de Smart Grids, incluyendo patrones de consumo de energía, registros de dispositivos y otros indicadores. La columna 'Label' indica si se ha producido un ataque (1) o no (0).

##Preprocesamiento
Antes de alimentar los datos a los modelos, se realiza un preprocesamiento que incluye:

Eliminación de columnas no relevantes.
Normalización o estandarización de los datos.
División del conjunto de datos en entrenamiento y prueba.
##Evaluación
Para evaluar el rendimiento de los modelos, se consideran las siguientes métricas:

Precisión
Recall
F1-score
Matriz de confusión

##Contribuciones
Este proyecto es de código abierto. Si deseas contribuir con mejoras o nuevos modelos, por favor crea un Pull Request.
