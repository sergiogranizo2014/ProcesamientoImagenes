###########################
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import SVM_Test
from sklearn import svm, metrics, datasets
from sklearn.utils import Bunch
from sklearn.model_selection import GridSearchCV, train_test_split
import skimage
from skimage.io import imread
from skimage.transform import resize
from sklearn.metrics import confusion_matrix #Matriz de confusión

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)

## Funcion para cargar imagenes, donde las clases estan separadas en subdirectorios
def cargar_imagenes(container_path, dimension=(150, 150)):#Cambiar a 64 x 64
    image_dir = Path(container_path)
    folders = [directory for directory in image_dir.iterdir() if directory.is_dir()]
    categories = [fo.name for fo in folders]

    descr = "Dataset de imagenes"
    images = []
    flat_data = []
    target = []
    for i, direc in enumerate(folders):
        for file in direc.iterdir():
            img = skimage.io.imread(file)
            img_resized = resize(img, dimension, anti_aliasing=True, mode='reflect')
            flat_data.append(img_resized.flatten()) 
            images.append(img_resized)
            target.append(i)
    flat_data = np.array(flat_data)
    target = np.array(target)
    images = np.array(images)

    return Bunch(data=flat_data,
                 target=target,
                 target_names=categories,
                 images=images,
                 DESCR=descr)
###############################################################################################################################
######################  DATOS DE ENTRENAMIENTO #################################



###########################################################################################################################################
#################################### Configuración para GENERO ############################################################################
#cargar imagenes de entrenamiento 
def clasificar(parte):
	print("############################ "+parte+" #########################################")
	image_dataset_genero = cargar_imagenes("data/entrenamiento/genero"+parte)
	Xg_train, Xg_test, yg_train, yg_test = train_test_split(image_dataset_genero.data, image_dataset_genero.target, test_size=0.20,random_state=21)
	param_gridg = [
	  {'C': [0.1,1,10, 100,1000], 'kernel': ['linear']},#linear
	  {'C': [0.1,1, 10,100,1000], 'gamma': [0.01, 0.0001], 'kernel': ['rbf']}, #rbf
	 ]


	


	###########################################################################################################################################
	#################################### Configuración para GRUPOS DE EDAD ####################################################################
	image_dataset_edad = cargar_imagenes("data/entrenamiento/edad"+parte)
	X_train, X_test, y_train, y_test = train_test_split(image_dataset_edad.data, image_dataset_edad.target, test_size=0.20,random_state=21)
	param_grid = [
	  {'C': [0.1,1,10, 100,1000], 'kernel': ['linear']},#linear
	  {'C': [0.1,1, 10,100,1000], 'gamma': [0.001, 0.00001], 'kernel': ['linear']}, #rbf
	 ]

	###Entrenar Datos con parametros de optimización, kernel lineal
	svc=svm.SVC()
	#######################################Entrenar datos de GENERO #################
	clfg = GridSearchCV(svc, param_gridg)
	clfg.fit(Xg_train, yg_train)
	yg_pred = clfg.predict(Xg_test)

	##########################################################################
	#######################################Entrenar datos de EDAD #################
	svc = svm.SVC()
	clf = GridSearchCV(svc, param_grid)
	clf.fit(X_train, y_train)
	y_pred = clf.predict(X_test)

	##############################################################################

	print("############################################################################")
	print("Reporte de clasificación GENERO \n{}:\n{}\n".format(clfg, metrics.classification_report(yg_test, yg_pred)))
	print("Matriz de confusion GENERO")
	mdc=confusion_matrix(yg_test, yg_pred)
	print(mdc)


	print("############################################################################")
	print("############################################################################")
	print("Reporte de clasificación GRUPOS DE EDAD \n{}:\n{}\n".format(clf, metrics.classification_report(y_test, y_pred)))

	print("Matriz de confusion GRUPOS EDAD")
	mdc=confusion_matrix(y_test, y_pred)
	print(mdc)
	print("############################ FIN #########################################")
	print("############################################################################")

#################################################################################################################################

clasificar("Cara")
clasificar("Torso")

