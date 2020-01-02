import cv2
import numpy as np
import SVM_Entrenador
import skimage
from skimage.io import imread
from skimage.transform import resize
from sklearn.model_selection import GridSearchCV, train_test_split
from pathlib import Path

#Cargar imagenes directamente desde un directorio
def cargar_imagenes_prueba(container_path, dimension=(150, 150)):#Cambiar a 64 x 64
    image_dir = Path(container_path)
    images = []
    flat_data = []
    target = []
    for file in image_dir.iterdir() :
        img = skimage.io.imread(file)
        img_resized = resize(img, dimension, anti_aliasing=True, mode='reflect')
        flat_data.append(img_resized.flatten()) 
        images.append(img_resized)
    flat_data = np.array(flat_data)
    target = np.array(target)
    images = np.array(images)
    return flat_data

print("################") 
image_dataset=cargar_imagenes_prueba("carpeta/")
print("Imprime dataset type")
print(type(image_dataset))
print(len(image_dataset))
#print("RESULTADOS :::::: USANDO IMAGENES TWITTER")
#Este método funciona siempre y cuando existan 28 imágenes
print(SVM_Entrenador.clasificarGruposEdad("Torso",image_dataset))
