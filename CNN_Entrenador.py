import sys
import os
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense, Activation
from keras.layers import  Convolution2D, MaxPooling2D
from keras import backend as K


###########################################################
K.clear_session()
 #Algoritmo de CNN = VGG16 aplicando KNN
 #Fuente: https://github.com/puigalex/AMP-Tech/blob/master/CNN%20desde%20cero/TransferLearning(VGG16).ipynb

def CrearModelo(modelo):
	#Datos de validadion y entrenamiento para el grupo de edad
	data_entrenamiento = './data/entrenamiento/'+modelo
	data_validacion = './data/validacion/'+modelo

	#Numero de capas 16
	#Parametros
	epocas=1 #Numero de veces que toda la data pasa por el modelo, cambiar a 5
	longitud, altura = 224, 224 #Tamaño de la imagen, cambiar a 150x150
	batch_size = 32 #Tamaño de lote fijo de imagenes para redes neuronales
	pasos = 100 #cambiar a 1000
	validation_steps = 20 # cambiar a 200
	filtrosConv1 = 32
	filtrosConv2 = 64
	#Capas de convolucion
	tamano_filtro1 = (3, 3)
	tamano_filtro2 = (2, 2)
	#Capas de agrupación
	tamano_pool = (2, 2) #Tamaño de agrupamiento
	clases = 2 #Numero de clases: Clase 1: Adolesecentes y Clase 2: No adolescentes [Puede haber mas de dos clases]
	#regresion logistica
	lr = 0.0004


	##Preparamos nuestras imagenes
	entrenamiento_datagen = ImageDataGenerator(
	    rescale=1./255, #re-escala
	    shear_range=0.3, #Rango de corte
	    zoom_range=0.3, #Rango de Zoom
	    horizontal_flip=True) #Volteo horizontal

	test_datagen = ImageDataGenerator(rescale=1./255)

	entrenamiento_generador = entrenamiento_datagen.flow_from_directory(
	    data_entrenamiento,
	    target_size=(altura, longitud),
	    batch_size=batch_size,
	    class_mode='categorical')

	validacion_generador = test_datagen.flow_from_directory(
	    data_validacion,
	    target_size=(altura, longitud),
	    batch_size=batch_size,
	    class_mode='categorical')
	#################################################################################################################
	############# Crea la red neuronal secuencial para que las capas se vayan generando en el orden correcto ########
	cnn = Sequential()
	###############################Primera Capa Secuencial #############################}
	#funcion de activación: RelU
	#Forma de Entrada(longitud, altura, canales)

	cnn.add(Convolution2D(filtrosConv1, tamano_filtro1, padding ="same", input_shape=(longitud, altura, 3), activation='relu'))
	cnn.add(MaxPooling2D(pool_size=tamano_pool))

	cnn.add(Convolution2D(filtrosConv2, tamano_filtro2, padding ="same"))
	cnn.add(MaxPooling2D(pool_size=tamano_pool))

	cnn.add(Flatten())
	cnn.add(Dense(256, activation='relu'))
	cnn.add(Dropout(0.5))
	cnn.add(Dense(clases, activation='softmax'))


	########################################################
	#Crear la red VGG16

	fperdida='binary_crossentropy' # Función de pérdidad cuando existen solo dos clases


	cnn.compile(loss=fperdida, 
	            optimizer=optimizers.Adam(lr=lr),
	            metrics=["binary_crossentropy","mean_squared_error",'accuracy'])

	cnn.fit_generator(
	    entrenamiento_generador,
	    steps_per_epoch=pasos,
	    epochs=epocas,
	    validation_data=validacion_generador,
	    validation_steps=validation_steps)

	# generar peso y modelo
	#target_dir = './modeloCNNcompleto/'+modelo+'/'
	target_dir = './'+modelo+'/'
	if not os.path.exists(target_dir):
		os.mkdir(target_dir)
	cnn.save(modelo+'/modelo.h5')
	cnn.save_weights(modelo+'/pesos.h5')

#####################################################################################
# Crear modelo: edad o genero      Dependiendo de lo que desee clasificar
# Genero C: genero Complet
# Genero : genero solo torso

# Edad C: edad completo
# Edad : edad solo torso

CrearModelo('edadCara')