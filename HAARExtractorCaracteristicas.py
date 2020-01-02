import Reconocimiento
import ReconocerCuerpo
import ReconocerGenero
import ReconocerEdad
import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model

#Parámetros
longitud, altura = 150, 150 #Cambiar a 150x150
canal="RGB"


def crearCSV(archivo,n,di):
	f = open (archivo,'a')
	"""
  if opcion ==0:
    f.write("imagen;longitud;altura;canal;Cara_detectada;Ojos_detectados;Posicion_Cara;Cuerpo_completo;Cuerpo_superior;Cuerpo_inferior")
    for i in range(1,n):
      foto=di+str(i)+".jpg"
      cara=Reconocimiento.DetectaCara(foto)
      ojos=Reconocimiento.DetectaOjos(foto)
      posicion=Reconocimiento.posicionCara(ojos)
      completo=ReconocerCuerpo.reconocerCuerpoCompleto(foto)
      superior=ReconocerCuerpo.reconocerCuerpoSuperior(foto)
      inferior=ReconocerCuerpo.reconocerCuerpoInferior(foto)
      #genero=ReconocerGenero.procesar(foto)
      f.write("\n"+foto+";"+str(longitud)+";"+str(altura)+";"+canal+";"+cara+";"+str(ojos)+";"+posicion+";"+completo+";"+superior+";"+inferior)

  elif opcion==1:  
  """ 
	#f.write("imagen;longitud;altura;canal;Cara_detectada;Ojos_detectados;Posicion_Cara;Cuerpo_completo;Cuerpo_superior;Cuerpo_inferior")
	f.write("imagen;longitud;altura;canal;Cara_detectada;Ojos_detectados;Posicion_Cara;Torso_completo;Parte_alta;Parte_Baja")
	for i in range(11,n):
	  foto=di+str(i)+".jpg"
	  cara=Reconocimiento.DetectaCara(foto)
	  ojos=Reconocimiento.DetectaOjos(foto)
	  posicion=Reconocimiento.posicionCara(ojos)
	  completo=ReconocerCuerpo.reconocerCuerpoCompleto(foto)
	  superior=ReconocerCuerpo.reconocerCuerpoSuperior(foto)
	  inferior=ReconocerCuerpo.reconocerCuerpoInferior(foto)
	  f.write("\n"+foto+";"+str(longitud)+";"+str(altura)+";"+canal+";"+cara+";"+str(ojos)+";"+posicion+";"+completo+";"+superior+";"+inferior)
	  	
	f.close()
	print(archivo + "  generado correctamente")
################### Ejecutar metodo crearCSV ##########################################################

# Nombre del fichero
# 0: Detección del GENERO ó 1: Deteccion de la EDAD
# n: El numero de fotos
# directorio de imagenes de prueba
#ruta="data/test3/"
ruta="data/entrenamiento/teen/"

crearCSV("CaracteristicasTeen71.csv",71,ruta)
