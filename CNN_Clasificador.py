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

################################# modelo y modeloA ########################
#modelo = 'modeloA/modelo.h5'
#pesos_modelo = 'modeloA/pesos.h5'
##########################################################################
#Modelo solo para el torso
modelo = 'modeloC/modelo.h5'
pesos_modelo = 'modeloC/pesos.h5'

cnn = load_model(modelo)
cnn.load_weights(pesos_modelo)

# PREDECIR EDAD [MENOR DE 14 AÑOS] sin importar generos
def predecirEdad(file):
  x = load_img(file, target_size=(longitud, altura))
  x = img_to_array(x)
  x = np.expand_dims(x, axis=0)
  array = cnn.predict(x)
  result = array[0]
  answer = np.argmax(result)
  if answer == 0:
    return "NO"
  elif answer == 1:
    return "SI"
  return answer


# PREDECIR GENERO[H, M] sin importar edad
def predecirGenero(file):
  x = load_img(file, target_size=(longitud, altura))
  x = img_to_array(x)
  x = np.expand_dims(x, axis=0)
  array = cnn.predict(x)
  result = array[0]
  answer = np.argmax(result)
  if answer == 0:
    return "H"
  elif answer == 1:
    return "M"
  return answer

def crearCSV(archivo,opcion,n,di):
  f = open (archivo,'a')
  if opcion ==0:
    f.write("imagen;longitud;altura;canal;Cara_detectada;Ojos_detectados;Posicion_Cara;GeneroHAAR;Cuerpo_completo;Cuerpo_superior;Cuerpo_inferior;GeneroCNN")
    for i in range(1,n):
      foto=di+str(i)+".jpg"
      cara=Reconocimiento.DetectaCara(foto)
      ojos=Reconocimiento.DetectaOjos(foto)
      posicion=Reconocimiento.posicionCara(ojos)
      completo=ReconocerCuerpo.reconocerCuerpoCompleto(foto)
      superior=ReconocerCuerpo.reconocerCuerpoSuperior(foto)
      inferior=ReconocerCuerpo.reconocerCuerpoInferior(foto)
      genero=ReconocerGenero.procesar(foto)
      f.write("\n"+foto+";"+str(longitud)+";"+str(altura)+";"+canal+";"+cara+";"+str(ojos)+";"+posicion+";"+genero+";"+completo+";"+superior+";"+inferior+";"+predecirGenero(foto))

  elif opcion==1:    
    f.write("imagen;longitud;altura;canal;Cara_detectada;Ojos_detectados;Posicion_Cara;EdadHAAR;Cuerpo_completo;Cuerpo_superior;Cuerpo_inferior;EdadCNN")
    for i in range(1,n):
      foto=di+str(i)+".jpg"
      cara=Reconocimiento.DetectaCara(foto)
      ojos=Reconocimiento.DetectaOjos(foto)
      posicion=Reconocimiento.posicionCara(ojos)
      completo=ReconocerCuerpo.reconocerCuerpoCompleto(foto)
      superior=ReconocerCuerpo.reconocerCuerpoSuperior(foto)
      inferior=ReconocerCuerpo.reconocerCuerpoInferior(foto)
      edad=ReconocerEdad.procesar(foto)
      f.write("\n"+foto+";"+str(longitud)+";"+str(altura)+";"+canal+";"+cara+";"+str(ojos)+";"+posicion+";"+edad+";"+completo+";"+superior+";"+inferior+";"+predecirEdad(foto))
  
  print("............")
  f.close()
  print(archivo + "  generado correctamente")
################### Ejecutar metodo crearCSV ##########################################################

# Nombre del fichero
# 0: Detección del GENERO ó 1: Deteccion de la EDAD
# n: El numero de fotos
# directorio de imagenes de prueba

crearCSV("GENERO.csv",0,20,"data/test3/")
