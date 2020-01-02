import numpy as np
import cv2
from tkinter import messagebox
from os import system

#Funcion para reconocer rostros en archivos jpg y png
def ReconocimientoFacial(foto):
	
	#Clasificadores XML

		#---------------CARA --------------
	
	profileface_cascade=cv2.CascadeClassifier('xml/haarcascade_profileface.xml') #cara de perfil
	lefteye_cascade=cv2.CascadeClassifier('xml/haarcascade_lefteye_2splits.xml') #ojo izquierdo
	righteye_cascade=cv2.CascadeClassifier('xml/haarcascade_righteye_2splits.xml') #ojo derecho
	
	face_cascade=cv2.CascadeClassifier('xml/haarcascade_frontalface_default.xml') #cara de frente
	eye_cascade=cv2.CascadeClassifier('xml/haarcascade_eye.xml') #ojos


	#-------------- CUERPO -------------
	fullbody_cascade=cv2.CascadeClassifier('xml/haarcascade_fullbody.xml') #cuerpo completo
	lowerbody_cascade=cv2.CascadeClassifier('xml/haarcascade_lowerbody.xml') #Parte inferior del cuerpo
	upperbody_cascade=cv2.CascadeClassifier('xml/haarcascade_upperbody.xml') #parte superior del cuerpo




	## Variables iniciales #####################
	caracteristicas=""
	img = cv2.imread(foto)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	###########################################
	faces = face_cascade.detectMultiScale(gray, 1.3, 5)
	print(type(faces))
	if len(faces)==0:
		caracteristicas="NO Detecta Cara"
		messagebox.showinfo("Python", "No se pudo realizar el Reconocimiento Facial")
	else:
		caracteristicas="Detecta Cara"
		for (x,y,w,h) in faces:
			cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
			roi_gray=gray[y:y+h,x:x+w]
			roi_color=img[y:y+h,x:x+w]
			eyes=eye_cascade.detectMultiScale(roi_gray)
			print("ojos")
			lo=9
			ojos=len(eyes)
			for (ex,ey,ew,eh) in eyes:
				cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
				cv2.imshow('img',img)
				cv2.waitKey(0)
				cv2.destroyAllWindows()

	return caracteristicas,ojos,lo

def DetectaCara(foto):
	#Librerias auxiliares para cara y ojos
	face_cascade=cv2.CascadeClassifier('xml/haarcascade_frontalface_default.xml')
	eye_cascade=cv2.CascadeClassifier('xml/haarcascade_eye.xml')
	## Variables iniciales #####################
	caracteristicas=""
	img = cv2.imread(foto)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	###########################################
	faces = face_cascade.detectMultiScale(gray, 1.03, 5)
	if len(faces)==0:
		caracteristicas="NO"
	else:
		caracteristicas="SI"
	return caracteristicas


def DetectaOjos(foto):
	#Librerias auxiliares para cara y ojos
	face_cascade=cv2.CascadeClassifier('xml/haarcascade_frontalface_default.xml')
	eye_cascade=cv2.CascadeClassifier('xml/haarcascade_eye.xml')
	## Variables iniciales #####################
	ojos=0
	img = cv2.imread(foto)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	###########################################
	faces = face_cascade.detectMultiScale(gray, 1.03, 5)

	if len(faces)==0:
		ojos=0
		rojos="NO"
	else:
		for (x,y,w,h) in faces:
			cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
			roi_gray=gray[y:y+h,x:x+w]
			roi_color=img[y:y+h,x:x+w]
			eyes=eye_cascade.detectMultiScale(roi_gray)
			ojos=len(eyes)
			rojos="SI"

	#return ojos
	return rojos

def posicionCara(entrada):
	pos=""
	if entrada==1:
		pos="perfil"
	elif entrada==2:
		pos="frontal"
	else:
		pos="NO"
	return pos

#data/entrenamiento/teen/1.jpg

foto=foto="data/entrenamiento/teen/1.jpg"
