#data/entrenamiento/teen/1.jpg

  
import numpy as np
import cv2
def detectarCuerpo(foto):
		#Clasificadores XML para cuerpo y cara
	#-------------- CUERPO -------------
	body_cascade = cv2.CascadeClassifier('xml/haarcascade_fullbody.xml')
	upper_cascade = cv2.CascadeClassifier('xml/haarcascade_upperbody.xml')
	lower_cascade = cv2.CascadeClassifier('xml/haarcascade_lowerbody.xml')
	img = cv2.imread(foto)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	bodies = body_cascade.detectMultiScale(gray, 1.001, 5)
	upper = upper_cascade.detectMultiScale(gray, 1.001, 5)
	lower = lower_cascade.detectMultiScale(gray, 1.001, 5)

	if len(bodies)==0:
		print("NO bodies")
	else:
		for (x,y,w,h) in bodies:
			cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
			roi_gray = gray[y:y+h, x:x+w]
			roi_color = img[y:y+h, x:x+w]

	if len(upper)==0:
		print("NO upper")
	else:
		for (x,y,w,h) in upper:
			cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
			roi_gray = gray[y:y+h, x:x+w]
			roi_color = img[y:y+h, x:x+w]

	if len(lower)==0:
		print("NO lower")
	else:
		for (x,y,w,h) in lower:
			cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
			roi_gray = gray[y:y+h, x:x+w]
			roi_color = img[y:y+h, x:x+w]

	cv2.imshow('img',img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()


def reconocerCuerpoCompleto(foto):
		#Clasificadores XML para cuerpo y cara
	#-------------- CUERPO -------------
	body_cascade = cv2.CascadeClassifier('xml/haarcascade_fullbody.xml')
	img = cv2.imread(foto)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	bodies = body_cascade.detectMultiScale(gray, 1.01, 5)


	if len(bodies)==0:
		return "NO"
	else:
		return "SI"
		#for (x,y,w,h) in bodies:
		#	cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
		#	roi_gray = gray[y:y+h, x:x+w]
		#	roi_color = img[y:y+h, x:x+w]


def reconocerCuerpoSuperior(foto):
	#Clasificadores XML para cuerpo y cara
#-------------- CUERPO -------------

	upper_cascade = cv2.CascadeClassifier('xml/haarcascade_upperbody.xml')

	img = cv2.imread(foto)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


	upper = upper_cascade.detectMultiScale(gray, 1.01, 5)


	if len(upper)==0:
		return "NO"
	else:
		return "SI"
		#for (x,y,w,h) in upper:
		#	cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
		#	roi_gray = gray[y:y+h, x:x+w]
		#	roi_color = img[y:y+h, x:x+w]

def reconocerCuerpoInferior(foto):
	#Clasificadores XML para cuerpo y cara
#-------------- CUERPO -------------

	lower_cascade = cv2.CascadeClassifier('xml/haarcascade_lowerbody.xml')
	img = cv2.imread(foto)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	lower = lower_cascade.detectMultiScale(gray, 1.01, 5)

	if len(lower)==0:
		return "NO"
	else:
		return "SI"
		#for (x,y,w,h) in lower:
		#	cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
		#	roi_gray = gray[y:y+h, x:x+w]
		#	roi_color = img[y:y+h, x:x+w]

foto="https://pbs.twimg.com/media/EKekaKtXUAEpKc8.jpg"
print(reconocerCuerpoSuperior(foto))