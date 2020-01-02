# MODULO PREDICCION DE EDAD

import cv2 as cv
import math
import time
import argparse
from tkinter import messagebox


def getFaceBox(net, frame, conf_threshold=0.7):
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]

    # Aplicacion de Redes Neuronales para clasificar
    blob = cv.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections = net.forward()
    bboxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            bboxes.append([x1, y1, x2, y2])
            cv.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight/150)), 8)
    return frameOpencvDnn, bboxes

def procesar(foto):
	parser = argparse.ArgumentParser(description='Use este script para ejecutar el reconocimiento de edad y género usando OpenCV.')
	parser.add_argument('--input', help='Ruta de acceso al archivo de imagen o video.')

	args = parser.parse_args()
	###################### MODELOS ##########################
	faceProto = "lib/opencv_face_detector.pbtxt"
	faceModel = "lib/opencv_face_detector_uint8.pb"
	ageProto = "lib/age_deploy.prototxt"
	ageModel = "lib/age_net.caffemodel"
	genderProto = "lib/gender_deploy.prototxt"
	genderModel = "lib/gender_net.caffemodel"

	#Modelos de valor medio
	MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

	#Edades
	#ageList = ['< 14', '> 14']
	ageList = ['< 14', '< 14', '< 14', '> 14', '> 14', '> 14', '> 14', '> 14']
	#generos
	genderList = ['Hombre', 'Mujer']

	#Carga de modelos
	ageNet = cv.dnn.readNet(ageModel, ageProto)
	genderNet = cv.dnn.readNet(genderModel, genderProto)
	faceNet = cv.dnn.readNet(faceModel, faceProto)
	cap = cv.VideoCapture(foto)
	padding = 20
	genero="No detectado"
	edad="No detectado"


	while cv.waitKey(1) < 0:

	    t = time.time()
	    hasFrame, frame = cap.read()
	    if not hasFrame:
	        cv.waitKey()
	        break

	    frameFace, bboxes = getFaceBox(faceNet, frame)
	    if not bboxes:
	        edad="No detectado"
	        break

	    for bbox in bboxes:
	        face = frame[max(0,bbox[1]-padding):min(bbox[3]+padding,frame.shape[0]-1),
	        max(0,bbox[0]-padding):min(bbox[2]+padding, frame.shape[1]-1)]
	        
	        blob = cv.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
	        #Analisis de Genero
	        genderNet.setInput(blob)
	        genderPreds = genderNet.forward()
	        gender = genderList[genderPreds[0].argmax()]
	        genero=gender

	        #Analisis de Edad
	        ageNet.setInput(blob)
	        agePreds = ageNet.forward()
	        edad = ageList[agePreds[0].argmax()]      

	return edad