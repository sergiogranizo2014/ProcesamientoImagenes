import cv2 as cv
import numpy as np

#Cargar imagenes 
#di="data/test4/"
di="data/entrenamiento/edad"
training_set=[]
training_labels=[]

for i in range(1,20):
	#foto=di+str(i)+".jpg"
	img = cv.imread(di+str(i)+".jpg")
	res=cv.resize(img,(250,250))
	gray_image = cv.cvtColor(res, cv.COLOR_BGR2GRAY)
	xarr=np.squeeze(np.array(gray_image).astype(np.float32))
	m,v=cv.PCACompute(xarr,mean=None)
	arr= np.array(v)
	flat_arr= arr.ravel()
	training_set.append(flat_arr)
	training_labels.append(1)
print("Carga exitosa")
print("version openCV")



#Entrenar Modelo

trainData=np.matrix(training_set, dtype=np.float32)
trainData=trainData.tolist()
responses=np.array(training_labels)
responses=responses.tolist()

################ Borrar despues ########


print(type(trainData))
print(trainData)
print(type(responses))
print(responses)



svm = cv.ml.SVM_create()
svm.setType(cv.ml.SVM_C_SVC)
svm.setKernel(cv.ml.SVM_LINEAR)
svm.setTermCriteria((cv.TERM_CRITERIA_MAX_ITER, 100, 1e-6))
svm.train(trainData, cv.ml.ROW_SAMPLE, responses)

svm.save("svm_data.dat")

print("Modelo SVM generado con exito")