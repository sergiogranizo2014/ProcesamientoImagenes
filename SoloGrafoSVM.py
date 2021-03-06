#from sklearn import svm
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# CLASIFICADOR SVM
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import make_blobs

# creamos 100 puntos
X, y = make_blobs(n_samples=100, centers=2, random_state=6)

#ajusta el modelo
clf = svm.SVC(kernel='linear', C=1000)
clf.fit(X, y)

plt.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=plt.cm.Paired)

# plotea la decision funcion
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

# crea la malla para evaluar el modelo
xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = clf.decision_function(xy).reshape(XX.shape)

# plotea decision margenes y fronteras
ax.contour(XX, YY, Z, colors='g', levels=[-1, 0, 1], alpha=0.5, linestyles=['-.', '-', '-.'])
# plotea support vectors
ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=100, linewidth=1, facecolors='none')
plt.show()