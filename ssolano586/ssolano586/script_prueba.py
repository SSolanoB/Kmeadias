from ssolano586 import unsupervised

import numpy as np

np.random.seed(1234)

mean = [4,3.5]
cov = [[0.25,0],[0,0.25]]
size = 1000
puntos_dist1 = np.random.multivariate_normal(mean, cov, size)

mean = [-1,-1]
cov = [[0.25,0],[0,0.25]]
size = 1000
puntos_dist2 = np.random.multivariate_normal(mean, cov, size)

mean = [-1,1]
cov = [[0.25,0],[0,0.25]]
size = 1000
puntos_dist3 = np.random.multivariate_normal(mean, cov, size)

mean = [1,-1]
cov = [[0.25,0],[0,0.25]]
size = 1000
puntos_dist4 = np.random.multivariate_normal(mean, cov, size)

puntos1 = np.append(puntos_dist1,puntos_dist2,axis=0)
puntos2 = np.append(puntos_dist3,puntos_dist4,axis=0)

puntos = np.append(puntos1, puntos2, axis = 0)

xprima2 = unsupervised.kmeans(puntos, 4)

print(xprima2)
