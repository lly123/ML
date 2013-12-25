from numpy import *
import pylab as pl

def generateDataSet(num, d):
	k = mat(zeros((num, 2)))
	for i in range(num):
		k[i,:] = d[0,:] + [(d[1, 0] - d[0, 0]) * random.rand(), (d[1, 1] - d[0, 1]) * random.rand()]
	return k

def distEclud(vecA, vecB):
	return sqrt(sum(power(vecA - vecB, 2)))

def randCent(dataSet, k):
	n = shape(dataSet)[1]
	centroids = mat(zeros((k,n)))
	for j in range(n):
		minJ = min(dataSet[:,j])
		rangeJ = float(max(dataSet[:,j]) - minJ)
		centroids[:,j] = minJ + rangeJ * random.rand(k,1)
	return centroids

def kMeans(dataSet, k, distMeas = distEclud, createCent = randCent):
	m = shape(dataSet)[0]
	clusterAssment = mat(zeros((m,2)))
	centroids = createCent(dataSet, k)
	
	clusterChanged = True
	while clusterChanged:
		clusterChanged = False
	
		for i in range(m):
			minDist = inf; minIndex = -1
			for j in range(k):
				distJI = distMeas(centroids[j,:],dataSet[i,:])
				if distJI < minDist:
					minDist = distJI; minIndex = j
			if clusterAssment[i,0] != minIndex: clusterChanged = True
			clusterAssment[i,:] = minIndex, minDist ** 2
	
		for cent in range(k):
			ptsInClust = dataSet[nonzero(clusterAssment[:,0].A==cent)[0]]
			centroids[cent,:] = mean(ptsInClust, axis=0)
	
	return centroids, clusterAssment

d1 = generateDataSet(50, mat([[1, 1], [60, 60]]))
d2 = generateDataSet(50, mat([[50, 1], [120, 80]]))
d3 = generateDataSet(50, mat([[30, 50], [90, 120]]))
dataSet = mat(append(append(d1.A, d2.A, axis = 0), d3.A, axis = 0))

centroids, clusterAssment = kMeans(dataSet, 8)

pl.plot(dataSet[:,0], dataSet[:,1], 'ro')
pl.plot(centroids[:,0], centroids[:,1], 'bo')
pl.show()
