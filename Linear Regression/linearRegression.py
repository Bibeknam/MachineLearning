import matplotlib.pyplot as plt
import numpy as np

def plotData(x,y):
	# plots the value of X and y in graph
	plt.figure()
	plt.plot(x,y,'rx');
	plt.ylabel('Profit in $10,000s');
	plt.xlabel('Population of City in 10,000s');
	#plt.show()


def computeCost(X,y,theta):
	#computes the cost of linear regression
	m = len(y)
	h = np.dot(X,theta)
	return (0.5/m * (np.dot(np.transpose(h - y),(h - y))))

def gradientDescent(X,y,theta, alpha, num_iteration):
	# performs gradient descent to learn theta taking
	# num_iters gradient steps with learning rate alpha
	m = len(y);
	for i in range(1,num_iteration):
		theta = theta - np.dot(np.transpose(X), (np.dot(X, theta) - y)) * alpha / m
	return theta


print 'Plotting data ...'

from numpy import genfromtxt
data = genfromtxt('data.txt', delimiter=',')

X = data[:,0] 
y = data[:,1]

plotData(X,y)

theta = np.zeros((2,1))

print 'Initial theta is ' + str(theta[0]) + ' and ' + str(theta[1])

# makes vector 2 dimensional explicitely
X = X.reshape(len(X),1)
y = y.reshape(len(y),1)

# adds a column on 1's to the X
X = np.hstack((np.ones((len(X),1)), X))

# initial cost for theta
J = computeCost(X,y,theta)

print 'The initial cost is ' + str(J)

# sets learning rate
alpha = 0.01

# sets number of iterations
num_iters = 1500

# performs gradient descent to minimize cost function
theta = gradientDescent(X,y,theta, alpha, num_iters)
plt.plot(X[:,1], np.dot(X,theta))
#plt.legend('right')
plt.show()