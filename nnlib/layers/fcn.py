import numpy as np

class FCN:
	def __init__(self,num_neurons, input_size):
		self.type    = "fcn" 
		self.num_neurons = num_neurons
		self.weights = np.random.randn(num_neurons, input_size)
		self.bias    = np.random.randn(num_neurons, 1)
		self.weight_grads = np.random.randn(num_neurons, input_size)
		self.bias_grads = np.random.randn(num_neurons, 1)
		self.input = None

	def forward(self,inputs):
		self.input = inputs
		return np.dot(self.weights,inputs) + self.bias

	def backward(self, upstream_grad):
		self.weight_grads = np.vstack([self.input.T for i in range(self.num_neurons)])*upstream_grad
		self.bias_grads   = upstream_grad
		return self.weights.T@upstream_grad

	def step(self, lr):
		self.weights -= lr*self.weight_grads
		self.bias    -= lr*self.bias_grads
