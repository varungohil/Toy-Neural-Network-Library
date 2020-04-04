import numpy as np

class ReLU:
	def __init__(self):
		self.type = "relu"
		self.input = None

	def forward(self, inputs):
		self.input = inputs
		return np.clip(inputs, a_min = 0, a_max = np.inf)

	def backward(self, upstream_grad):
		return (self.input>0).astype(int)*upstream_grad


