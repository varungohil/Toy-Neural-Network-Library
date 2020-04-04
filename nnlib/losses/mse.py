import numpy as np

class MSE:
	def __init__(self):
		self.input = None

	def forward(self,prediction, gtruth):
		self.input = prediction
		return (prediction[0][0] - gtruth)**2

	def backward(self, upstream_grad):
		return 2*self.input