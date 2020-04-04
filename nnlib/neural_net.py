import numpy as np
from . import layers
from . import losses

class NeuralNet:
	def __init__(self,layer_list, loss_func, input_shape):
		assert(loss_func in ["mse"])
		self.loss = losses.MSE()
		self.layer_list = []
		self.input_shape = input_shape
		for item in layer_list:
			layer_info = item.split("-")
			if layer_info[0].lower() == "fcn":
				self.layer_list.append(layers.FCN( int(layer_info[1]), input_shape))
				input_shape = int(layer_info[1])
			elif layer_info[0].lower() == "relu":
				self.layer_list.append(layers.ReLU())

	def infer(self,data):
		output = data
		for layer in self.layer_list:
			output = layer.forward(output)
		return output

	def compute_loss(self, prediction, gtruth):
		return self.loss.forward(prediction, gtruth)

	def backprop(self):
		upstream_grad = 1
		upstream_grad = self.loss.backward(upstream_grad)
		for idx in range(1,len(self.layer_list)+1):
			layer_index = -1*idx
			upstream_grad = self.layer_list[layer_index].backward(upstream_grad)
	
	def step(self, lr):
		for layer in self.layer_list:
			if layer.type == "fcn":
				layer.step(lr)






