import nnlib
import pandas as pd
import numpy as np


if __name__ == '__main__':
	input_data_size = 6
	layer_list = [
			"fcn-5",
			"relu",
			"fcn-5",
			"relu",
			"fcn-1"
				]
	loss_func = "mse"

	nn = nnlib.NeuralNet(layer_list, loss_func, input_data_size)

	num_iter = 10
	lr = 0.0001

	data = pd.read_excel('realestate.xlsx')
	data = data.drop('No', axis=1)
	columns = list(data.columns)
	id2columns = {i: columns[i] for i in range(len(columns))}
	data.columns = list(range(len(columns)))

	data=(data-data.min())/(data.max()-data.min())
	X = data.drop(6, axis=1)
	y = data[6]

	split = int(len(X)*0.7)
	x_train = X[: split]
	y_train = y[: split]
	x_test = X[split: ]
	y_test = y[split: ]

	num_epoch = 1000	
	for epoch in range(num_epoch):
		print(f"Epoch {epoch} ______________________________________")
		for index in range(x_train.shape[0]):
			output = nn.infer(np.array(x_train.loc[0, :]).reshape(-1,1))
			# print(output)
			loss = nn.compute_loss(output, y_train.loc[0])
			nn.backprop()
			nn.step(lr)
			print(f"Iteration {index}, Loss : {loss}, Output: {output}, GT: {y_train.loc[index]}")

