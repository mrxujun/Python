# 逻辑回归 y = theta0 + theta1*x + theta2*x2
import numpy as np


def sigmoid(x):
	return 1 / (1 + np.exp(-x))


def weights(x_train, y_train):
	# 初始化参数
	m, n = x_train.shape
	theta = np.random.rand(n)
	# 学习率
	alpha = 0.001
	# 迭代次数
	cnt = 0
	# 阈值
	threshold = 0.01

	# 对打迭代次数
	max_iter = 50000

	while cnt < max_iter:
		cnt += 1
		diff = [0, 0, 0]
		for i in range(m):
			# 随机梯度下降
			diff = (y_train[i] - sigmoid(theta @ x_train[i])) * x_train[i]

			theta = theta + alpha * diff
			if (abs(diff) < threshold).all():
				break
	return theta


def predict(x_test, weights):
	if sigmoid(weights.T @ x_test) > 0.5:
		return 1
	else:
		return 0


if __name__ == "__main__":
	x_train = np.array([[1, 2.697, 6.254],
	                    [1, 1.872, 2.014],
	                    [1, 2.312, 0.812],
	                    [1, 1.983, 4.990],
	                    [1, 0.932, 3.920],
	                    [1, 1.321, 5.583],
	                    [1, 2.215, 1.560],
	                    [1, 1.659, 2.932],
	                    [1, 0.865, 7.362],
	                    [1, 1.685, 4.763],
	                    [1, 1.786, 2.523]])
	y_train = np.array([1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1])  # 对应的标签

    #得到学习到的参数进行预测
	weights = weights(x_train, y_train)
	print(predict([1, 2.697, 6.254], weights))
