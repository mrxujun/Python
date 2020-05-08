# y = theta0 + theta1*x
import numpy as np
import matplotlib.pyplot as plt
X = [4, 8, 5, 10, 12]
y = [20, 50, 30, 70, 60]

theta0 = theta1 = 0
#学习率
alpha = 0.00001
# 迭代次数
cnt = 0
# 误差
error0 = error1 = 0
# 结束条件
threshold = 0.0000001

while True:
	# 梯度，diff[0]是theta0的梯度，diff[1]是theta1的梯度
	diff = [0, 0]
	for i in range(len(y)):
		diff[0] += y[i]-(theta0 + theta1*X[i])#线性回归中梯度的求解
		diff[1] += (y[i]-(theta0 + theta1*X[i]))*X[i]
	theta0 = theta0 + alpha *diff[0]
	theta1 = theta1 + alpha *diff[1]
	# 求误差
	for i in range(len(y)):
		error1 += (y[i]-(theta0 + theta1*X[i])*X[i])**2;
	error1 /=len(y)#求平均
	if abs(error1 - error0) < threshold:
		break
	else:
		error0 = error1

	cnt +=1
print(theta0, theta1, cnt)


# # 预测
def predict (theta0, theta1, x_test):
	return theta0 + theta1*x_test

print(predict(theta0, theta1, 15))