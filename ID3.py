import operator

import numpy as np


def createDataSet():
	# 天气数据集
	# outlook:0：sunny// 1:overcast/ 2:rain
	# temperature:0:hot/1:mild/ 2:cool
	# humidity :0:high/ 1:normal
	# windy : 0:false/ 1:true
	dataSet = np.array([[0, 0, 0, 0, "N"],
	                    [0, 0, 0, 1, "N"],
	                    [1, 0, 0, 0, "Y"],
	                    [2, 1, 0, 0, "Y"],
	                    [2, 2, 1, 0, "Y"],
	                    [2, 2, 1, 1, "N"],
	                    [1, 2, 1, 1, "Y"]])

	labels = np.array(["outlook", "temperature", "humidity", "windy"])

	return dataSet, labels


# 创建测试集
def createTestSet():
	testSet = np.array([[0, 1, 0, 0],
	                    [0, 2, 1, 0],
	                    [1, 0, 0, 0],
	                    [2, 1, 1, 0],
	                    [0, 1, 1, 1],
	                    [1, 1, 0, 1],
	                    [1, 0, 1, 0],
	                    [2, 1, 0, 1]])

	return testSet


# 计算信息熵
def dataSet_entropy(dataSet):
	classLabel = dataSet[:, -1]  # 标签数据
	labelCount = {}  # 计算标签的数量
	for i in range(classLabel.size):
		label = classLabel[i]
		labelCount[label] = labelCount.get(label, 0) + 1

	# 计算信息熵
	ent = 0  # 熵值
	for k, v in labelCount.items():
		ent += -v / classLabel.size * np.log2(v / classLabel.size)
	return ent


# 切分子集

def splitDataSet(dataSet, featureIndex, value):

	subdataset = []  # 划分后的子集

	for example in dataset:
		if example[featureIndex] == value:
			subdataset.append(example)
	return np.delete(subdataset, featureIndex, axis=1)


# 选取最佳特征
def chooseBestFeature(dataset, labels):
	featureNum = labels.size  # 特征的个数
	minEntropy, bestFeatureIndex = 1, None  # 最小熵值
	n = dataset.shape[0]  # 样本的总数
	for i in range(featureNum):
		# 指定特征的条件熵
		featureEntropy = 0
		# 返回所有子集

		featureList = dataset[:, i]
		featureValues = set(featureList)
		for value in featureValues:
			subDataSet = splitDataSet(dataset, i, value)
			# 条件信息熵
			featureEntropy += subDataSet.shape[0] / n * dataSet_entropy(subDataSet)

		if minEntropy > featureEntropy:
			minEntropy = featureEntropy
			bestFeatureIndex = i

	return bestFeatureIndex

#选取类别中样本最多的数据

def mayorClass(classList):
	labelCount = {}
	for i in range(classList.size):
		label = classList[i]
		labelCount[label] = labelCount.get(label, 0) + 1
	sortedLabel = sorted(labelCount.items(), key=operator.itemgetter(1), reverse=True)  # 排序，从大到小，选出类别最多的那个
	return sortedLabel[0][0]


def createTree(dataset, labels):
	classList = dataset[:, -1]
	if len(set(classList)) == 1:  # 类别全部相同
		return dataset[:, -1][0]
	if labels.size == 0:  # 特征全部完，选数据量最多的那个
		return mayorClass(classList)
	# 递归
	bestFeatureIndex = chooseBestFeature(dataset, labels)

	bestFeature = labels[bestFeatureIndex]#选取最优特征
	dtree = {bestFeature:{}}#决策树的格式
	featureList = dataset[:,bestFeatureIndex]#最优特征的所有样本数据
	featureValues = set(featureList)#集合中不允许重复

	bestFeature = labels[bestFeatureIndex]  # 选取最优特征
	dtree = {bestFeature: {}}  # 决策树的格式
	featureList = dataset[:, bestFeatureIndex]
	featureValues = set(featureList)

	for value in featureValues:
		subdataset = splitDataSet(dataset, bestFeatureIndex, value)
		sublabels = np.delete(labels, bestFeatureIndex)  # 删除最优特征列
		dtree[bestFeature][value] = createTree(subdataset, sublabels)

	return dtree


# 单个样本进行 预测

def predict(tree, labels, testData):
	rootName = list(tree.keys())[0]#根节点
	rootValue = tree[rootName]
	featureIndex = list(labels).index(rootName)
	classLabel = None
	for key in rootValue.keys():

		if testData[featureIndex] ==int(key):
			if type(rootValue[key]).__name__=="dict":#字典格式进行递归
				classLabel = predict(rootValue[key],labels, testData)

		if testData[featureIndex] == int(key):
			if type(rootValue[key]).__name__ == "dict":
				classLabel = predict(rootValue[key], labels, testData)

		if testData[featureIndex] == int(key):
			if type(rootValue[key]).__name__ == "dict":
				classLabel = predict(rootValue[key], labels, testData)

			else:
				classLabel = rootValue[key]
	return classLabel


#多个样本预测

def predictAll(tree, labels, testSet):
	classLabels = []
	for i in testSet:
		classLabels.append(predict(tree, labels, i))
	return classLabels


if __name__ == "__main__":
	dataset, labels = createDataSet()
	tree = createTree(dataset, labels)
	testSet = createTestSet()
	print(predictAll(tree, labels, testSet))  # 对测试集进行测试
