import operator

import numpy as np

# 对ID3的改进
# 1.用信息增益率代替信息增益
# 2.能够完成对连续属性的离散化处理
# 3.能处理缺失值的情况
# 4.在决策树构造完成之后进行剪枝

def createDataSet():
	# 天气数据集
	# outlook:0：sunny/ 1:overcast/ 2:rain
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
def splitDataSet(dataset, featureIndex, value):
	subdataset = []  # 划分后的子集

	for example in dataset:
		if example[featureIndex] == value:
			subdataset.append(example)
	return np.delete(subdataset, featureIndex, axis=1)


# 选取最佳特征
def chooseBestFeature(dataset, labels):
	featureNum = labels.size  # 特征的个数
	baseEntropy = dataSet_entropy(dataset)
	maxRatio, bestFeatureIndex = 0, None  # 最小熵值
	n = dataset.shape[0]  # 样本的总数
	minGini = 1
	for i in range(featureNum):
		gini = 0
		# 返回所有子集

		featureList = dataset[:, i]
		featureValues = set(featureList)
		for value in featureValues:
			subDataSet = splitDataSet(dataset, i, value)
			pi = subDataSet.shape[0]/n
			# 基尼值
			gini += (1-classLabelPi(subDataSet)) * pi



		if minGini > gini:
			minGini = gini
			bestFeatureIndex = i
	return bestFeatureIndex

def classLabelPi(dataset):
	classLabel = dataset[:, -1]  # 标签数据
	labelCount = {}  # 计算标签的数量
	for i in range(classLabel.size):
		label = classLabel[i]
		labelCount[label] = labelCount.get(label, 0) + 1
	valueList = list(labelCount.values())
	sum = np.sum(valueList)
	pi = 0
	for i in valueList:
		pi +=(i/sum)**2
	return pi

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
	bestFeature = labels[bestFeatureIndex]  # 选取最优特征
	dtree = {bestFeature: {}}  # 决策树的格式
	featureList = dataset[:, bestFeatureIndex]
	featureValues = set(featureList)
	for value in featureValues:
		subdataset = splitDataSet(dataset, bestFeatureIndex, value)
		sublabels = np.delete(labels, bestFeatureIndex)  # 删除最优特征列
		dtree[bestFeature][value] = createTree(subdataset, sublabels)

	return dtree


def predict(tree, labels, testData):
	rootName = list(tree.keys())[0]
	rootValue = tree[rootName]
	featureIndex = list(labels).index(rootName)
	classLabel = None
	for key in rootValue.keys():
		if testData[featureIndex] == int(key):
			if type(rootValue[key]).__name__ == "dict":
				classLabel = predict(rootValue[key], labels, testData)
			else:
				classLabel = rootValue[key]
	return classLabel


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