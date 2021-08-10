# 决策树代码实现(python+numpy)

## 前言

我学习机器学习主要依据的是周志华的西瓜书和Peter Harrington的《机器学习实战》，理论部分可以通过西瓜书学习，看不下去的时候（西瓜书理论性强emmm你懂的）就看机器学习实战手敲下代码，两者结合，资源我放在网盘上了。

我这篇博客是在此基础之上，学习了理论之后想自己手打一遍决策树，但发现还是有些耗时。最后就放了点水，在《机器学习实战》代码的基础上利用numpy进行数据处理，一方面熟悉numpy为后面的学习做准备，一方面再巩固一下决策树。

## 用到的numpy知识

首先要知道numpy数组和一半的列表操作有很多类似之处，比如索引、切片。但numpy数组还有更多强大的功能，一起来看下吧。（我就总结下我用到的）

> np.array(object，...)

将object准换成ndarray

```python
 # 转为numpy数组进行处理
train_sample_list = np.array(train_sample_list)
train_label_list = np.array(train_label_list)
```

> np.hstack((a,b))

将a和b水平拼接（列方向）

类似的np.vstack((a,b))是将a,b垂直拼接（行方向）

```python
# # np.hstack 将数组横向拼接，也就是去除第axis维属性
return_sample_list = np.hstack((filtered_sample_list[...,:axis], filtered_sample_list[...,axis+1:]))
```

> np的布尔索引（这个太好用了）

```python
# sample_list[sample_list[...,axis] == value] 利用了numpy数组的布尔索引，只有对应行是ture的才被保存
filtered_sample_list = sample_list[sample_list[...,axis] == value]
```

目前就用到这么多。

## 决策树实现

我按照流程讲，完整代码和数据可以从我的github上下载

### 第一步：数据预处理

先读入数据，我这里是从txt文件中读取，就是西瓜书上的西瓜数据集2.0。我是将样本，标记分开存储的（sample_list, label_list）。

具体函数我都写了详细注解。

> 几个概念
>
> 样本（sample）/示例（instance）：一条记录
>
> 标记（label）: 样本结果信息（类别）
>
> 样例（example）：拥有标记的样本

**数据集：**



![](D:\output\机器学习总结\决策树\img\dataset.png)

**代码：**

```python
def getAll(src):
    """
    函数说明：
        读取文件，获取数据集
    Parameters：
        src - 文件地址（同一根目录下）
    Returns：
        sample_list - 样本集
        label_list - 标记集（类别集）
        attr_list - 属性集
    Author:
        Swagger
    Blog:
        https://blog.csdn.net/weixin_46684748
    Modify:
        2021/8/8
   """
    with open(src, encoding='utf-8', errors='ignore') as f:
        sample_list = []
        label_list = []
        attr_list = f.readline()[:-1].split(",")[1:]
        for line in f:
            if(len(line) > 0):
                example = line.split(",")[1:]
                sample_list.append(example[:-1])
                label_list.append(example[-1][:-1])
    return sample_list, label_list, attr_list

def getDataSet(test_size = 0.2):
    """
    函数说明：
        将数据集划分为训练集和测试集
    Parameters：
        test_size - 测试集的比例
    Returns：
        train_sample_list - 训练样本集
        train_label_list - 训练标记集
        test_sample_list - 测试样本集
        test_label_list - 测试标记集
        attr_list - 属性集
    Author:
        Swagger
    Blog:
        https://blog.csdn.net/weixin_46684748
    Modify:
        2021/8/8
   """
    sample_list, label_list, attr_list = getAll('watermelon_2.0.txt')
    # 划分训练集和测试集
    sample_label_list = list(zip(sample_list, label_list))      # 将sample_list, label_list打包
    random.shuffle(sample_label_list)
    random.shuffle(sample_label_list)
    index = int(len(sample_list) * test_size) + 1               # index为划分下表，利用切片进行划分
    train_list = sample_label_list[index:]
    test_list = sample_label_list[:index]
    train_sample_list, train_label_list = zip(*train_list)      # zip解压缩
    test_sample_list, test_label_list = zip(*test_list)

    # 转为numpy数组进行处理
    train_sample_list = np.array(train_sample_list)
    train_label_list = np.array(train_label_list)
    test_sample_list = np.array(test_sample_list)
    test_label_list = np.array(test_label_list)

    return train_sample_list, train_label_list, test_sample_list, test_label_list, attr_list
```

### 第二步：训练算法

决策树算法，我的理解是他其实还是个贪心。每次选择属性划分的时候他都要找到纯度最高的属性，这也就会导致局部最优，但不一定是全局最优。找最优划分属性其实就是在比较信息增益（我只用了信息增益），在找到最优划分属性之后就要对数据集按照最优属性的取值进行划分。我主要还是放代码，代码里面我都有详细注解。

##### 核心算法

首先要把核心算法实现，放张决策树算法的伪代码，递归结束条件有三种。

![](D:\output\机器学习总结\决策树\img\algorithm.png)

其中的重点是最优属性的划分。而这就涉及到纯度的数学化表示，这里我只采用了信息增益，具体的公式含义请看西瓜书P75。

公式1：信息熵的数学定义。（值越小，纯度越高）

![](D:\output\机器学习总结\决策树\img\公式1.png)

公式2：信息增益。（值越大，纯度越高）

**下面三个函数是在计算信息增益：**

```python
def calcShannonEnt(label_list):
    """
    函数说明：
        计算信息熵
        对应公式 Ent(D) = -∑ Pk*log2(Pk) k=1..len(label_set)
    Parameters：
        label_list - 标记集
    Returns：
        shannonEnt - 当前标记集的信息熵
    Author:
        Swagger
    Blog:
        https://blog.csdn.net/weixin_46684748
    Modify:
        2021/8/8
   """
    label_set = set(label_list)
    len_label_list = label_list.size
    shannonEnt = 0.0
    for label in label_set:
        prob = equalNums(label_list, label)/len_label_list
        shannonEnt -= prob * math.log2(prob)
    return shannonEnt

def conditionnalEntropy(feature_list, label_list):
    """
    函数说明：
        计算条件信息熵，对应信息增益公式中的被减项
    Parameters：
        feature_list - sample_list中某一列，表示当前属性的所有值
        label_list - 标记集
    Returns：
        entropy - 条件信息熵
    Author:
        Swagger
    Blog:
        https://blog.csdn.net/weixin_46684748
    Modify:
        2021/8/8
   """
    feature_list = np.asarray(feature_list)
    label_list = np.asarray(label_list)
    feature_set = set(feature_list)
    entropy = 0.0

    for feat in feature_set:
        pro = equalNums(feature_list, feat)/feature_list.size
        entropy += pro * calcShannonEnt(label_list[feature_list == feat])

    return entropy

def calcInfoGain(feature_list, label_list):
    """
    函数说明：
        计算信息增益
    Parameters：
        feature_list - sample_list中某一列，表示当前属性的所有值
        label_list - 标记集
    Returns：
        当前属性的信息增益
    Author:
        Swagger
    Blog:
        https://blog.csdn.net/weixin_46684748
    Modify:
        2021/8/8
   """
    return calcShannonEnt(label_list) - conditionnalEntropy(feature_list, label_list)
```

**下面两个函数是在比较选择最优划分属性，并划分数据集：**

```python
def splitDataSet(sample_list, label_list, axis, value):
    """
    函数说明：
        决策树在选好当前最优划分属性之后划分样本集
        依据value选择对应样例，并去除第axis维属性
    Parameters：
        feature_list - sample_list中某一列，表示当前属性的所有值
        label_list - 标记集
    Returns：
        return_sample_list, return_label_list
    Author:
        Swagger
    Blog:
        https://blog.csdn.net/weixin_46684748
    Modify:
        2021/8/8
   """
    # sample_list[sample_list[...,axis] == value] 利用了numpy数组的布尔索引
    filtered_sample_list = sample_list[sample_list[...,axis] == value]
    return_label_list = label_list[sample_list[...,axis] == value]
    # np.hstack 将数组横向拼接，也就是去除第axis维属性
    return_sample_list = np.hstack((filtered_sample_list[...,:axis], filtered_sample_list[...,axis+1:]))
    return return_sample_list, return_label_list

def chooseBestFeatureToSplit(sample_list, label_list):
    """
    函数说明：
        选取最优划分属性
    Parameters：
        sample_list - 样本集
        label_list - 标记集
    Returns：
        bestFeat_index - 最优划分属性的索引值
    Author:
        Swagger
    Blog:
        https://blog.csdn.net/weixin_46684748
    Modify:
        2021/8/8
   """
    numFeatures = sample_list.shape[1]
    bestInfoGain = 0
    bestFeat_index = -1

    for i in range(numFeatures):
        infoGain = calcInfoGain(sample_list[..., i], label_list)
        if infoGain > bestInfoGain:
            bestInfoGain = infoGain
            bestFeat_index = i
    return bestFeat_index
```

##### 生成决策树：

```python
def createTree(sample_list, label_list, attr_list_copy):
    """
    函数说明：
        生成决策树
    Parameters：
        sample_list - 样本集
        label_list - 标记集
        attr_list_copy - 属性集（之所以加copy是为了删属性的时候是在副本上，防止递归出错）
    Returns：
        myTree - 最终的决策树
    Author:
        Swagger
    Blog:
        https://blog.csdn.net/weixin_46684748
    Modify:
        2021/8/8
   """
    # attr_list 有del操作，不用副本的话递归会出错
    attr_list = attr_list_copy.copy()
    if len(set(label_list)) == 1:       # 如果只有一种标记，直接返回标记
        return label_list[0]
    elif sample_list.size == 0:         # 如果所有属性都被遍历，返回最多的标记
        return voteLabel(label_list)

    # bestFeat_index 最优划分属性的索引值
    # bestAttr 最优划分属性对应的名字
    bestFeat_index = chooseBestFeatureToSplit(sample_list, label_list)
    bestAttr = attr_list[bestFeat_index]
    myTree = {bestAttr: {}}
    del(attr_list[bestFeat_index])
    feat_set = set(sample_list[..., bestFeat_index])
    # 依据最优划分属性进行划分，并向下递归
    for feat in feat_set:
        return_sample_list, return_label_list = splitDataSet(sample_list, label_list, bestFeat_index, feat)
        myTree[bestAttr][feat] = createTree(return_sample_list, return_label_list, attr_list)
    return myTree

def voteLabel(label_list):
    """
    函数说明：
        这个函数是用在遍历完所有特征时，返回最多的类别
    Parameters：
        label_list: 标记列表
    Returns：
        数量最多的标记
    Author:
        Swagger
    Blog:
        https://blog.csdn.net/weixin_46684748
    Modify:
        2021/8/8
   """

    # unique_label_list 是label_list中标记种类列表
    # label_num 是unique_label_list对应的数量列表
    unique_label_list = list(set(label_list))
    label_num_list = []

    for label in unique_label_list:
        label_num_list.append(equalNums(label_list, label))

    # label_num.index(max(label_num))是label_num数组中最大值的下标
    return unique_label_list[label_num_list.index(max(label_num_list))]
```

### 第三步：测试算法

测试下来的正确率不是很高，因为算法本身就很简陋。后面考虑用后剪枝对算法进行优化，会再写个博客。

```python
def classify(decisionTree, testVec, attr_list):
    """
    函数说明：
        对tesVec进行分类
    Parameters：
        decisionTree - 决策树
        attr_list - 属性名列表
        testVec - 测试向量
    Returns：
        label - 预测的标记
    Author:
        Swagger
    Blog:
        https://blog.csdn.net/weixin_46684748
    Modify:
        2021/8/8
   """

    feature = list(decisionTree.keys())[0]          # feature为决策树的根节点
    feature_dict = decisionTree[feature]            # feature_dict为根节点下的子树
    feature_index = attr_list.index(feature)        # feature_index为feature对应的属性名索引
    feature_value = testVec[feature_index]          # feature_value为测试集中对应属性的值
    label = None
    if feature_value in feature_dict.keys():
        # 如果没有结果就继续向下找
        if type(feature_dict[feature_value]) == dict:
            label = classify(feature_dict[feature_value], testVec, attr_list)
        else:
            label = feature_dict[feature_value]
    return label

def testAccuracy(decisionTree, test_sample_list, test_label_list, attr_list):
    """
    函数说明：
        测试十次得到正确率均值
    Parameters：
        decisionTree - 决策树
        test_sample_list - 测试样本集
        test_label_list - 测试标记集
        attr_list - 属性集
    Returns：
        
    Author:
        Swagger
    Blog:
        https://blog.csdn.net/weixin_46684748
    Modify:
        2021/8/8
   """
    def oneTime(decisionTree, test_sample_list, test_label_list, attr_list):
        rightNum = 0
        predict_label_list = []
        for i in range(len(test_sample_list)):
            predict_label = classify(decisionTree, test_sample_list[i], attr_list)
            predict_label_list.append(predict_label)
            if predict_label == test_label_list[i]:
                rightNum += 1
        accuracy = rightNum/len(test_sample_list)
        return accuracy
    sum = 0.0
    for i in range(10):
        sum += oneTime(decisionTree, test_sample_list, test_label_list, attr_list)
    av_accuracy = sum/10
    return av_accuracy
```

正确率：

![](D:\output\机器学习总结\决策树\img\accuracy.png)

## 总结

虽然看起来算法很清晰，但是做的时候会出现很多问题，比如递归的时候涉及到改变原对象的，一定要用副本。还有numpy使用也要注意，搞清楚每个轴。真的很想把后剪枝，缺失值处理也自己写一遍，但确实太耗时了，光手写个决策树就用了我一个礼拜的时间，我对我的代码能力已经。。。接下来先把进度推进吧，代码可以有选择的练一练。