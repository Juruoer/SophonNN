# Sophon
A Simple Neural Network Framework

最近一段时间在学习机器学习，学习过程中也自己实现了一个简单的深度学习框架。

## 简介
这个框架使用 `python` 语言和 `numpy` 库搭建，包括常见的激活函数（`linear`、`relu`、`sigmoid`、`softmax`）和损失函数（平方差损失、二元交叉熵、稀疏分类交叉熵），以及 `CNN` 中需要的卷积层和池化层。实现了正向传播和反向传播（但还没有研究自动求导，但这个简单的框架中实现了对上述激活函数和损失函数的求导）。实现了一些优化手段，包括 `miniBatch`、`momentum`、`RMSprop`、`Adam`。这个框架还有很多东西没有完善，实际最后也不会去使用他，但目的在于检验学习成果并更好的了解神经网络运行的过程。


## 源码
框架的源码在 [SophonNN](https://github.com/Juruoer/SophonNN/tree/master)。


## 使用方法
### 导入 sophon
首先确保当前计算机已经配置了 `Python` 环境，并且安装了 `numpy` 库。

下载好 `Sophon` 源码解压缩到任意一个文件夹下，记住 `Sophon` 文件夹路径，例如：`G:\Machine_Learning\Sophon\`。

```
G:\Machine_Learning
├─Sophon
│  │
│  ├─sophon
```

然后找到计算机中存放 `python` 包的目录，`Anaconda` 环境一般在 `\Anaconda\Lib\site-packages`，非 Anaconda 环境一般在 `\Pythonxx\Lib\site-packages`（`xx`指 `Python` 版本），在 `site-packages` 文件夹下创建 `cus_sophon.pth` 文件，使用记事本打开并写入 `Sophon` 文件夹的路径，例如上述的 `G:\Machine_Learning\Sophon\` (注意最后一定要有 `\`)。

## 构建神经网络示例

 假设需要一个模型来判断烘烤咖啡豆的最佳时间和温度，并且假设下图中红叉表示的为优秀的烘烤情况，蓝圈表示失败的烘烤情况，数据来自 [deeplearning.ai 的神经网络课程](https://www.coursera.org/learn/neural-networks-deep-learning)。

![coffdate](https://user-images.githubusercontent.com/53604157/222099861-a7e74672-bbe5-499d-ad15-3d737bd5c609.png)

创建一个 `.py` 文件，写入以下代码即可构建一个两层的二元分类神经网络来完成该任务：

```python
import sophon
import sophon.neuralNetwork.layers as layers
from sophon.neuralNetwork.nn import sequence
import matplotlib.pyplot as plt # 这个示例中需要使用 matplotlib 画出最终测试的结果图，如果没有安装 matplotlib，可以将下面使用 plt 的代码注释，这不会影响程序运行，只是缺少图像直观的检测结果

# 创建训练数据，这是一个模拟烤咖啡的最佳时间和温度的数据
def load_coffee_data(trainSize:int, testSize:int):
    """
    创建靠咖啡的时间和温度的数据
    烘烤时间: 在 12-15 分钟之间最佳
    温度范围: 在 175-260 C 之间最佳

    Parameters
    ----------
    trainSize : int
        训练集大小
    testSize : int
        测试集大小

    Returns
    -------
    TYPE
        训练集和测试集

    """
    rng = np.random.default_rng(2)
    # 生成 (trainSize + testSize) * 2 个在 0 到 1 之间的随机数
    # 改变形状为 X: [trainSize + testSize, 2]
    X = rng.random((trainSize + testSize) * 2).reshape(-1,2)
    X[:,0] = X[:,0] * (285-150) + 150  # X[i, 0] 表第 i 个示例的温度，将它们分散到 150 到 285 之间
    X[:,1] = X[:,1] * 4 + 11.5          # X[i, 1] 表第 i 个示例的时间，将它们分散到 11.5 到 15.5 之间
    Y = np.zeros(len(X)) # 初始化标签
    
    i=0
    for t,d in X:
        # 并不是满足温度在 175 ~ 260 之间且时间在 12 ~ 15 度之间的烘烤方案就一定好
        # 温度和之间之间的关系也需要衡量，温度高时烘烤时间要更短，反之则反
        # 以 t 和 d 为坐标轴，最佳的烘烤方案其实在一个三角形范围
        y = -3/(260-175)*t + 21
        if (t > 175 and t < 260 and d > 12 and d < 15 and d<=y ):
            Y[i] = 1
        else:
            Y[i] = 0
        i += 1
    Y = Y.reshape(-1, 1)
    # X: [trainSize + testSize, 2], Y: [trainSize + testSize, 1]
    return X[:trainSize, :], Y[:trainSize, :], X[trainSize:,:], Y[trainSize:, :]


# 获取数据
print("\n靠咖啡豆测试")
train_x, train_y, test_x, test_y = load_coffee_data(400, 50)

# 特征值相差很大，归一化
mu = np.mean(train_x, axis=0, keepdims=True)
sigma = np.std(train_x, axis=0, keepdims=True)
train_x_norm = somath.normalize(train_x, mu, sigma)
test_x_norm = somath.normalize(test_x, mu, sigma) # 注意测试集的归一化应当使用训练集的分布

# 构建两层网络
nnet = sequence(train_x_norm.shape[1:],  # 输入数据的形状
        [layers.dense(3, "sigmoid"),     # 一个 3 个神经元的 sigmoid 层
         layers.dense(1, "sigmoid")],    # 一个 1 个神经元的 sigmoid 层
        1234)                            # 随机种子，以影响权重和偏移量的初始化

nnet.showOutputShape() # 显示每层的输出形状

## 方案一：学习率为 0.01，不使用优化器
# nnet.options(lr = 0.01)
# history = nnet.fit(train_x_norm, train_y, epochs=125000) # 未使用 adam 优化器

# 方案二：学习率为 0.01，使用 adam 优化器
nnet.options(lr = 0.01, optimizer=opt.adam())
history = nnet.fit(train_x_norm, train_y, epochs=5000) # 使用 adam 后，加快了成本下降

# 显示最终权重和偏移量
W1 = nnet.layerList[0].weight
b1 = nnet.layerList[0].bias
W2 = nnet.layerList[1].weight
b2 = nnet.layerList[1].bias
print(f"成本记录：\n{history}")
print(f"第一层权重：\n{W1}")
print(f"第一层偏移量：\n{b1}")
print(f"第二层权重：\n{W2}")
print(f"第二层偏移量：\n{b2}")

# 显示结果
print("训练数据原始图像：")
plt.scatter(train_x[:, 0], train_x[:, 1], c=train_y[:, 0])
plt.show()

prob_y = nnet.prediction(train_x_norm)
pre_y = np.where(prob_y >= 0.5, 1, 0)
print("训练数据预测图像：")
plt.scatter(train_x[:, 0], train_x[:, 1], c=pre_y[:, 0])
plt.show()
acc = np.sum(pre_y == train_y) / pre_y.shape[0]
print(f"训练数据准确率：{acc}")

print("测试数据原始图像：")
plt.scatter(test_x[:, 0], test_x[:, 1], c=test_y[:, 0])
plt.show()

prob_y = nnet.prediction(test_x_norm)
pre_y = np.where(prob_y >= 0.5, 1, 0)
print("测试数据预测图像：")
plt.scatter(test_x[:, 0], test_x[:, 1], c=pre_y[:, 0])
plt.show()
acc = np.sum(pre_y == test_y) / pre_y.shape[0]
print(f"测试数据准确率：{acc}")

```

准确度如下：

方案一：训练数据准确率：0.98，测试数据准确率：0.96

方案二：训练数据准确率：0.995，测试数据准确率：0.98
