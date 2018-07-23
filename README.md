# TensorFlow 学习笔记

## 安装 TensorFlow

**1. 安装 python3**
    下载安装包并安装。

**2. 安装 TensorFlow**(pip 或 pip3 视情况而定)

    `pip3 install tensorflow`

**3. 安装 Numpy**

    `pip3 install numpy`

**4. 安装 Matplotlib**

    `pip3 install matplotlib`


**测试代码：test.py**

```
import tensorflow as tf
import numpy as np

# http://www.tensorfly.cn/tfdoc/get_started/introduction.html

# 使用 NumPy 生成假数据(phony data), 总共 100 个点.
x_data = np.float32(np.random.rand(2, 100)) # 随机输入
y_data = np.dot([0.100, 0.200], x_data) + 0.300

# 构造一个线性模型
# 
b = tf.Variable(tf.zeros([1]))
W = tf.Variable(tf.random_uniform([1, 2], -1.0, 1.0))
y = tf.matmul(W, x_data) + b

# 最小化方差
loss = tf.reduce_mean(tf.square(y - y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

# 初始化变量
init = tf.global_variables_initializer()

# 启动图 (graph)
sess = tf.Session()
sess.run(init)


# 拟合平面
for step in range(201):
    sess.run(train)
    if step % 20 == 0:
        print (step, sess.run(W), sess.run(b))

# 得到最佳拟合结果 W: [[0.100  0.200]], b: [0.300]

```

执行：

`python3 test.py`

输出结果：

```
192:TensorFlowLearn my_mac$ python3 test.py
2018-07-21 06:29:05.943835: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
0 [[ 0.29652631  0.04795191]] [ 0.63011295]
20 [[ 0.09099636  0.13966224]] [ 0.33555996]
40 [[ 0.09177598  0.18426827]] [ 0.31275466]
60 [[ 0.09623011  0.19533727]] [ 0.30457038]
80 [[ 0.09851403  0.19849078]] [ 0.3016369]
100 [[ 0.09944548  0.19948611]] [ 0.30058613]
120 [[ 0.09979778  0.1998204 ]] [ 0.30020985]
140 [[ 0.09992698  0.1999364 ]] [ 0.30007514]
160 [[ 0.09997375  0.19997735]] [ 0.30002689]
180 [[ 0.09999061  0.19999191]] [ 0.30000964]
200 [[ 0.09999663  0.1999971 ]] [ 0.30000344]
```