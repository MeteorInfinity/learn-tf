# 分类问题-MINST （简单版本）

# 图像：28×28=784(向量) 像素点(0-1)
# 数据集：[60000,784](张量)--[图片索引，像素点索引]
# 标签：0-9[one-hot vectors]--([1,0,0,...,0])

# 输入层：784(像素)
# 输出层：10(分类)

# Softmax回归模型 （给不同对象分配概率--分类）
# softmax(x)i = exp(xi) / sumj(exp(xj))

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

minst = input_data.read_data_sets('MINST_data', one_hot=True)

# 批次（大小与数量）
batch_size = 100
batch_num = minst.train.num_examples

# 定义占位符
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

# 神经网络
# 输入层
W_L1 = tf.Variable(tf.zeros([784, 400]))
b_L1 = tf.Variable(tf.zeros([400]))
exp_L1 = tf.matmul(x, W_L1)+b_L1
L1 = tf.nn.tanh(exp_L1)

# 中间层-1
W_L2 = tf.Variable(tf.zeros([400, 200]))
b_L2 = tf.Variable(tf.zeros([200]))
exp_L2 = tf.matmul(L1, W_L2)+b_L2
L2 = tf.nn.tanh(exp_L2)

# 中间层-2
W_L3 = tf.Variable(tf.zeros([200, 50]))
b_L3 = tf.Variable(tf.zeros([50]))
exp_L3 = tf.matmul(L2, W_L3)+b_L3
L3 = tf.nn.tanh(exp_L3)

# 中间层-3
W_L4 = tf.Variable(tf.zeros([50, 10]))
b_L4 = tf.Variable(tf.zeros([10]))
exp_L4 = tf.matmul(L3, W_L4)+b_L4
prediction = tf.nn.softmax(exp_L4)

# 代价函数与梯度下降
loss = tf.reduce_mean(tf.square(y-prediction, name='loss'))
train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)

init = tf.global_variables_initializer()

# 标签分类概率比较（分类正确性判断） -- 结果存放在布尔列表中
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))  # 找到最大的概率在哪个位置
# 准确率
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(200):
        for batch in range(batch_num):
            batch_xs, batch_ys = minst.train.next_batch(batch_size)
            sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys})

        acc = sess.run(accuracy, feed_dict={x: minst.test.images, y: minst.test.labels})
        print("Iter " + str(epoch) + ",Testing Accuracy " + str(acc))
