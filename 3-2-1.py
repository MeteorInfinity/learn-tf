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

# 神经网路
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
prediction = tf.nn.softmax(tf.matmul(x, W)+b)

# 代价函数与梯度下降
loss = tf.reduce_mean(-tf.reduce_sum(y*tf.log(prediction), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

init = tf.global_variables_initializer()

# 标签分类概率比较（分类正确性判断） -- 结果存放在布尔列表中
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))  # 找到最大的概率在哪个位置
# 准确率
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(21):
        for batch in range(batch_num):
            batch_xs, batch_ys = minst.train.next_batch(batch_size)
            sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys})

        acc = sess.run(accuracy, feed_dict={x: minst.test.images, y: minst.test.labels})
        print("Iter " + str(epoch) + ",Testing Accuracy " + str(acc))
