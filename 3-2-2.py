# 分类问题-MINST

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
# 隐藏层
W_L1 = tf.Variable(tf.random_normal([784, 300], stddev=0.03))
b_L1 = tf.Variable(tf.random_normal([300]))
exp_L1 = tf.add(tf.matmul(x, W_L1), b_L1)
L1 = tf.nn.relu(exp_L1)

# 输出层
W_L2 = tf.Variable(tf.random_normal([300, 10], stddev=0.03))
b_L2 = tf.Variable(tf.random_normal([10]))
exp_L2 = tf.add(tf.matmul(L1, W_L2), b_L2)
y_ = tf.nn.softmax(exp_L2)

# 代价函数与优化器
y_clipped = tf.clip_by_value(y_, 1e-10, 0.9999999)
cross_entropy = -tf.reduce_mean(tf.reduce_sum(y * tf.log(y_clipped) + (1 - y) * tf.log(1 - y_clipped), axis=1))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.5).minimize(cross_entropy)

init = tf.global_variables_initializer()

# 标签分类概率比较（分类正确性判断） -- 结果存放在布尔列表中
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))  # 找到最大的概率在哪个位置
# 准确率
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(10):
        avg_cost = 0
        for batch in range(batch_num):
            batch_xs, batch_ys = minst.train.next_batch(batch_size)
            _, c = sess.run([optimizer, cross_entropy], feed_dict={x: batch_xs, y: batch_ys})
            avg_cost += c / batch_num

        acc = sess.run(accuracy, feed_dict={x: minst.test.images, y: minst.test.labels})
        print("Epoch:", (epoch + 1), "cost = ", "{:.3f}".format(avg_cost), "accuracy = ", str(acc))
    print(sess.run(accuracy, feed_dict={x: minst.test.images, y: minst.test.labels}))
