# Tensorboard 网络结构

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 清空缓存图
tf.reset_default_graph()

minst = input_data.read_data_sets('MINST_data', one_hot=True)

# 批次（大小与数量）
batch_size = 100
batch_num = minst.train.num_examples

# 命名空间
with tf.name_scope('input'):
    # 定义占位符
    x = tf.placeholder(tf.float32, [None, 784], name='x_input')
    y = tf.placeholder(tf.float32, [None, 10], name='y_input')

with tf.name_scope('layer'):
    # 神经网路
    with tf.name_scope('weight'):
        W = tf.Variable(tf.zeros([784, 10]), name='w')
    with tf.name_scope('biases'):
        b = tf.Variable(tf.zeros([10]), name='b')
    with tf.name_scope('expression'):
        e = tf.matmul(x, W)+b
    with tf.name_scope('softmax'):
        prediction = tf.nn.softmax(e)

# 代价函数与梯度下降
with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.square(y-prediction, name='loss'))
with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)

init = tf.global_variables_initializer()

# 标签分类概率比较（分类正确性判断） --
with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
        # 结果存放在布尔列表中
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))  # 找到最大的概率在哪个位置
    with tf.name_scope('accuracy'):
        # 准确率
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(init)
    writer = tf.summary.FileWriter('logs/', sess.graph)
    for epoch in range(2):
        for batch in range(batch_num):
            batch_xs, batch_ys = minst.train.next_batch(batch_size)
            sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys})

        acc = sess.run(accuracy, feed_dict={x: minst.test.images, y: minst.test.labels})
        print("Iter " + str(epoch) + ",Testing Accuracy " + str(acc))
