# Dropout
# 防止过拟合

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

minst = input_data.read_data_sets('MINST_data', one_hot=True)

# 批次（大小与数量）
batch_size = 100
batch_num = minst.train.num_examples

# 定义占位符
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)

# 神经网路
W1 = tf.Variable(tf.truncated_normal([784, 2000], stddev=0.1))
b1 = tf.Variable(tf.zeros([2000])+0.1)
L1 = tf.nn.tanh(tf.matmul(x, W1)+b1)
L1_drop = tf.nn.dropout(L1, keep_prob)

W2 = tf.Variable(tf.truncated_normal([2000, 2000], stddev=0.1))
b2 = tf.Variable(tf.zeros([2000])+0.1)
L2 = tf.nn.tanh(tf.matmul(L1_drop, W2)+b2)
L2_drop = tf.nn.dropout(L2, keep_prob)

W3 = tf.Variable(tf.truncated_normal([2000, 1000], stddev=0.1))
b3 = tf.Variable(tf.zeros([1000])+0.1)
L3 = tf.nn.tanh(tf.matmul(L2_drop, W3)+b3)
L3_drop = tf.nn.dropout(L3, keep_prob)

W4 = tf.Variable(tf.truncated_normal([1000, 10], stddev=0.1))
b4 = tf.Variable(tf.zeros([10])+0.1)
prediction = tf.nn.softmax(tf.matmul(L3_drop, W4)+b4)

# 代价函数与梯度下降
# loss = tf.reduce_mean(-tf.reduce_sum(y*tf.log(prediction), reduction_indices=[1]))
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))
train_step = tf.train.GradientDescentOptimizer(0.7).minimize(loss)

init = tf.global_variables_initializer()

# 标签分类概率比较（分类正确性判断） -- 结果存放在布尔列表中
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))  # 找到最大的概率在哪个位置
# 准确率
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(30):
        for batch in range(batch_num):
            batch_xs, batch_ys = minst.train.next_batch(batch_size)
            sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.0})

        test_acc = sess.run(accuracy, feed_dict={x: minst.test.images, y: minst.test.labels, keep_prob: 1.0})
        train_acc = sess.run(accuracy, feed_dict={x: minst.train.images, y: minst.train.labels, keep_prob: 1.0})
        print("Iter " + str(epoch) + ",Testing Accuracy " + str(test_acc) + ",Training Accuracy " + str(train_acc))
