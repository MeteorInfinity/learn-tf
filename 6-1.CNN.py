import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

minst = input_data.read_data_sets('MINST_data', one_hot=True)

# 批次（大小与数量）
batch_size = 100
batch_num = minst.train.num_examples

# 权值
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

# 偏置值
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return  tf.Variable(initial)

# 卷积层
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

# 池化层
def max_pool_2x2(conv):
    # ksize [1,x,y,1]
    return  tf.nn.max_pool(conv, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

# 占位符
x = tf.placeholder(tf.float32, [None,784])
y = tf.placeholder(tf.float32, [None, 10])

# 格式转换( x --> 4d [batch, in_height, in_width, in_channels] )
x_image = tf.reshape(x, [-1,28,28,1])

# 卷积层 1
W_conv1 = weight_variable([5,5,1,32])  # 5*5卷积窗口,32个卷积核从1个平面抽取特征值
b_conv1 = bias_variable([32])  # 每个卷积核一个偏置值
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)  # 卷积(Wx + b),应用于relu激活函数
h_pool1 = max_pool_2x2(h_conv1)  # 最大池化(28x28->14x14, X32)

# 卷积层 2
W_conv2 = weight_variable([5,5,32,64])  # 5*5卷积窗口,64个卷积核从32个平面抽取特征值
b_conv2 = bias_variable([64])  #每个卷积核一个偏置值
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)  # 卷积(Wx + b),应用于relu激活函数
h_pool2 = max_pool_2x2(h_conv2)  # 最大池化(14x14->7x7, X64)

# 全连接层 1
W_fc1 = weight_variable([7*7*64, 1024])  # 全连接上一层(7*7*64->1024)
b_fc1 = bias_variable([1024])  # 1024个节点
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])  # 将卷积层2的池化层输出扁平化为1维
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)  # 求全连接层1的输出

# drpoout
keep_prob = tf.placeholder(tf.float32)  # 神经元的输出概率
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# 全连接层 2
W_fc2 = weight_variable([1024, 10])  # 全连接上一层(1024->10)
b_fc2 = bias_variable([10])  # 1024个节点

# 计算输出
prediction = tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2) + b_fc2)

# 代价函数
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))
# 优化器
train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

# 结果集(布尔表)
correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
# 准确率
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(21):
        for batch in range(batch_num):
            batch_xs,batch_ys = minst.train.next_batch(batch_size)
            sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 0.7})
        acc = sess.run(accuracy, feed_dict={x: minst.test.images, y: minst.test.labels, keep_prob: 1.0})
        print("Iter " + str(epoch) + ", Testing Accuracy= " + str(acc))
