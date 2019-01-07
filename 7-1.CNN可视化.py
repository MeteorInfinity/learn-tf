import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

minst = input_data.read_data_sets('MINST_data', one_hot=True)

# 批次（大小与数量）
batch_size = 100
batch_num = minst.train.num_examples


# 参数概要
def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)  # 均值
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)  # 标准差
        tf.summary.scalar('max', tf.reduce_max(var))  # 最大值
        tf.summary.scalar('min', tf.reduce_min(var))  # 最小值
        tf.summary.histogram('histogram', var)  # 直方图


# 权值
def weight_variable(shape, name):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=name)


# 偏置值
def bias_variable(shape, name):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name)


# 卷积层
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')


# 池化层
def max_pool_2x2(conv):
    # ksize:[1,x,y,1]
    return  tf.nn.max_pool(conv, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')


# 输入
with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, [None,784], name='x_input')
    y = tf.placeholder(tf.float32, [None, 10], name='y_input')
    with tf.name_scope('x_image'):
        x_image = tf.reshape(x, [-1, 28, 28, 1], name='x_image')  # x->[batch, in_height, in_width, in_channels]

# 卷积层 1
with tf.name_scope('conv1'):
    with tf.name_scope('W_conv1'):
        W_conv1 = weight_variable([5,5,1,32], name='W_conv1')  # 5*5卷积窗口,32个卷积核从1个平面抽取特征值
    with tf.name_scope('b_conv1'):
        b_conv1 = bias_variable([32], name='b_conv1')  # 每个卷积核一个偏置值
    with tf.name_scope('conv2d_1'):
        conv2d_1 = conv2d(x_image, W_conv1) + b_conv1  # 卷积(Wx + b)
    with tf.name_scope('relu_1'):
        h_conv1 = tf.nn.relu(conv2d_1)  # 激活函数
    with tf.name_scope('h_pool1'):
        h_pool1 = max_pool_2x2(h_conv1)  # 最大池化(28x28->14x14, X32)

# 卷积层 2
with tf.name_scope('conv2'):
    with tf.name_scope('W_conv2'):
        W_conv2 = weight_variable([5,5,32,64], name='W_conv2')  # 5*5卷积窗口,64个卷积核从32个平面抽取特征值
    with tf.name_scope('b_conv2'):
        b_conv2 = bias_variable([64], name='b_conv2')  #每个卷积核一个偏置值
    with tf.name_scope('conv2d_2'):
        conv2d_2 = conv2d(x_image, W_conv2) + b_conv2  # 卷积(Wx + b)
    with tf.name_scope('relu_2'):
        h_conv2 = tf.nn.relu(conv2d_2)  # 激活函数
    with tf.name_scope('h_pool2'):
        h_pool2 = max_pool_2x2(h_conv2)  # 最大池化(14x14->7x7, X64)

# 全连接层 1
with tf.name_scope('fc1'):
    with tf.name_scope('W_fc1'):
        W_fc1 = weight_variable([7*7*64, 1024], name='W_fc1')  # 全连接上一层(7*7*64->1024)
    with tf.name_scope('b_fc1'):
        b_fc1 = bias_variable([1024], name='b_fc1')  # 1024个节点
    with tf.name_scope('h_pool2_flat'):
        h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64], name='h_pool2_flat')  # 将卷积层2的池化层输出扁平化为1维
    with tf.name_scope('wx_plus_b1'):
        wx_plus_b1 = tf.matmul(h_pool2_flat, W_fc1) + b_fc1  # 求全连接层1的输出
    with tf.name_scope('relu'):
        h_fc1 = tf.nn.relu(wx_plus_b1)  # 求全连接层1的输出

# Drpoout
with tf.name_scope('keep_prob'):
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')  # 神经元的输出概率
with tf.name_scope('h_fc1_drop'):
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob, name='h_fc1_drop')

# 全连接层 2
with tf.name_scope('fc2'):
    with tf.name_scope('W_fc2'):
        W_fc2 = weight_variable([1024, 10], name='W_fc2')  # 全连接上一层(1024->10)
    with tf.name_scope('b_fc2'):
        b_fc2 = bias_variable([10], name='b_fc2')  # 1024个节点
    with tf.name_scope('wx_plus_b2'):
        wx_plus_b2 = tf.matmul(h_fc1_drop, W_fc2) + b_fc2  # 求全连接层1的输出
    with tf.name_scope('softmax'):
        prediction = tf.nn.softmax(wx_plus_b2)  # 计算输出
with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction), name='cross_entropy')  # 代价函数
    tf.summary.scalar('cross_entropy', loss)
with tf.name_scope('train'):
    train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)  # 优化器

with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
        correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))  # 结果集(一维向量中最大值位置)
    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))  # 准确率

# 合并所有的summary
merged = tf.summary.merge_all()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    train_writer = tf.summary.FileWriter('logs/train', sess.graph)
    test_writer = tf.summary.FileWriter('logs/test', sess.graph)

    for i in range(1001):
        # 训练模型
        batch_xs, batch_ys = minst.train.next_batch(batch_size)
        sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 0.5})

        # 记录训练集计算的参数
        summary = sess.run(merged, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.0})
        train_writer.add_summary(summary, i)

        # 记录测试集计算的参数
        batch_xs, batch_ys = minst.test.next_batch(batch_size)
        summary = sess.run(merged, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.0})
        test_writer.add_summary(summary, i)

        if i % 100 == 0:
            test_acc = sess.run(accuracy, feed_dict={x: minst.test.images, y: minst.test.labels, keep_prob: 1.0})
            train_acc = sess.run(accuracy, feed_dict={x: minst.train.images[:10000], y: minst.train.labels[:10000], keep_prob: 1.0})
            print("Iter " + str(i) + ", Testing Accuracy= " + str(test_acc) + ", Training Accuracy= " + str(train_acc))

    for epoch in range(21):
        for batch in range(batch_num):
            batch_xs,batch_ys = minst.train.next_batch(batch_size)
            sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 0.7})
        acc = sess.run(accuracy, feed_dict={x: minst.test.images, y: minst.test.labels, keep_prob: 1.0})
        print("Iter " + str(epoch) + ", Testing Accuracy= " + str(acc))
