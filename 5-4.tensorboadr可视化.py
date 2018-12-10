# Tensorboard 可视化

import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.tensorboard.plugins import projector

# 清空缓存图
tf.reset_default_graph()

minst = input_data.read_data_sets('MINST_data', one_hot=True)

# 运行次数
max_steps = 1001
# 图片数量
image_num = 3000

sess = tf.Session()

# 载入图片
embedding = tf.Variable(tf.stack(minst.test.images[:image_num]), trainable=False, name='embedding')

# 批次大小
batch_size = 100
# 批次数量
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

# 命名空间
with tf.name_scope('input'):
    # 定义占位符
    x = tf.placeholder(tf.float32, [None, 784], name='x-input')
    y = tf.placeholder(tf.float32, [None, 10], name='y-input')

with tf.name_scope('input_reshape'):
    # 显示图片
    image_shaped_input = tf.reshape(x, [-1, 28, 28, 1])
    tf.summary.image('input', image_shaped_input, 10)

with tf.name_scope('layer'):
    # 神经网路
    with tf.name_scope('weight'):
        W = tf.Variable(tf.zeros([784, 10]), name='w')
        variable_summaries(W)
    with tf.name_scope('biases'):
        b = tf.Variable(tf.zeros([10]), name='b')
        variable_summaries(b)
    with tf.name_scope('expression'):
        e = tf.matmul(x, W) + b
    with tf.name_scope('softmax'):
        prediction = tf.nn.softmax(e)

# 代价函数与梯度下降
with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.square(y - prediction, name='loss'))
    tf.summary.scalar('loss', loss)
with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)

sess.run(tf.global_variables_initializer())

# 标签分类概率比较（分类正确性判断）
with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
        # 结果存放在布尔列表中
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))  # 找到最大的概率在哪个位置
    with tf.name_scope('accuracy'):
        # 准确率
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', accuracy)

# 产生metadata文件
if tf.gfile.Exists('projector/porjector/metadata.tsv'):
    tf.gfile.DeleteRecursively('projector/porjector/metadata.tsv')
with open('projector/porjector/metadata.tsv', 'w') as file:
    labels = sess.run(tf.arg_max(minst.test.labels[:], 1))
    for i in range(image_num):
        file.write(str(labels[i]) + '\n')

# 合并定义的summary
merged = tf.summary.merge_all()

projector_writer = tf.summary.FileWriter('projector/porjector', sess.graph)
saver = tf.train.Saver()
config = projector.ProjectorConfig()
embed = config.embeddings.add()
embed.tensor_name = embedding.name
embed.metadata_path = 'projector/porjector/metadata.tsv'
embed.sprite.image_path = 'projector/data/minst_10k_sprite.png'
embed.sprite.single_image_dim.extend([28,28])
projector.visualize_embeddings(projector_writer, config)

for i in range(max_steps):
    batch_xs, batch_ys = minst.train.next_batch(batch_size)
    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()
    summary, _ = sess.run([merged, train_step], feed_dict={x: batch_xs, y: batch_ys}, options=run_options, run_metadata=run_metadata)
    projector_writer.add_run_metadata(run_metadata, 'step%03d' % i)
    projector_writer.add_summary(summary, i)

    if i%100 == 0:
        acc = sess.run(accuracy, feed_dict={x: minst.test.images, y: minst.test.labels})
        print("Iter " + str(i) + ",Testing Accuracy " + str(acc))

saver.save(sess, 'projector/porjector/a_model.ckpt', global_step=max_steps)
projector_writer.close()
sess.close()