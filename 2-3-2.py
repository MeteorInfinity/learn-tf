import tensorflow as tf

#Feed
#创建占位符
input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)

output = tf.multiply(input1, input2)

with tf.Session() as sess:
    #feed的数据以字典的形式传入
    print(sess.run(output, feed_dict={input1:[8.], input2:[2.]}))
