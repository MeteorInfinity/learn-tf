import tensorflow as tf

#声明并初始化变量为0
state = tf.Variable(0, name = 'counter')
#运算op
new_value = tf.add(state, 1)
#赋值op
update = tf.assign(state, new_value)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    print(sess.run(state))
    for _ in range(5):
        sess.run(update)
        print(sess.run(state))
