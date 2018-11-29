import tensorflow as tf

#常量操作
m1 = tf.constant([[3,3]])
m2 = tf.constant([[2],[3]])

#矩阵乘法操作
product = tf.matmul(m1,m2)

#定义一个会话，启动默认图
session = tf.Session()
#调用run方法执行操作
results = session.run(product)

print(results)
session.close()

#一次性会话
with tf.Session() as session:
    results = session.run(product)
    print(results)
