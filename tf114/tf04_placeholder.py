import tensorflow as tf
print(tf.__version__)
print(tf.executing_eagerly())   #True

tf.compat.v1.disable_eager_execution()
print(tf.executing_eagerly())   #False

# node1 = tf.constant(3.0, tf.float32)
# node2 = tf.constant(4.0)
# node3 = tf.add(node1, node2)

sess = tf.compat.v1.Session()

a = tf.compat.v1.placeholder(tf.float32)    # a와 b라는 input 변수 생성, 빈공간을 만들어준다.
b = tf.compat.v1.placeholder(tf.float32)
add_node = a + b

print(sess.run(add_node, feed_dict={a:3, b:4}))
print(sess.run(add_node, feed_dict={a:30, b:4.5}))

add_and_triple = add_node * 3
print(add_and_triple)   # Tensor("mul:0", dtype=float32)
print(sess.run(add_and_triple, feed_dict={a:3, b:4}))   # 21.0
