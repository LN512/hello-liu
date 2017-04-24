import tensorflow as tf
state = tf.Variable(0,name="counter")
one = tf.constant(1)
new_state = tf.add(state,one)
update = tf.assign(state,new_state)   #交换state与new_state的值


init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for _ in range(10):
        sess.run(update)
        print(sess.run(state))


