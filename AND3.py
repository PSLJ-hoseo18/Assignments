import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

with tf.device('/cpu:0'):
    T, F = 1., -1.
    bias = 1.

    train_in = [[T, T, T, bias], [T, T, F, bias], [T, F, T, bias], [T, F, F, bias], [F, T, T, bias], [F, T, F, bias], [F, F, T, bias], [F, F, F, bias]]
    train_out = [[T], [F], [F], [F], [F], [F], [F], [F]]

    w = tf.Variable(tf.random_normal([4, 1]))
    def step(x):
        is_greater = tf.greater(x, 0)
        as_float = tf.to_float(is_greater)
        doubled = tf.multiply(as_float, 2)
        return tf.subtract(doubled, 1)

    output = step(tf.matmul(train_in, w))
    error = tf.subtract(train_out, output)
    mse = tf.reduce_mean(tf.square(error))

    delta = tf.matmul(train_in, error, transpose_a=True)
    train = tf.assign(w, tf.add(w, delta))

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    err, target = 1, 0
    epoch, max_epochs = 0, 100

    def test():
        print('\nweight/bias\n', sess.run(w))
        print('output\n', sess.run(output))
        print('mse: ', sess.run(mse), '\n')

    test()
    while err > target and epoch < max_epochs:
        epoch += 1
        err, _ = sess.run([mse, train])
        print('epoch:', epoch, 'mse', err)
    
    test()