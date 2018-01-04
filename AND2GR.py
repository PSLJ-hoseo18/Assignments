import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

with tf.device('/cpu:0'):
    T, F = 1., -1.
    bias = 1.

    train_in = [[T, T, bias], [T, F, bias], [F, T, bias], [F, F, bias]]
    train_out = [[T], [F], [F], [F]]

    w = tf.Variable(tf.random_normal([3, 1]))

    output = tf.tanh(tf.matmul(train_in, w))
    error = tf.subtract(train_out, output)
    mse = tf.reduce_mean(tf.square(error))

    opt = tf.train.GradientDescentOptimizer(learning_rate=0.5)
    train = opt.minimize(mse)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    err, target = 1, 0
    epoch, max_epochs = 0, 1000

    def test():
        print('\nweight/bias\n', sess.run(w))
        print('output\n', sess.run(output))
        print('mse:', sess.run(mse), '\n')

    test()
    while err > target and epoch < max_epochs:
        epoch += 1
        err, _ = sess.run([mse, train])
        print('epoch:', epoch, 'mse', err)
    test()