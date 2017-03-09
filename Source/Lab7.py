# import csv
# import tensorflow as tf
# import numpy as np
#
# crime = []
# zone = []
# indus = []
# chas = []
# nox_train = []
# room = []
# age = []
# dist = []
# highway = []
# tax = []
# ptratio = []
# blk = []
# lstat = []
# medv_train = []
#
# with open('/home/jackie/Documents/data_train', 'r') as train_file:
#     reader = csv.reader(train_file, delimiter=' ', skipinitialspace=True)
#     for c,z,i,ch,n,r,a,d,h,t,p,b,l,m in reader:
#         crime.append(c)
#         zone.append(z)
#         indus.append(i)
#         chas.append(ch)
#         nox_train.append(n)
#         room.append(r)
#         age.append(a)
#         dist.append(d)
#         highway.append(h)
#         tax.append(t)
#         ptratio.append(p)
#         blk.append(b)
#         lstat.append(l)
#         medv_train.append(m)
#
# nox_test = []
# medv_test = []
#
# with open('/home/jackie/Documents/data_test', 'r') as test_file:
#     reader = csv.reader(test_file, delimiter=' ', skipinitialspace=True)
#     for c,z,i,ch,n,r,a,d,h,t,p,b,l,m in reader:
#         crime.append(c)
#         zone.append(z)
#         indus.append(i)
#         chas.append(ch)
#         nox_test.append(n)
#         room.append(r)
#         age.append(a)
#         dist.append(d)
#         highway.append(h)
#         tax.append(t)
#         ptratio.append(p)
#         blk.append(b)
#         lstat.append(l)
#         medv_test.append(m)
#
# rng = np.random
# trX = nox_train
# trY = medv_train
#
# X = tf.placeholder("float")
# Y = tf.placeholder("float")
#
# w = tf.Variable(rng.rand(), name="weights")
# b = tf.Variable(rng.randn(), name="bias")
#
# y_model=tf.add(tf.multiply(X,w), b)
#
# cost = tf.reduce_sum(tf.pow(y_model-Y, 2))/(2*100)
#
# train_op = tf.train.GradientDescentOptimizer(0.5).minimize(cost)
#
# sess = tf.Session()
#
# init = tf.global_variables_initializer()
#
# sess.run(init)
#
# for (x, y) in zip(trX, trY):
#     sess.run(train_op, feed_dict={X: x, Y: y})
#
# print("Optimization finished!")
# training_cost = sess.run(cost, feed_dict={X: trX, Y: trY})
#
# print("Training cost = ", training_cost, "W = ", sess.run(w), "b = ", sess.run(b), '\n')
#
# test_X = nox_test
# test_Y = medv_test
#
# # print("Testing... (Mean square loss comparison)")
# #
# # testing_cost = sess.run(tf.reduce_sum(tf.pow(y_model - Y, 2)) / (2 * test_X.shape[0]),
# #                         feed_dict={X: test_X, Y: test_Y})
# #
# # print("Testing cost = ", testing_cost)
# #print("Absolute mean square loss difference: ", abs(training_cost - testing_cost))

import matplotlib.pyplot as plot
import tensorflow as tf
import numpy as np
from numpy import genfromtxt
from sklearn.datasets import load_boston


def read_data(filePath, delimiter = ","):
    return genfromtxt(filePath, delimiter=delimiter)

def read_boston_data():
    boston = load_boston()
    features = np.array(boston.data)
    labels = np.array(boston.target)
    return features, labels

def feature_normalize(dataset):
    mu = np.mean(dataset,axis=0)
    sigma = np.std(dataset, axis=0)
    return (dataset - mu)/sigma

def append_bias_reshape(features, labels):
    n_training_samples = features.shape[0]
    n_dim = features.shape[1]
    f = np.reshape(np.c_[np.ones(n_training_samples), features],[n_training_samples,n_dim + 1])
    l = np.reshape(labels, [n_training_samples,1])
    return f, l


features, labels = read_boston_data()
normalized_features = feature_normalize(features)
f,l = append_bias_reshape(normalized_features, labels)
n_dim = f.shape[1]
rnd_indices = np.random.rand(len(f)) < 0.8

train_x = f[rnd_indices]
train_y = l[rnd_indices]
test_x = f[-rnd_indices]
test_y = l[-rnd_indices]

learning_rate = 0.01
training_epocs = 1000
cost_history = np.empty(shape=[1],dtype=float)

X = tf.placeholder(tf.float32,[None,n_dim])
Y = tf.placeholder(tf.float32,[None,1])
W = tf.Variable(tf.ones([n_dim, 1]))

init = tf.initialize_all_variables()

y = tf.matmul(X, W)
cost = tf.reduce_mean(tf.square(y - Y))
training_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

sess = tf.Session()
sess.run(init)

for epoch in range(training_epocs):
    sess.run(training_step, feed_dict={X: train_x, Y: train_y})
    cost_history = np.append(cost_history, sess.run(cost, feed_dict={X: train_x, Y: train_y}))

plot.plot(range(len(cost_history)), cost_history)
plot.axis([0, training_epocs, 0, np.max(cost_history)])
plot.show()

predict_y = sess.run(y, feed_dict={X: test_x})
mse = tf.reduce_mean(tf.square(predict_y - test_y))
print("Mean square error: %.4f" % sess.run(mse))

fig, ax = plot.subplots()
ax.scatter(test_y, predict_y)
ax.plot([test_y.min(), test_y.max()], [test_y.min(), test_y.max()], 'k--', lw=3)
ax.set_xlabel('measured')
ax.set_ylabel('predicted')
plot.show()