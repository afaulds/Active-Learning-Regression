import numpy as np
import tensorflow as tf


class SGDLinear:

    def __init__(self):
        self.learning_rate = 0.05
        self.num_epochs = 1
        self.sess = None

    def __create(self, num_features):
        if self.sess is None:
            self.x = tf.placeholder(dtype="float", shape=[None, num_features])
            self.y = tf.placeholder(dtype="float", shape=[None, 1])
            self.num_training = tf.placeholder(dtype="float")

            # Set model weights
            self.W = tf.Variable(tf.zeros([num_features, 1]), name="Weight")
            self.b = tf.Variable(tf.zeros([1, 1]), name="bias")

            # Construct a linear model
            self.predictor = tf.add(tf.matmul(self.x, self.W), self.b)

            # Mean squared error
            self.cost = tf.reduce_sum(tf.pow(self.predictor-self.y, 2))/(2*self.num_training)

            # Gradient descent
            #  Note, minimize() knows to modify W and b because Variable objects are trainable=True by default
            self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.cost)

            init = tf.global_variables_initializer()

            # Start training
            self.sess = tf.Session()
            self.sess.run(init)

    def fit(self, train_x, train_y):
        # Add ones to training set.
        num_training = train_x.shape[0]
        num_features = train_x.shape[1]
        self.__create(num_features)

        for epoch in range(self.num_epochs):
            #for (a, b) in zip(train_x, train_y):
            _, c = self.sess.run([self.optimizer, self.cost], feed_dict={self.x: train_x, self.y: train_y, self.num_training: num_training})
            #print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c))
        #print("W=", self.sess.run(self.W), "b=", self.sess.run(self.b))

    def predict(self, X):
        return self.sess.run(self.predictor, feed_dict={self.x: X})
