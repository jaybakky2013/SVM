#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow as tf
from functools import reduce
from numpy import unique, array, vectorize
from sklearn.metrics import accuracy_score, f1_score


# In[ ]:


class SVMClassifier:

    def __init__(self, train_data=None):
        self.train_step = None
        self.sess = None
        self.accuracy = None
        self.loss = None
        self.X = None
        self.y = None
        self.prediction = None
        
        data, labels = train_data
        labels = self._transform_labels(labels)
        data = self._flatten_input(data)
        self.train_data = (data, labels)
        self._open_session()

        self.assemble_graph()

        if train_data:
            self.train()

    def assemble_graph(self, learning_rate=0.02):
        self.X = tf.placeholder(name="input", dtype=tf.float32, shape=(None, 784))
        self.y = tf.placeholder(name="label", dtype=tf.float32, shape=(None, 1))

        self.w = tf.Variable(tf.random_normal(shape=[784, 1]))
        self.b = tf.Variable(tf.random_normal(shape=[1, 1]))

        model_output = tf.subtract(tf.matmul(self.X, self.w), self.b)

        self.loss = tf.reduce_mean(tf.maximum(0., 1 - self.y * model_output))+0.001*tf.norm(self.w)

        self.prediction = tf.sign(model_output)
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.prediction, self.y),
                                               tf.float32))

        opt = tf.train.GradientDescentOptimizer(learning_rate)
        self.train_step = opt.minimize(self.loss)
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def train(self, epochs=20, minibatch_size=256):

        data = self._create_minibatches(minibatch_size)
        for i in range(epochs * len(data)):
            datax, labely = data[i % len(data)]
            self.sess.run(self.train_step, feed_dict={self.X: datax, self.y: labely})
            train_loss = self.sess.run(self.loss, feed_dict={self.X: datax, self.y: labely})
            accuracy = self.sess.run(self.accuracy, feed_dict={self.X: datax, self.y: labely})
            if i % len(data) == 0:
                print("Epoch %d, loss: %.2f accuracy %.2f." % (
                    i // (len(data)), train_loss, accuracy))

    def predict(self, data):
        data = self._flatten_input(data)
        pred = self.sess.run(self.prediction, feed_dict={self.X: data}).flatten()
        pred[pred == -1] = 0
        return pred

    def _create_minibatches(self, minibatch_size):
        pos = 0

        data, labels = self.train_data
        n_samples = len(labels)

        batches = []
        while pos + minibatch_size < n_samples:
            batches.append((data[pos:pos + minibatch_size, :], labels[pos:pos + minibatch_size]))
            pos += minibatch_size

        if pos < n_samples:
            batches.append((data[pos:n_samples, :], labels[pos:n_samples, :]))

        return batches

    def _transform_labels(self, labels):
        labels[labels == 0] = -1
        labels=labels.reshape((-1, 1))
        return labels
      
    def _flatten_input(self, data):
        return data.reshape((-1,784))

    def _open_session(self):
        self.sess = tf.Session()


# In[ ]:


if __name__ == "__main__":



    def mnist_to_binary(train_data, train_label, test_data, test_label):

        binarized_labels = []
        for labels in [train_label, test_label]:
            remainder_2 = vectorize(lambda x: x%2)
            binarized_labels.append(remainder_2(labels))

        train_label, test_label = binarized_labels

        return train_data, train_label, test_data, test_label


# In[44]:


((train_data, train_labels),
    (eval_data, eval_labels)) = tf.keras.datasets.mnist.load_data()

train_data, train_labels, test_data, test_labels = mnist_to_binary(train_data, train_labels, eval_data, eval_labels)

svm = SVMClassifier((train_data, train_labels))
print("Testing score f1: {}".format(f1_score(test_labels, svm.predict(test_data))))

