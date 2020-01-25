import os
import pandas as pd
import tensorflow as tf
import logging
from utils import BASE_PATH
import time


class MonkeyObserver:

    def __init__(self):
        pass

    def create_tf_scoringset_pd(self, pair_history_data):
        pass

    def create_tf_dataset_pd(self, pair_history_data, test_size=1):
        pass

    def create_basis(self):
        pass


class MonkeyEngine:

    def __init__(self, name='Bob', version='_1', n_input=500, n_classes=10, batch_size=64, learn_rate=0.00001):

        self.name = name
        self.version = version
        self.sess = tf.InteractiveSession()
        self.batch_size = batch_size
        self.learn_rate = learn_rate
        self.save_path = BASE_PATH + '/trained_models/' + name + '/'

        # data:
        self.features = tf.placeholder(tf.float32, [None, n_input])
        self.labels = tf.placeholder(tf.float32, [None, n_classes])
        self.test_features = tf.placeholder(tf.float32, [None, n_input])
        self.test_labels = tf.placeholder(tf.float32, [None, n_classes])

        # Layers:
        self.input_layer = tf.reshape(self.features, [-1, 10, 50, 1])
        regularizer = tf.contrib.layers.l2_regularizer(scale=0.00000001)
        self.conv = tf.layers.conv2d(
            inputs=self.input_layer,
            filters=10,
            kernel_size=[10, 4],
            # padding="same",
            activation=tf.nn.relu,
            kernel_regularizer=regularizer)

        self.flat = tf.contrib.layers.flatten(self.conv)

        self.dense = tf.layers.dense(inputs=self.flat, units=500,
                                     activation=tf.nn.relu)  # relu -> softmax as in arxiv article
        self.dropout = tf.layers.dropout(
            inputs=self.dense, rate=0.7, seed=0.1, training=True,
            name='dropout')  # , training = tf.estimator.ModeKeys.TRAIN) # =mode=
        self.logits = tf.layers.dense(inputs=self.dropout, units=10, activation=tf.nn.softmax, name='logits')
        self.learning_rate = tf.placeholder(tf.float32)
        self.loss = tf.reduce_mean(tf.reduce_max(self.labels, axis=1)) - tf.reduce_mean(
            tf.scalar_mul(1, tf.matmul(self.logits, self.labels, transpose_b=True)),
            name='loss')
        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.00001).minimize(self.loss)
        self.saver = tf.train.Saver()

        self.init = tf.global_variables_initializer()

    def extract(self):
        pass

    def save_model(self, sess, path, breed=False):
        save_path = self.saver.save(sess, path)

        if not os.path.exists(path):
            os.makedirs(path)

        logging.debug("Model saved in: %s" % save_path)
        return save_path

    def restore_model(self, sess, path):
        self.saver.restore(sess, path)
        logging.debug("Model restored  from: %s" % path)

    def print_epoch_stats(self, epoch_i, sess, last_features, last_labels, test_features, test_labels):

        current_cost = sess.run(
            self.loss,
            feed_dict={self.features: last_features, self.labels: last_labels})

        test_cost = sess.run(
            self.loss,
            feed_dict={self.features: test_features, self.labels: test_labels})

        EPOCH_END_TIME = time.time()
        print('Epoch: {:<4} - Cost: {:<8.9} loss_test: {:<8.9} epoch end time: {:<8.9}'.format(
            epoch_i,
            current_cost,
            test_cost,
            EPOCH_END_TIME))  # ,valid_accuracy
        return current_cost, test_cost

    def batches(self, batch_size, features, labels):
        """
            Create batches of features and labels
            :param batch_size: The batch size
            :param features: List of features
            :param labels: List of labels
            :return: Batches of (Features, Labels)
            """
        assert len(features) == len(labels)
        outout_batches = []

        sample_size = len(features)
        for start_i in range(0, sample_size, batch_size):
            end_i = start_i + batch_size
            batch = [features[start_i:end_i], labels[start_i:end_i]]
            outout_batches.append(batch)

        return outout_batches

    def train(self, train_dataset=None, predict_features=None, epochs=900000, save=True, restore=False):
        """
        Provide train_dataset to train and save model or predict_features to restore and use model
        :param train_dataset: expected 4-element tuple formed by CNN7_test.create_tf_dataset_pd
        :param predict_features: expected 50-element dataframe of features to portfolio_weights
        :param epochs: int train epochs
        :param save: save model after training
        :param restore: restore and continue training or start from scratch
        :return:
        """

        current_cost = 0.0
        test_cost = 0.0
        predictions = None
        validation_train = None
        validation_test = None

        if train_dataset is None and predict_features is None:
            raise Exception("Must provide one of: train_dataset or predict_features")

        save_path = BASE_PATH + '/trained_models/' + str(self.name) + str(self.version) + '/'
        with tf.Session() as sess:

            sess.run(self.init)

            # Restore and continue training or start new
            if restore:
                self.restore_model(sess, save_path)

            if train_dataset is not None:

                train_features, train_labels, test_features, test_labels = train_dataset

                train_batches = self.batches(self.batch_size, train_features, train_labels)
                # Training cycle
                for epoch_i in range(epochs):
                    # Loop over all batches
                    for batch_features, batch_labels in train_batches:
                        train_feed_dict = {
                            self.features: batch_features,
                            self.labels: batch_labels,
                            self.learning_rate: 0.00001}
                        sess.run(self.optimizer, feed_dict=train_feed_dict)

                    # Print cost and validation accuracy of an epoch
                    if epoch_i % 1000 == 0:
                        current_cost, test_cost = self.print_epoch_stats(epoch_i, sess, batch_features, batch_labels,
                                                                         test_features, test_labels)

                validation_train = sess.run(
                    self.logits,
                    feed_dict={self.features: train_features})

                validation_test = sess.run(
                    self.logits,
                    feed_dict={self.features: test_features})

                if save:
                    self.save_model(sess, save_path)

            elif predict_features is not None:
                # Always try to restore sess if predict_features where provided
                self.restore_model(sess, save_path)
                predictions = sess.run(
                    self.logits,
                    feed_dict={self.features: predict_features})

        return validation_train, validation_test, predictions, current_cost, test_cost

    def predict(self, scoring_set):

        with tf.Session() as sess:
            sess.run(self.init)

            save_path = BASE_PATH + '/trained_models/' + str(self.name) + str(self.version) + '/'

            self.restore_model(sess, save_path)
            predictions = sess.run(
                self.logits,
                feed_dict={self.features: scoring_set})

        return predictions
