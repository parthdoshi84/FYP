import tensorflow as tf
import numpy as np

class CNN:
    def __init__(self,input_tensor,is_training=True):
        self.input_tensor = input_tensor;
        self.is_training = is_training
        self.forward

    def convolutional_layer(self,input_tensor,filters,kernel_size,name):
        with tf.name_scope(name):
            conv = tf.layers.conv2d(
                inputs= input_tensor,
                filters=filters,
                kernel_size=[kernel_size,kernel_size],
                padding="same",
                activation=tf.nn.relu)
            return conv

    def max_pooling_layer(self,input_tensor,pool_size,strides,name):
        with tf.name_scope(name):
            max1 = tf.layers.max_pooling2d(inputs=input_tensor, pool_size= [pool_size,pool_size], strides = strides);
            return max1

    def forward(self):

        #conv1
        conv1 = self.convolutional_layer(self.input_tensor,128,5,"conv1")

        #pool1
        pool1 = self.max_pooling_layer(conv1,2,2,"pool1")

        # conv2
        conv2 = self.convolutional_layer(pool1, 256, 5, "conv2")

        # pool2
        pool2 = self.max_pooling_layer(conv2, 2, 2, "pool2")

        # conv3
        conv3 = self.convolutional_layer(pool2, 512, 4, "conv3")

        # pool3
        pool3 = self.max_pooling_layer(conv3, 2, 2, "pool3")

        pool3_flat = tf.reshape(pool3, [-1,6*6*512], name = "flat")

        #Dropout
        dropout = tf.layers.dropout(inputs=pool3_flat, rate=0.3,name = "dropout")

        # Dense Layer 1
        flat1 = tf.layers.dense(inputs=dropout, units=2048, activation=tf.nn.relu, name="full1")

        # Dense Layer
        flat2 = tf.layers.dense(inputs=flat1, units=3072, activation=tf.nn.relu, name="full2")

        # Logits layer
        logits = tf.layers.dense(inputs=flat1, units=7, activation=tf.nn.relu, name="last_layer")

        return logits

def main():

    #load training data
    train_data = np.load("../train_data.npy")  # Returns np.array
    train_data = np.float32(train_data)
    train_labels = np.load("../train_labels.npy")

    #load eval data
    eval_data = np.load("../test_data.npy")
    eval_data = np.float32(eval_data)
    eval_labels = np.load("../test_labels.npy")

    batch_size = 100
    total = train_data.shape[0]

    #inputs
    input_te = tf.placeholder(tf.float32, [None,48*48], name = "input")
    input_tensor = tf.reshape(input_te, [-1,48,48,1])
    tf.summary.image('input', input_tensor, 3)

    cnn_obj = CNN(input_tensor)
    forward_op = cnn_obj.forward()

    predict = tf.nn.softmax(forward_op, name = "prediction")
    probab = tf.nn.softmax(forward_op)

    onehot_labels = tf.placeholder(tf.int32, shape = [None,7])
    loss = tf.losses.softmax_cross_entropy(onehot_labels = onehot_labels, logits = forward_op)
    xent = tf.reduce_mean(loss)
    tf.summary.scalar("xent",xent);

    optimizer = tf.train.AdagradOptimizer(learning_rate = 0.01)
    train_op = optimizer.minimize(loss= loss, global_step = tf.train.get_global_step())

    n_epochs = 200
    n_classes = 7

    init = tf.global_variables_initializer()

    summ = tf.summary.merge_all()

    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(init)
        writer = tf.summary.FileWriter("logs", graph = tf.get_default_graph())
        for epoch in range(n_epochs):
            X = train_data
            y = train_labels
            for i in range(0,total,batch_size):
                x_curr = X[i:i+batch_size]
                y_curr = y[i:i+batch_size]
                one_hot_targets = np.eye(n_classes)[y_curr]
                loss_value = sess.run(loss, feed_dict = {input_te:x_curr,onehot_labels:one_hot_targets})
                sess.run(train_op, feed_dict={input_te: x_curr, onehot_labels: one_hot_targets})

            print("epoch: " +str(epoch)+" loss: "+str(loss_value))
            saver.save(sess, "model/", global_step = epoch, write_meta_graph = True)
        saver.save(sess, "model_final/")

main()
