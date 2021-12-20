import tensorflow as tf
from tensorflow.python.training import moving_averages


def variable_weight(name, shape, initializer, trainable=True):
    return tf.get_variable(name, shape=shape, dtype=tf.float32,
                           initializer=initializer, trainable=trainable)


def batch_norm(x, decay=0.999, epsilon=1e-03, scope="scope"):
    x_shape = x.get_shape()
    input_channels = x_shape[-1]
    reduce_dims = list(range(len(x_shape) - 1))

    with tf.variable_scope(scope):
        beta = variable_weight("beta", [input_channels, ],
                               initializer=tf.zeros_initializer())
        gamma = variable_weight("gamma", [input_channels, ],
                                initializer=tf.ones_initializer())
        # for inference
        moving_mean = variable_weight("moving_mean", [input_channels, ],
                                      initializer=tf.zeros_initializer(), trainable=False)
        moving_variance = variable_weight("moving_variance", [input_channels],
                                          initializer=tf.ones_initializer(), trainable=False)

    mean, variance = tf.nn.moments(x, axes=reduce_dims)
    update_move_mean = moving_averages.assign_moving_average(moving_mean, mean, decay=decay)
    update_move_variance = moving_averages.assign_moving_average(moving_variance, variance, decay=decay)
    tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_move_mean)
    tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_move_variance)

    return tf.nn.batch_normalization(x, mean, variance, beta, gamma, epsilon)


def AdaptiveFunc(z, name):
    with tf.variable_scope(name) as scope:
        a = tf.Variable(tf.constant(1.0), name='a')
        b = tf.Variable(tf.constant(1.0), name='b')
        c = tf.Variable(tf.constant(0.0), name='c')
        d = tf.Variable(tf.constant(0.0), name='d')

       # z = batch_norm(z, scope="bn" + name)
    ff = tf.nn.leaky_relu(z, alpha=0.2, name=None)
   # ff = z * tf.sigmoid(b * z)
    #ff = b * tf.tanh(a * z + c) + d  ###AT
    #ff = tf.maximum(a*z+c,b*z+d) ###AR
    #ff = tf.nn.relu(z)
    #ff = b * tf.sigmoid(a * z + c) + d  ###AS
    #ff = tf.sigmoid(z)
    #ff = tf.tanh(z)
    return ff

def AdaptiveFunc_A(z, name):
    with tf.variable_scope(name) as scope:
        a = tf.Variable(tf.constant(1.0), name='a')
        b = tf.Variable(tf.constant(1.0), name='b')
        c = tf.Variable(tf.constant(0.0), name='c')
        d = tf.Variable(tf.constant(0.0), name='d')

        z = batch_norm(z, scope="bn" + name)

    #ff = tf.nn.leaky_relu(z, alpha=0.2, name=None)

    alpha = tf.Variable(tf.constant(0.0), name='alpha')
    pos = tf.nn.relu(z)
    neg = alpha * (z - abs(z)) * 0.5
    #ff = z * tf.sigmoid(b * z)
    #ff = b * tf.tanh(a * z + c) + d  ###AT
    #ff = tf.maximum(a*z+c,b*z+d) ###AR
    #ff = tf.nn.relu(z)
    #ff = b * tf.sigmoid(a * z + c) + d  ###AS
    #ff = tf.sigmoid(z)
    #ff = tf.tanh(z)
    return pos + neg
    #return ff

def conv_layer_A(x, num_filters, name, filter_height=3, filter_width=3, stride=1, padding='SAME'):
    input_channels = int(x.get_shape()[-1])

    with tf.variable_scope(name) as scope:
        W = tf.get_variable('weights', shape=[filter_height, filter_width, input_channels, num_filters],
                            initializer=tf.random_normal_initializer(mean=0, stddev=0.01))
        b = tf.get_variable('biases', shape=[num_filters],
                            initializer=tf.constant_initializer(0.0))

    conv = tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding=padding)
    z = tf.nn.bias_add(conv, b)

    return AdaptiveFunc_A(z, name)

def conv_layer(x, num_filters, name, filter_height=3, filter_width=3, stride=1, padding='SAME'):
    input_channels = int(x.get_shape()[-1])

    with tf.variable_scope(name) as scope:
        W = tf.get_variable('weights', shape=[filter_height, filter_width, input_channels, num_filters],
                            initializer=tf.random_normal_initializer(mean=0, stddev=0.01))
        b = tf.get_variable('biases', shape=[num_filters],
                            initializer=tf.constant_initializer(0.0))

    conv = tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding=padding)
    z = tf.nn.bias_add(conv, b)

    return AdaptiveFunc(z, name)


def fc_layer(x, input_size, output_size, name, activation='relu'):
    with tf.variable_scope(name) as scope:
        W = tf.get_variable('weights', shape=[input_size, output_size],
                            initializer=tf.random_normal_initializer(mean=0, stddev=0.01))
        b = tf.get_variable('biases', shape=[output_size],
                            initializer=tf.constant_initializer(1.0))

    z = tf.nn.bias_add(tf.matmul(x, W), b, name=scope.name)

    if activation == 'relu':
        # Apply ReLu non linearity.
        return AdaptiveFunc(z, name)
    elif activation == 'softmax':
        return tf.nn.softmax(z, name=scope.name)
    else:
        return z


def max_pool(x, name, filter_height=2, filter_width=2, stride=2, padding='SAME'):
    return tf.nn.max_pool(x, ksize=[1, filter_height, filter_width, 1],
                          strides=[1, stride, stride, 1], padding=padding,
                          name=name)


def dropout(x, keep_prob):
    return tf.nn.dropout(x, keep_prob=keep_prob)


class VGG16(object):
    def __init__(self, x, keep_prob, num_classes):
        self.X = x
        self.KEEP_PROB = keep_prob
        self.NUM_CLASSES = num_classes
        self._build_model()

    def _build_model(self):
        # Block 1
        block1_conv1 = conv_layer(self.X, 64, name='block1_conv1')
        block1_conv2 = conv_layer_A(block1_conv1, 64, name='block1_conv2')
        # block1_pool = max_pool(block1_conv2, name='block1_pool')

        # Block 2
        block2_conv1 = conv_layer(block1_conv2, 128, name='block2_conv1')
        block2_conv2 = conv_layer_A(block2_conv1, 128, name='block2_conv2')
        # block2_pool = max_pool(block2_conv2, name = 'block2_pool')

        # Block 3
        block3_conv1 = conv_layer(block2_conv2, 256, name='block3_conv1')
        block3_conv2 = conv_layer(block3_conv1, 256, name='block3_conv2')
        block3_conv3 = conv_layer_A(block3_conv2, 256, name='block3_conv3')
        block3_pool = max_pool(block3_conv3, name='block3_pool')

        # Block 4
        block4_conv1 = conv_layer(block3_pool, 512, name='block4_conv1')
        block4_conv2 = conv_layer(block4_conv1, 512, name='block4_conv2')
        block4_conv3 = conv_layer_A(block4_conv2, 512, name='block4_conv3')
        block4_pool = max_pool(block4_conv3, name='block4_pool')

        # Block 5
        block5_conv1 = conv_layer(block4_pool, 512, name='block5_conv1')
        block5_conv2 = conv_layer(block5_conv1, 512, name='block5_conv2')
        block5_conv3 = conv_layer_A(block5_conv2, 512, name='block5_conv3')
        block5_pool = max_pool(block5_conv3, name='block5_pool')

        # Full connection layers
        feature_H = 4
        feature_W = 4
        # In the original paper implementaion this will be:
        # flattened = tf.reshape(block5_pool, [-1, 7*7*512])
        # fc1 = fc_layer(flattened, 7*7*512, 7*7*512, name = 'fc1')
        flattened = tf.reshape(block5_pool, [-1, feature_H * feature_W * 512])
        fc1 = fc_layer(flattened, feature_H * feature_W * 512, feature_H * feature_W * 512, name='fc1',
                       activation='relu')
        dropout1 = dropout(fc1, self.KEEP_PROB)

        # In the original paper implementaion this will be:
        # fc2 = fc_layer(dropout1, 7*7*512, 7*7*512, name = 'fc1')
        fc2 = fc_layer(dropout1, feature_H * feature_W * 512, feature_H * feature_W * 512, name='fc2',
                       activation='relu')
        dropout2 = dropout(fc2, self.KEEP_PROB)

        # In the original paper implementaion this will be:
        # self.fc3 = fc_layer(dropout2, 7*7*512, self.NUM_CLASSES, name = 'fc3', relu = False)
        # fc3 = fc_layer(dropout2, 1*1*512, self.NUM_CLASSES, name = 'fc3', activation = 'softmax')
        fc3 = fc_layer(dropout2, feature_H * feature_W * 512, self.NUM_CLASSES, name='fc3', activation='none')
        self.output = fc3
