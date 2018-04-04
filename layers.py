import tensorflow as tf

def Convolution2D_Layer(layer_idx, input, filter, size, stride, trainable=False, alpha=0.1):

    channels = input.get_shape()[3]
    weight = tf.Variable(tf.truncated_normal([size, size, int(channels), filter], stddev=0.1), trainable=trainable)
    bias = tf.Variable(tf.constant(0.1, shape=[filter]), trainable=trainable)
    #
    # pad_size = size // 2
    # pad_mat = np.array([[0, 0], [pad_size, pad_size], [pad_size, pad_size], [0, 0]])
    # padded_input = tf.pad(input, pad_mat)
    conv = tf.nn.conv2d(input, weight, strides=[1, stride, stride, 1], padding='SAME', name=str(layer_idx)+'_conv')
    conv_bias = tf.add(conv, bias)
    print('Layer {0}: Type: Convolution Stride: {1} Filter: {2}, Input Shape: {3}'.format(layer_idx, stride, str([size, size, int(channels), filter]), str(input.get_shape())))
    return tf.maximum(tf.multiply(alpha, conv_bias), conv_bias)

def MaxPool_Layer(layer_idx, input, size, stride):
    channels = input.get_shape()[3]
    print('Layer {0}: Type: Pooling Stride: {1} Filter: {2}, Input Shape: {3}'.format(layer_idx, stride, str([size, size, int(channels), int(input.get_shape()[1]/ size)]), str(input.get_shape())))
    return tf.nn.max_pool(input, strides=[1, stride, stride, 1], ksize=[1, size, size, 1], padding='SAME')

def FC_Layer(layer_idx, input, hidden, flat=False, linear=False, trainable=False, alpha=0.1):
    input_shape = input.get_shape().as_list()
    if flat:
        dim = input_shape[1] *input_shape[2] * input_shape[3]
        input_T = tf.transpose(input, (0, 3, 1, 2))
        input_processed = tf.reshape(input_T, [-1, dim])
    else:
        dim = input_shape[1]
        input_processed = input

    weight = tf.Variable(tf.zeros([dim, hidden]), trainable=trainable)
    bias = tf.Variable(tf.constant(0.1, shape=[hidden]), trainable=trainable)
    ip = tf.add(tf.matmul(input_processed, weight), bias)
    print('Layer {0}: Type: Fully Connected Output:{1} Input Shape: {2}'.format(layer_idx, hidden, str(input.get_shape())))
    return tf.maximum(tf.multiply(alpha, ip), ip)

def Dropout_Layer(layer_idx, input, dropput_prob):
    print('Layer {0}: Type: Dropout Rate:{1} Input Shape: {2}'.format(layer_idx, dropput_prob, str(input.get_shape())))
    return tf.nn.dropout(input, keep_prob=dropput_prob)