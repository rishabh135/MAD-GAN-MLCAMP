import tensorflow as tf
from helper import batch_norm, leaky_relu, dense, conv2d



def deconv2d(input_, output_shape,k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
             name="deconv2d", with_w=False, reuse=False, padding='SAME'):
    with tf.variable_scope(name):
        # share variable
        if reuse:
            tf.get_variable_scope().reuse_variables()
            # filter : [height, width, output_channels, in_channels]
            w = tf.get_variable('w', [k_h, k_h, output_shape[-1], input_.get_shape()[-1]],
                                initializer=tf.random_normal_initializer(stddev=stddev))
            
        try:
            deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
                                strides=[1, d_h, d_w, 1], padding=padding)

        # Support for verisons of TensorFlow before 0.7.0
        except AttributeError:
            deconv = tf.nn.deconv2d(input_, w, output_shape=output_shape,
                                strides=[1, d_h, d_w, 1], padding=padding)

        biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

        if with_w:
            return deconv, w, biases
        else:
            return deconv
       

def lrelu(x, leak=0.2, name="lrelu"):
    return tf.maximum(x, leak*x)

def prelu(x, stddev=0.02, name="prelu", reuse=False):
    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        t = tf.get_variable("tangent", [1], tf.float32, tf.random_normal_initializer(stddev=stddev))
        return tf.maximum(x, tf.mul(x, t))

def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False, reuse=False):
    shape = input_.get_shape().as_list()

    with tf.variable_scope(scope or "Linear"):
    # share variable
        if reuse:
            tf.get_variable_scope().reuse_variables()
            matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                                     tf.random_normal_initializer(stddev=stddev))
            bias = tf.get_variable("bias", [output_size],
                initializer=tf.constant_initializer(bias_start))
            if with_w:
                return tf.matmul(input_, matrix) + bias, matrix, bias
            else:
                return tf.matmul(input_, matrix) + bias


def multi_generator(gan, z, num, share_params = True, reuse = False):



    """
    Args:
        gan : instance of a generative adversarial network 
        num: Number of the generator
        reuse : Whether you want to reuse variables from previous 
        share_params : Whether weights are tied in initial layers
        z :  Latent space distribution
        

        gan.batch_size: The size of batch. Should be specified before training. [64]
        gan.output_size:  The resolution in pixels of the images. [64]
        gan.gf_dim:  Dimension of gen filters in first conv layer. [64]
        gan.gfc_dim:  Dimension of gen units for for fully connected layer. [1024]
        gan.c_dim:  Dimension of image color. For grayscale input, set to 1. [3]        
        


        Shared Layers:

        gan.g_bn0
        gan.g_bn1
        gan.g_bn2
         
    """


    if(num == 0):
        share_params = False
    z = gan.z
    s = gan.output_size
    s2, s4 = int(s/2), int(s/4)


    h0 = prelu(gan.g_bn0(linear(z, gan.gfc_dim, 'g_h0_lin', reuse=share_params), reuse=share_params), 
                    name='g_h0_prelu', reuse=share_params)

    h1 = prelu(gan.g_bn1(linear(z, gan.gf_dim*2*s4*s4,'g_h1_lin',reuse=share_params),reuse=share_params),
                    name='g_h1_prelu', reuse=share_params)
    h1 = tf.reshape(h1, [gan.batch_size, s4, s4, gan.gf_dim * 2])

    h2 = prelu(gan.g_bn2(deconv2d(h1, [gan.batch_size,s2,s2,gan.gf_dim * 2], 
        name='g_h2', reuse=share_params), reuse=share_params), name='g_h2_prelu', reuse=share_params)


    gpu_num = 0

    with tf.device('/gpu:%d' % gpu_num):

        with tf.variable_scope('g_%d' % num):
            if reuse:
                tf.get_variable_scope().reuse_variables()

            output = tf.nn.sigmoid(deconv2d(h2, [gan.batch_size, s, s, gan.c_dim], name='g'+num+'_h3', reuse=False))

            return output









# def generator(gan,num,reuse = False):

#     """
#     Args:
#         gan : instance of a generative adversarial network 
#         num: Number of the discriminator
#         reuse : Whether you want to reuse variables from previous 
#     """


#     gpu_num = 0

#     with tf.device('/gpu:%d' % gpu_num):

#         with tf.variable_scope('generator_%d' % num):
#             if reuse:
#                 tf.get_variable_scope().reuse_variables()
#             with tf.variable_scope('deconv0'):
#                 f = tf.Variable(tf.truncated_normal([3, 3, gan.num_hidden, gan.num_latent], mean=0.0,
#                                                     stddev=0.02, dtype=tf.float32),
#                                 name='filter')
#                 b = tf.Variable(tf.zeros([gan.num_hidden], dtype=tf.float32), name='b')
#                 h0 = tf.nn.bias_add(tf.nn.conv2d_transpose(gan.z, f,
#                                                            [gan.channel_size, 4, 4, gan.num_hidden],
#                                                            strides=[1, 4, 4, 1]), b)
#                 h0 = batch_norm(h0, gan.num_hidden)
#                 h0 = tf.nn.relu(h0)

#             with tf.variable_scope('deconv1'):
#                 f = tf.Variable(tf.truncated_normal([5, 5, gan.num_hidden / 2, gan.num_hidden], mean=0.0,
#                                                     stddev=0.02, dtype=tf.float32),
#                                 name='filter')
#                 b = tf.Variable(tf.zeros([gan.num_hidden / 2], dtype=tf.float32), name='b')
#                 h1 = tf.nn.bias_add(tf.nn.conv2d_transpose(h0, f,
#                                                            [gan.channel_size, 8, 8, gan.num_hidden / 2],
#                                                            strides=[1, 2, 2, 1]), b)
#                 h1 = batch_norm(h1, gan.num_hidden / 2)
#                 h1 = tf.nn.relu(h1)

#             with tf.variable_scope('deconv2'):
#                 f = tf.Variable(tf.truncated_normal([5, 5, gan.num_hidden / 4, gan.num_hidden / 2], mean=0.0,
#                                                     stddev=0.02, dtype=tf.float32),
#                                 name='filter')
#                 b = tf.Variable(tf.zeros([gan.num_hidden / 4], dtype=tf.float32), name='b')
#                 h2 = tf.nn.bias_add(tf.nn.conv2d_transpose(h1, f,
#                                                            [gan.channel_size, 16, 16, gan.num_hidden / 4],
#                                                            strides=[1, 2, 2, 1]), b)
#                 h2 = batch_norm(h2, gan.num_hidden / 4)
#                 h2 = tf.nn.relu(h2)

#             with tf.variable_scope('gen_images'):
#                 f = tf.Variable(tf.truncated_normal([5, 5, gan.num_channels, gan.num_hidden / 4], mean=0.0,
#                                                     stddev=0.02, dtype=tf.float32),
#                                 name='filter')
#                 b = tf.Variable(tf.zeros([gan.num_channels], dtype=tf.float32), name='b')
#                 gen_image = tf.nn.tanh(
#                     tf.nn.bias_add(tf.nn.conv2d_transpose(h2, f,
#                                                           [gan.channel_size, gan.side, gan.side,
#                                                            gan.num_channels],
#                                                           strides=[1, 2, 2, 1]), b))
#         return gen_image






#         self.g_bn0 = batch_norm(name='g_bn0')
#         self.g_bn1 = batch_norm(name='g_bn1')
#         self.g_bn2 = batch_norm(name='g_bn2')



# def generator(gan, inp, num, keep_prob, reuse=False):
#     gpu_num = 0
#     hidden_units = gan.h_adv
#     print(hidden_units)
#     print(keep_prob)
#     with tf.device('/gpu:%d' % gpu_num):
#         with tf.variable_scope('discriminator_%d' % num):
#             if reuse:
#                 tf.get_variable_scope().reuse_variables()
#             with tf.variable_scope('conv0'):
#                 h0 = conv2d(inp, [3, 3, gan.num_channels, hidden_units / 4], [hidden_units / 4],
#                             stride=2, name='h0')
#                 h0 = leaky_relu(0.2, h0)
#                 h0 = tf.nn.dropout(h0, keep_prob)
#             with tf.variable_scope('conv1'):
#                 h1 = conv2d(h0, [3, 3, hidden_units / 4, hidden_units / 2], [hidden_units / 2],
#                             stride=2, name='h0')
#                 h1 = leaky_relu(0.2, h1)
#                 h1 = tf.nn.dropout(h1, keep_prob)
#             with tf.variable_scope('conv2'):
#                 h2 = conv2d(h1, [3, 3, hidden_units / 2, hidden_units], [hidden_units],
#                             stride=1, name='h0')
#                 h2 = leaky_relu(0.2, h2)
#             with tf.variable_scope('reshape'):
#                 shape = h2.get_shape().as_list()
#                 num_units = shape[1] * shape[2] * shape[3]
#                 flattened = tf.reshape(h2, [gan.batch_size, num_units])
#             with tf.variable_scope('prediction'):
#                 pred = dense(flattened, [num_units, 1], [1])
#     return pred


