import tensorflow as tf
from helper import batch_norm, leaky_relu, dense, conv2d


########## Using CoGANs as baseline ####################s
def discriminator(gan, image,  reuse=False, name='Discriminator'):




    """
    Args:
        gan : instance of a generative adversarial network 
        reuse : Whether you want to reuse variables from previous 
        share_params : Whether weights are tied in initial layers

        gan.batch_size: The size of batch. Should be specified before training. [64]
        gan.output_size:  The resolution in pixels of the images. [64]
        gan.df_dim:  Dimension of gen filters in first conv layer. [64]
        gan.dfc_dim:  Dimension of gen units for for fully connected layer. [1024]
        gan.c_dim:  Dimension of image color. For grayscale input, set to 1. [3]        
        



    """



    # layers that don't share variable
    d_bn1 = batch_norm(name='d_bn1')
    d_bn2 = batch_norm(name='d_bn2')
    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()

        h0 = prelu(conv2d(image, gan.c_dim, name='d_h0_conv', reuse=False), name='d_h0_prelu', reuse=False)
        h1 = prelu(d_bn1(conv2d(h0, gan.df_dim, name='d_h1_conv', reuse=False), reuse=reuse), name='d_h1_prelu', reuse=False)
        h1 = tf.reshape(h1, [self.batch_size, -1])            

        # layers that share variables
        h2 = prelu(d_bn2(linear(h1, gan.dfc_dim, 'd_h2_lin', reuse = False),reuse = False), name='d_h2_prelu', reuse = False)
        h3 = linear(h2, 1, 'd_h3_lin', reuse = False)
            
        return tf.nn.sigmoid(h3), h3













##### Taken from GMAN implementation ###########
# def discriminator(gan, inp, num, keep_prob, reuse=False):
#     gpu_num = 0
#     hidden_units = 1024
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

