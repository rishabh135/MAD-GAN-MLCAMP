from __future__ import division
import os
import time
from glob import glob
import tensorflow as tf
import numpy as np
from six.moves import xrange

from ops import *
from utils import *
import pdb
import re

class MADGAN(object):
    def __init__(self, sess, image_size=128, is_crop=True,
                 batch_size=64, sample_size = 64, output_size=64,
                 y_dim=None, z_dim=100, gf_dim=64, df_dim=64, N = 5,
                 gfc_dim=1024, dfc_dim=1024, c_dim=3, dataset_name='default',
                 checkpoint_dir=None):
        """
        Args:
            sess: TensorFlow session
            batch_size: The size of batch. Should be specified before training.
            output_size: (optional) The resolution in pixels of the images. [64]
            y_dim: (optional) Dimension of dim for y. [None]
            z_dim: (optional) Dimension of dim for Z. [100]
            gf_dim: (optional) Dimension of gen filters in first conv layer. [64]
            df_dim: (optional) Dimension of discrim filters in first conv layer. [64]
            gfc_dim: (optional) Dimension of gen units for for fully connected layer. [1024]
            dfc_dim: (optional) Dimension of discrim units for fully connected layer. [1024]
            c_dim: (optional) Dimension of image color. For grayscale input, set to 1. [3]
        
            N : number of generators taken
        """
        self.sess = sess
        self.is_crop = is_crop
        self.is_grayscale = (c_dim == 1)
        self.batch_size = batch_size
        self.image_size = image_size
        self.sample_size = sample_size
        self.output_size = output_size

        # y_dim is the conditional signal
        # self.y_dim = y_dim
        self.z_dim = z_dim

        self.gf_dim = gf_dim
        self.df_dim = df_dim

        self.gfc_dim = gfc_dim
        self.dfc_dim = dfc_dim

        self.c_dim = c_dim

        self.N = N

	# ------------batch norm-------------------
	

    # batchnorm that share vars for generators
        self.g_bn0 = batch_norm(name='g_bn0')
        self.g_bn1 = batch_norm(name='g_bn1')
        self.g_bn2 = batch_norm(name='g_bn2')
	

        #self.d_bn2 = batch_norm(name='d_bn2')

        # # batchnorm that doesn't share vars
        # self.d1_bn1 = batch_norm(name='d1_bn1')
        # self.d2_bn1 = batch_norm(name='d2_bn1')
	# -----------------------------------------
	
        self.dataset_name = dataset_name
        self.checkpoint_dir = checkpoint_dir
        self.build_model_2()


    def build_model_2(self):




        if self.crop:
            image_dims = [self.output_height, self.output_width, self.c_dim]
        else:
            image_dims = [self.input_height, self.input_width, self.c_dim]


        ## input in the form of real images ??? Do I make inputs copy for all generators
        self.inputs = tf.placeholder(tf.float32, [self.batch_size] + image_dims, name='real_images')
        inputs = self.inputs


        ## Latent space placeholders
        self.z = tf.placeholder(tf.float32, [None, self.z_dim], name='z')
        self.z_sum = histogram_summary("z", self.z)


        ## Generator and discriminator described
        self.generator_list = [None] * self.N 
        self.discriminator_fake = [None] * self.N
        self.discriminator_fake_logits = [None] * self.N

        ## generator and discriminator list
        for ith_generator in range(self.N):
            self.generator_list.append(multi_generator(self, self.z, ith_generator, share_params = True, reuse=False))
            self.discriminator_fake[ith_generator] , self.discriminator_fake_logits[ith_generator] = self.discriminator(self.generator_list[ith_generator], reuse=True)
            self.discriminator_fake_sum[ith_generator] =  tf.summary.histogram("d_", self.discriminator_fake[ith_generator])            




        self.discriminator_real, self.discriminator_real_logits = self.discriminator(inputs)
        self.sampler = self.multi_generator(self.z)   

        self.discriminator_real_sum = tf.summary.histogram("d", self.discriminator_real)
        #self.discriminator_fake_sum = histogram_summary("d_", self.discriminator_fake)
        


        ### discriminator loss fake scalar
        self.discriminator_loss_fake = 0
        
        ## generator loss sum scalar
        self.generator_loss_sum = 0
        
        ## generator_loss list
        self.generator_loss  = []

        ##generator loss sum list for each geenrator loss
        self.generator_loss_sum_scalar = []
        
        self.discriminator_loss_real = tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.discriminator_real_logits, tf.ones_like(self.discriminator_real)))
        for ith_generator in range(self.N):
            self.discriminator_loss_fake = self.discriminator_loss_fake + tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.discriminator_fake_logits[ith_generator], tf.zeros_like(self.discriminator_fake[ith_generator])))    
            self.generator_loss.append(tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.discriminator_fake_logits[ith_generator], tf.ones_like(self.discriminator_fake[ith_generator]))))
            self.generator_loss_sum = generator_loss_sum + self.generator_loss[ith_generator]  
            self.generator_loss_sum_scalar.append(scalar_summary("g_loss", self.generator_loss[ith_generator]))

        self.discriminator_loss_real_sum = scalar_summary("d_loss_real", self.discriminator_loss_real)
        self.discriminator_loss_fake_sum = scalar_summary("d_loss_fake", self.discriminator_loss_fake)
                          
        self.discriminator_loss = self.discriminator_loss_real + self.discriminator_loss_fake

        # self.generator_loss_sum_scalar = scalar_summary("g_loss", self.generator_loss)
        self.discriminator_loss_sum_scalar = scalar_summary("d_loss", self.discriminator_loss)
        self.saver = tf.train.Saver()



        # all variable
        t_vars = tf.trainable_variables()
        self.d_variables = [var for var in t_vars if 'd_' in var.name]
    
        # binning variables for each generator
        self.g_variables = [[] for i in range(self.N)]

        for ith_generator in range(self.N):
            self.g_variables[ith_generator].append(var for var in t_vars if 'g_' + str(ith_generator) in var.name)
        




    def sigmoid_cross_entropy_with_logits(x, y):
        
        try:
            return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=y)
        
        except:
            return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, targets=y)








    def train_madgan(self,config):

        self.d_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1).minimize(self.discriminator_loss, var_list=self.d_variables)
        self.g_optim =[] 

        for ith_generator in range(self.N):
            #sample_z = np.random.normal(size=(self.batch_size , self.z_dim))
            self.g_optim.append(tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1).minimize(self.generator_loss[ith_generator], var_list=self.g_variables[ith_generator]))


        try:
            tf.global_variables_initializer().run()

        except:
              tf.initialize_all_variables().run()





        sample_z = tf.contrib.distributions.Normal(loc=0., scale=1.).sample([self.batch_size , self.z_dim])


        ## list of  fucntions to run for sess.run          
        self.g_sum = []

        for ith_generator in range(self.N):
            #sample_z = np.random.normal(size=(self.batch_size , self.z_dim))
            self.g_optim.append(tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1).minimize(self.generator_loss[ith_generator], var_list=self.g_variables[ith_generator]))
            self.g_sum[ith_generator] = merge_summary([self.z_sum, self.discriminator_fake_sum,self.G_sum, self.d_loss_fake_sum, self.g_loss_sum])
        

        self.d_sum = merge_summary([self.z_sum, self.discriminator_real_sum, self.discriminator_loss_real_sum, self.discriminator_loss_sum_scalar])
        self.writer = tf.summary.FileWriter("./logs", self.sess.graph)


        # if config.dataset == 'mnist':
        #     sample_inputs = self.data_X[0:self.sample_num]
        #     sample_labels = self.data_y[0:self.sample_num]
        # else:
        
        sample_files = self.data[0:self.sample_num]
        sample = [get_image(sample_file,
                    input_height=self.input_height,
                    input_width=self.input_width,
                    resize_height=self.output_height,
                    resize_width=self.output_width,
                    crop=self.crop,
                    grayscale=self.grayscale) for sample_file in sample_files]

        if (self.grayscale):
            sample_inputs = np.array(sample).astype(np.float32)[:, :, :, None]
        else:
            sample_inputs = np.array(sample).astype(np.float32)
  
        
        counter = 1
        start_time = time.time()
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        
        if could_load:
            counter = checkpoint_counter
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        for epoch in xrange(config.epoch):
            
            # if config.dataset == 'mnist':
            #     batch_idxs = min(len(self.data_X), config.train_size) // config.batch_size
            
            # else:

            self.data = glob(os.path.join("./data", config.dataset, self.input_fname_pattern))
            batch_idxs = min(len(self.data), config.train_size) // config.batch_size

    
            for idx in xrange(0, batch_idxs):
                
                # if config.dataset == 'mnist':
                #     batch_images = self.data_X[idx*config.batch_size:(idx+1)*config.batch_size]
                #     batch_labels = self.data_y[idx*config.batch_size:(idx+1)*config.batch_size]
                
                # else:
                batch_files = self.data[idx*config.batch_size:(idx+1)*config.batch_size]
                batch = [
                        get_image(batch_file,
                        input_height=self.input_height,
                        input_width=self.input_width,
                        resize_height=self.output_height,
                        resize_width=self.output_width,
                        crop=self.crop,
                        grayscale=self.grayscale) for batch_file in batch_files]
          

                if self.grayscale:
                    batch_images = np.array(batch).astype(np.float32)[:, :, :, None]
              
                else:
                    batch_images = np.array(batch).astype(np.float32)

                
        
                # if config.dataset == 'mnist':
                    
                #     # Update D network
                #     _, summary_str = self.sess.run([d_optim, self.d_sum],
                #         feed_dict={ self.inputs: batch_images,
                #             self.z: batch_z,
                #             self.y:batch_labels,
                #         })            
                #     self.writer.add_summary(summary_str, counter)

                #     # Update G network
                #     _, summary_str = self.sess.run([g_optim, self.g_sum],
                #         feed_dict={
                #             self.z: batch_z, 
                #             self.y:batch_labels,
                #         })
                #     self.writer.add_summary(summary_str, counter)


                #     # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
                #     _, summary_str = self.sess.run([g_optim, self.g_sum],
                #         feed_dict={ self.z: batch_z, self.y:batch_labels })
                #     self.writer.add_summary(summary_str, counter)
              



                #     errD_fake = self.d_loss_fake.eval({
                #         self.z: batch_z, 
                #         self.y:batch_labels
                #     })
                    
                #     errD_real = self.d_loss_real.eval({
                #         self.inputs: batch_images,
                #         self.y:batch_labels
                #     })
                    
                #     errG = self.g_loss.eval({
                #         self.z: batch_z,
                #         self.y: batch_labels
                #     })


                # else:
                    
                # Update D network
                
                for ith_generator in range(self.N):

                    batch_z = tf.contrib.distributions.Normal(loc=0., scale=1.).sample([self.batch_size , self.z_dim])

                    _, summary_str = self.sess.run([d_optim, self.d_sum],
                    feed_dict={ self.inputs: batch_images, self.z: batch_z })
                    self.writer.add_summary(summary_str, counter)




                    # Update G network
                    _, summary_str = self.sess.run([g_optim, self.g_sum],
                    feed_dict={ self.z: batch_z })
                    self.writer.add_summary(summary_str, counter)


                    # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
                    _, summary_str = self.sess.run([g_optim, self.g_sum],
                    feed_dict={ self.z: batch_z })
                    self.writer.add_summary(summary_str, counter)
              
                    errD_fake = self.d_loss_fake.eval({ self.z: batch_z })
                    errD_real = self.d_loss_real.eval({ self.inputs: batch_images })
                    errG = self.g_loss.eval({self.z: batch_z})

                counter += 1
                print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" % (epoch, idx, batch_idxs,time.time() - start_time, errD_fake+errD_real, errG))

                if np.mod(counter, 100) == 1:
                    if config.dataset == 'mnist':
                        samples, d_loss, g_loss = self.sess.run([self.sampler, self.d_loss, self.g_loss],
                            feed_dict={
                            self.z: sample_z,
                            self.inputs: sample_inputs,
                            self.y:sample_labels,
                            }
                        )
                        
                        save_images(samples, image_manifold_size(samples.shape[0]),'./{}/train_{:02d}_{:04d}.png'.format(config.sample_dir, epoch, idx))
                        print("[Sample] d_loss: %.8f, g_loss: %.8f" % (d_loss, g_loss)) 
                    else:
                        try:
                            samples, d_loss, g_loss = self.sess.run([self.sampler, self.d_loss, self.g_loss],
                                feed_dict={
                                self.z: sample_z,
                                self.inputs: sample_inputs,
                                },
                            )
                            save_images(samples, image_manifold_size(samples.shape[0]),'./{}/train_{:02d}_{:04d}.png'.format(config.sample_dir, epoch, idx))
                            print("[Sample] d_loss: %.8f, g_loss: %.8f" % (d_loss, g_loss)) 
                        
                        except:
                            print("one pic error!...")

                    if np.mod(counter, 500) == 2:
                        self.save(config.checkpoint_dir, counter)





    def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):
        shape = input_.get_shape().as_list()

        with tf.variable_scope(scope or "Linear"):
            matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,tf.random_normal_initializer(stddev=stddev))
        
            bias = tf.get_variable("bias", [output_size],initializer=tf.constant_initializer(bias_start)) 
           if with_w:
                return tf.matmul(input_, matrix) + bias, matrix, bias
            else:
                return tf.matmul(input_, matrix) + bias






    #------------------- Finding the number of the generator from their name-----------------

    def find_number_of_generator(self,name_of_generator):
        return int(re.search(r'\d+', name_of_generator).group())






































































    def build_model(self):

	   # G1, D1
        self.images1 = tf.placeholder(tf.float32, [self.batch_size] + [self.output_size, self.output_size, self.c_dim],
                                    name='real_images1')
        self.sample_images1 = tf.placeholder(tf.float32, [self.sample_size] + [self.output_size,self.output_size,self.c_dim],
                                        name='sample_images1')
	   # G2, D2	
        self.images2 = tf.placeholder(tf.float32, [self.batch_size] + [self.output_size, self.output_size, self.c_dim],
                                    name='real_images2')
        self.sample_images2 = tf.placeholder(tf.float32, [self.sample_size] + [self.output_size,self.output_size,self.c_dim],
                                        name='sample_images2')
	# Generative model input
        if self.y_dim:
            self.y= tf.placeholder(tf.float32, [self.batch_size, self.y_dim], name='y')
	# latent variable
        self.z = tf.placeholder(tf.float32, [None, self.z_dim], name='z')
        self.z_sum = tf.summary.histogram("z", self.z)

	'''
	The share_params FLAG denotes the share weight btn two network(G, D)
	The reuse FLAG denotes that we will use it to do the inference
	Note: every network should be declared with (False, False) FLAG
	'''
	# input of the generator is the concat of z, y
        self.G1 = self.generator(self.z, self.y, share_params=False, reuse=False, name='G1')
	self.G2 = self.generator(self.z, self.y, share_params=True, reuse=False, name='G2')
        # input the paired input image(natural images)
        self.D1_logits, self.D1 = self.discriminator(self.images1, self.y, share_params=False, reuse=False, name='D1')
	self.D2_logits, self.D2 = self.discriminator(self.images2, self.y, share_params=True, reuse=False, name='D2')
	# generate sample
        self.sampler1 = self.generator(self.z, self.y, share_params=True, reuse=True, name='G1')
	self.sampler2 = self.generator(self.z, self.y, share_params=True, reuse=True, name='G2')
	# input the fake images
        self.D1_logits_, self.D1_ = self.discriminator(self.G1, self.y, share_params=True, reuse=True, name='D1')
	self.D2_logits_, self.D2_ = self.discriminator(self.G2, self.y, share_params=True, reuse=True, name='D2')
        
	# B1
        self.d1_sum = tf.summary.histogram("d1", self.D1)
        self.d1__sum = tf.summary.histogram("d1_", self.D1_)
        self.G1_sum = tf.summary.image("G1", self.G1)

	# B2
	self.d2_sum = tf.summary.histogram("d2", self.D2)
        self.d2__sum = tf.summary.histogram("d2_", self.D2_)
        self.G2_sum = tf.summary.image("G2", self.G2)

	# B1
        self.d1_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.D1_logits, tf.ones_like(self.D1)*0.9))
        self.d1_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.D1_logits_,tf.ones_like(self.D1_)*0.1))
        self.g1_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.D1_logits_, tf.ones_like(self.D1_)*0.9))
	self.d1_loss_real_sum = tf.summary.scalar("d1_loss_real", self.d1_loss_real)
        self.d1_loss_fake_sum = tf.summary.scalar("d1_loss_fake", self.d1_loss_fake)
	self.d1_loss = self.d1_loss_real + self.d1_loss_fake
	self.g1_loss_sum = tf.summary.scalar("g1_loss", self.g1_loss)
        self.d1_loss_sum = tf.summary.scalar("d1_loss", self.d1_loss)

	# B2
        self.d2_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.D2_logits, tf.ones_like(self.D2)*0.9))
        self.d2_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.D2_logits_,tf.ones_like(self.D2_)*0.1))
        self.g2_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.D2_logits_, tf.ones_like(self.D2_)*0.9))
        self.d2_loss_real_sum = tf.summary.scalar("d2_loss_real", self.d2_loss_real)
        self.d2_loss_fake_sum = tf.summary.scalar("d2_loss_fake", self.d2_loss_fake)
        self.d2_loss = self.d2_loss_real + self.d2_loss_fake
        self.g2_loss_sum = tf.summary.scalar("g2_loss", self.g2_loss)
        self.d2_loss_sum = tf.summary.scalar("d2_loss", self.d2_loss)

	# sum together
	self.d_loss = self.d1_loss+self.d2_loss
	self.g_loss = self.g1_loss+self.g2_loss
	self.g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)
        self.d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)
	# all variable
        t_vars = tf.trainable_variables()
	# variable list
	self.g_vars = [var for var in t_vars if 'g1_' in var.name] + [var for var in t_vars if 'g2_' in var.name] \
				+ [var for var in t_vars if 'g_' in var.name]
	self.d_vars = [var for var in t_vars if 'd1_' in var.name] + [var for var in t_vars if 'd2_' in var.name] \
				+ [var for var in t_vars if 'd_' in var.name]

        self.saver = tf.train.Saver()

    def train(self, config):
        """Train CoGAN"""
	# data_X1 is the original image
	# data_X2 is the black-white image
	# data_y is the label
        data_X1, data_y = self.load_mnist()
	data_X2 = self.load_invert_mnist()

	# do the random shuffle for two sets -> without paired images
	idx = np.arange(len(data_y))
	np.random.shuffle(idx)
	data_X1 = data_X1[idx]
	data_y1 = data_y[idx]
	idx = np.arange(len(data_y))
	np.random.shuffle(idx)
	data_X2 = data_X2[idx]
	data_y2 = data_y[idx]

	d_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
                          .minimize(self.d_loss, var_list=self.d_vars)
        g_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
                          .minimize(self.g_loss, var_list=self.g_vars)

	tf.global_variables_initializer().run()

        self.g1_sum = tf.summary.merge([self.z_sum, self.d1__sum, 
            self.G1_sum, self.d1_loss_fake_sum, self.g1_loss_sum])
        self.d1_sum = tf.summary.merge([self.z_sum, self.d1_sum, self.d1_loss_real_sum, self.d1_loss_sum])

        self.g2_sum = tf.summary.merge([self.z_sum, self.d2__sum,
            self.G2_sum, self.d2_loss_fake_sum, self.g2_loss_sum])
        self.d2_sum = tf.summary.merge([self.z_sum, self.d2_sum, self.d2_loss_real_sum, self.d2_loss_sum])
        self.writer = tf.summary.FileWriter("./logs", self.sess.graph)

	# sample noise
        sample_z = np.random.normal(size=(self.batch_size , self.z_dim))
        sample_images1 = data_X1[0:self.batch_size]
	sample_images2 = data_X2[0:self.batch_size]
        sample_labels1 = data_y1[0:self.batch_size]
        sample_labels2 = data_y2[0:self.batch_size]
            
        counter = 1
        start_time = time.time()

        if self.load(self.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        for epoch in xrange(config.epoch):
            batch_idxs = min(len(data_X1), config.train_size) // config.batch_size

            for idx in xrange(0, batch_idxs):
                batch_images1 = data_X1[idx*config.batch_size:(idx+1)*config.batch_size]
		batch_images2 = data_X2[idx*config.batch_size:(idx+1)*config.batch_size]
                batch_labels1 = data_y1[idx*config.batch_size:(idx+1)*config.batch_size]
		batch_labels2 = data_y2[idx*config.batch_size:(idx+1)*config.batch_size]
		# z is the noise
                batch_z = np.random.normal(size=[config.batch_size, self.z_dim]).astype(np.float32)
		# Update D network
                _, summary_str = self.sess.run([d_optim, self.d1_sum],
                        feed_dict={ self.images1: batch_images1, self.images2: batch_images2, 
					self.z: batch_z})
                self.writer.add_summary(summary_str, counter)

                # Update G network
                _, summary_str = self.sess.run([g_optim, self.g1_sum],
                        feed_dict={ self.z: batch_z})
                self.writer.add_summary(summary_str, counter)
                # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
                _, summary_str = self.sess.run([g_optim, self.g1_sum],
                        feed_dict={ self.z: batch_z})
                self.writer.add_summary(summary_str, counter)

		errD = self.d_loss.eval({self.z: batch_z, self.images1: batch_images1, self.images2: batch_images2})
		errG = self.g_loss.eval({self.z: batch_z})
                counter += 1
                print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
                    % (epoch, idx, batch_idxs,
                        time.time() - start_time, errD, errG))

                if np.mod(counter, 100) == 1:
		    self.evaluate(sample_images1,sample_images2,sample_labels1,batch_labels1,sample_labels2,batch_labels2, 
				sample_z, './samples/top/train_{:02d}_{:04d}.png'.format(epoch, idx))

                if np.mod(counter, 500) == 2:
                    self.save(config.checkpoint_dir, counter)
