from __future__ import print_function
import os
import time
import random

from PIL import Image
import tensorflow as tf
import tensorflow.contrib.slim as slim
from skimage import color,filters
import numpy as np

from utils import *


def grad_loss(input_r_low, input_r_high):
    input_r_low_gray = tf.image.rgb_to_grayscale(input_r_low)
    input_r_high_gray = tf.image.rgb_to_grayscale(input_r_high)
    x_loss = tf.square(gradient_(input_r_low_gray, 'x') - gradient_(input_r_high_gray, 'x'))
    y_loss = tf.square(gradient_(input_r_low_gray, 'y') - gradient_(input_r_high_gray, 'y'))
    grad_loss_all = tf.reduce_mean(x_loss + y_loss)
    return grad_loss_all

def ssim_loss(output_r, input_high_r):
    output_r_1 = output_r[:,:,:,0:1]
    input_high_r_1 = input_high_r[:,:,:,0:1]
    ssim_r_1 = tf_ssim(output_r_1, input_high_r_1)
    output_r_2 = output_r[:,:,:,1:2]
    input_high_r_2 = input_high_r[:,:,:,1:2]
    ssim_r_2 = tf_ssim(output_r_2, input_high_r_2)
    output_r_3 = output_r[:,:,:,2:3]
    input_high_r_3 = input_high_r[:,:,:,2:3]
    ssim_r_3 = tf_ssim(output_r_3, input_high_r_3)
    ssim_r = (ssim_r_1 + ssim_r_2 + ssim_r_3)/3.0
    loss_ssim1 = 1-ssim_r
    return loss_ssim1
    
def concat(layers):
    return tf.concat(layers, axis=3)
    
def lrelu(x):
    return tf.nn.leaky_relu(x, alpha=0.1)

def upsample_and_concat(x1, x2, output_channels, in_channels, scope_name, trainable=True):
    with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE) as scope:
        pool_size = 2
        deconv_filter = tf.get_variable('weights', [pool_size, pool_size, output_channels, in_channels], trainable= True)
        deconv = tf.nn.conv2d_transpose(x1, deconv_filter, tf.shape(x2) , strides=[1, pool_size, pool_size, 1], name=scope_name)

        deconv_output =  tf.concat([deconv, x2],3)
        deconv_output.set_shape([None, None, None, output_channels*2])

        return deconv_output

def UNet1(input_256):
    with tf.variable_scope("Unet1", reuse=tf.AUTO_REUSE):
        conv1=slim.conv2d(input_256,16,[3,3], rate=1, activation_fn=lrelu, scope='g_conv1_1')
        pool1=slim.max_pool2d(conv1, [2, 2], stride = 2, padding='SAME' )
        conv2=slim.conv2d(pool1,32,[3,3], rate=1, activation_fn=lrelu, scope='g_conv2_1')
        pool2=slim.max_pool2d(conv2, [2, 2], stride = 2, padding='SAME' )
        conv3=slim.conv2d(pool2,64,[3,3], rate=1, activation_fn=lrelu, scope='g_conv3_1')
        
        up4 =  upsample_and_concat( conv3, conv2, 32, 64 , 'g_up_1')
        conv4=slim.conv2d(up4,  32,[3,3], rate=1, activation_fn=lrelu, scope='g_conv4_1')
        up5 =  upsample_and_concat( conv4, conv1, 16, 32 , 'g_up_2')
        conv5=slim.conv2d(up5,  16,[3,3], rate=1, activation_fn=lrelu, scope='g_conv5_1')
        
        conv6=slim.conv2d(conv5, 1, [1,1], rate=1, activation_fn=None, scope='g_conv6_1')
        out = tf.sigmoid(conv6)
    
    return out

def UNet2(input_128, input_256):
    with tf.variable_scope("Unet2", reuse=tf.AUTO_REUSE):
        bigone = slim.conv2d(input_256, 8, [1,1], rate=1, scope='nothing')
    
        conv1=slim.conv2d(input_128,16,[3,3], rate=1, activation_fn=lrelu, scope='g_conv1_2')
        pool1=slim.max_pool2d(conv1, [2, 2], stride = 2, padding='SAME' )
        conv2=slim.conv2d(pool1,32,[3,3], rate=1, activation_fn=lrelu, scope='g_conv2_2')
        pool2=slim.max_pool2d(conv2, [2, 2], stride = 2, padding='SAME' )
        conv3=slim.conv2d(pool2,64,[3,3], rate=1, activation_fn=lrelu, scope='g_conv3_2')
        
        up4 =  upsample_and_concat( conv3, conv2, 32, 64 , 'g_up_3')
        conv4=slim.conv2d(up4,  32,[3,3], rate=1, activation_fn=lrelu, scope='g_conv4_2')
        up5 =  upsample_and_concat( conv4, conv1, 16, 32 , 'g_up_4')
        conv5=slim.conv2d(up5,  16,[3,3], rate=1, activation_fn=lrelu, scope='g_conv5_2')
        
        deconv_filter = tf.get_variable('weights_u2', [2, 2, 8, 16], trainable= True)
        deconv = tf.nn.conv2d_transpose(conv5, deconv_filter, tf.shape(bigone) , strides=[1, 2, 2, 1], name='g_up_7')
        
        conv6=slim.conv2d(deconv, 1, [1,1], stride=1, rate=1, activation_fn=None, scope='g_conv6_2')
        out = tf.sigmoid(conv6)
    
    return out

def UNet3(input_64, input_128, input_256):
    with tf.variable_scope("Unet3", reuse=tf.AUTO_REUSE):
        bigone = slim.conv2d(input_128, 8, [1,1], rate=1)
        bigtwo = slim.conv2d(input_256, 4, [1,1], rate=1)
    
        conv1=slim.conv2d(input_64,16,[3,3], rate=1, activation_fn=lrelu, scope='g_conv1_3')
        pool1=slim.max_pool2d(conv1, [2, 2], stride = 2, padding='SAME' ) # 32
        conv2=slim.conv2d(pool1,32,[3,3], rate=1, activation_fn=lrelu, scope='g_conv2_3')
        pool2=slim.max_pool2d(conv2, [2, 2], stride = 2, padding='SAME' ) # 16
        conv3=slim.conv2d(pool2,64,[3,3], rate=1, activation_fn=lrelu, scope='g_conv3_3')
        
        up4 =  upsample_and_concat( conv3, conv2, 32, 64 , 'g_up_5') # 32
        conv4=slim.conv2d(up4,  32,[3,3], rate=1, activation_fn=lrelu, scope='g_conv4_3')
        up5 =  upsample_and_concat( conv4, conv1, 16, 32 , 'g_up_6')
        conv5=slim.conv2d(up5,  16,[3,3], rate=1, activation_fn=lrelu, scope='g_conv5_3')
        
        deconv_filter_1 = tf.get_variable('weights_u3_1', [2, 2, 8, 16], trainable= True)
        deconv_1 = tf.nn.conv2d_transpose(conv5, deconv_filter_1, tf.shape(bigone) , strides=[1, 2, 2, 1], name='g_up_8')        
        conv6 = slim.conv2d(deconv_1, 8, [3,3], rate=1, activation_fn=lrelu, scope='g_conv6_3')
        
        deconv_filter_2 = tf.get_variable('weights_u3_2', [2, 2, 4, 8], trainable= True)
        deconv_2 = tf.nn.conv2d_transpose(conv6, deconv_filter_2, tf.shape(bigtwo) , strides=[1, 2, 2, 1], name='g_up_9')        
        conv7 = slim.conv2d(deconv_2, 1, [1,1], rate=1, activation_fn=None, scope='g_conv7_3')
        
        out = tf.sigmoid(conv7)
    
    return out

def resBlock(x, nChannels, index):
    with tf.variable_scope('resBlock%d' % index):
        out0 = tf.layers.conv2d(x, nChannels, 3, padding='same', activation=lrelu)
        out1 = tf.layers.conv2d(out0, nChannels, 3, padding='same')
        out = tf.add(x, out1)
    return out

def denseBlock(x, nChannels, index):
    with tf.variable_scope('denseBlock%d' % index):
        out0 = tf.layers.conv2d(x, nChannels, 3, padding='same', activation=lrelu)
        out1 = tf.layers.conv2d(out0+x, nChannels, 3, padding='same', activation=lrelu)
        out2 = tf.layers.conv2d(out1+out0+x, nChannels, 3, padding='same', activation=lrelu)
    return out2
    
def DenseNet(input_concat, nChannels):
    with tf.variable_scope("DenseNet"):
        F_0 = tf.layers.conv2d(input_concat, nChannels, 3, padding='same', activation=lrelu)
        F_1 = resBlock(F_0, nChannels, 1)
        F_2 = denseBlock(F_1, nChannels, 1)
        F_2c = concat([input_concat, F_2])
        F_2c_ = tf.layers.conv2d(F_2c, nChannels, 3, padding='same', activation=lrelu)
        F_3 = denseBlock(F_2c_, nChannels, 2)
        F_4 = denseBlock(F_3, nChannels, 3)
        F_5 = resBlock(F_4, nChannels, 2)
        F_5c = concat([F_3, F_5])
        F_6 = tf.layers.conv2d(F_5c, 3, 1, padding='same')

        rain_streak_map = tf.sigmoid(F_6)
        
    return rain_streak_map
    
def RefineNet(input_derain):
    with tf.variable_scope("RefineNet", reuse=tf.AUTO_REUSE):
        conv1=slim.conv2d(input_derain,32,[3,3], rate=1, activation_fn=lrelu, scope='g_conv1_4')
        pool1=slim.max_pool2d(conv1, [2, 2], stride = 2, padding='SAME' )
        conv2=slim.conv2d(pool1,64,[3,3], rate=1, activation_fn=lrelu, scope='g_conv2_4')
        pool2=slim.max_pool2d(conv2, [2, 2], stride = 2, padding='SAME' )
        conv3=slim.conv2d(pool2,128,[3,3], rate=1, activation_fn=lrelu, scope='g_conv3_4')
        
        up4 =  upsample_and_concat( conv3, conv2, 64, 128 , 'g_up_1')
        conv4=slim.conv2d(up4,  64,[3,3], rate=1, activation_fn=lrelu, scope='g_conv4_4')
        up5 =  upsample_and_concat( conv4, conv1, 32, 64 , 'g_up_2')
        conv5=slim.conv2d(up5,  32,[3,3], rate=1, activation_fn=lrelu, scope='g_conv5_4')
        
        conv6=slim.conv2d(conv5, 3, [1,1], rate=1, activation_fn=None, scope='g_conv6_4')
        out = tf.sigmoid(conv6)
    
    return out

class derain(object):
    def __init__(self, sess):
        self.sess = sess
        
        # build the model
        self.input_low = tf.placeholder(tf.float32, [None, None, None, 3], name='input_low')
        self.input_high = tf.placeholder(tf.float32, [None, None, None, 3], name='input_high')
        self.h = tf.placeholder(tf.int32, name='h')
        self.w = tf.placeholder(tf.int32, name='w')
        
        input_256 = self.input_low
        input_128 = tf.compat.v2.image.resize(input_256, [self.h//2, self.w//2])
        input_64 = tf.compat.v2.image.resize(input_128, [self.h//4, self.w//4])
        
        feature_1 = UNet1(input_256)
        feature_2 = UNet2(input_128, input_256)
        feature_3 = UNet3(input_64, input_128, input_256)
        
        fusion0 = concat([feature_1, feature_2, feature_3])
        
        rain_streak_map = DenseNet(fusion0, nChannels=32)
        
        derain_map = self.input_low - rain_streak_map
        
        self.output = RefineNet(derain_map)
        
        # Define loss function here
        self.loss_grad = grad_loss(self.output, self.input_high)
        self.loss_square = tf.losses.mean_squared_error(self.output, self.input_high)
        self.loss_Dense = self.loss_square + 0.2 * self.loss_grad
        
        # learning rate & optimizer & train_option & trainable variables
        self.lr = tf.placeholder(tf.float32, name='learning_rate')
        optimizer = tf.train.AdamOptimizer(self.lr, name='AdamOptimizer')
        
        self.var = [var for var in tf.trainable_variables() if ('DenseNet' in var.name or 'RefineNet' in var.name or 'UNet' in var.name)]
        self.train_op = optimizer.minimize(self.loss_Dense, var_list = self.var)
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(var_list = self.var)
        
        print("[*] Initialize model successfully......")
    
    def evaluate(self, epoch_num, eval_low_data, eval_high_data, sample_dir):
        print("[*] Evaluating for epoch %d..." % (epoch_num))
        
        # For visualiztion, we don't save too many images
        for idx in range(min(len(eval_low_data), 6)):
            input_low_eval = np.expand_dims(eval_low_data[idx], axis=0)
            input_high_eval = np.expand_dims(eval_high_data[idx], axis=0)
            
            result = self.sess.run([self.output], feed_dict={self.input_low: input_low_eval, self.h: 256, self.w: 256})            
            save_images(os.path.join(sample_dir, 'eval_%d_%d.png' % (idx+1, epoch_num)), result)
    
    def train(self, train_low_data, train_high_data, eval_low_data, eval_high_data, batch_size, epoch, lr, eval_every_epoch, sample_dir, ckpt_dir):
        assert len(train_low_data) == len(train_high_data)
        assert len(eval_low_data) == len(eval_high_data)
        numBatch = len(train_low_data) // int(batch_size)
        
        # load pre-trained model
        saver = self.saver
        load_model_status, global_step = self.load(saver, ckpt_dir)
        if load_model_status:
            iter_num = global_step
            start_epoch = global_step // numBatch
            start_step = global_step % numBatch
            print("[*] Model restore success!")
        else:
            iter_num = 0
            start_epoch = 0
            start_step = 0
            print("[*] Not find pretrained model!")
        
        print("[*] Start training with start epoch %d start iter %d : " % (start_epoch, iter_num))
        # start training here
        start_time = time.time()
        train_op = self.train_op
        train_loss = self.loss_Dense
        image_id=0
        for epoch in range(start_epoch, epoch):
            for batch_id in range(start_step, numBatch):
                # generate data for a batch
                batch_input_low = np.zeros((batch_size, 256, 256, 3), dtype="float32")
                batch_input_high = np.zeros((batch_size, 256, 256, 3), dtype="float32")
                for patch_id in range(batch_size):
                    rand_mode = random.randint(0, 7)
                    batch_input_low[patch_id, :, :, :] = data_augmentation(train_low_data[image_id], rand_mode)
                    batch_input_high[patch_id, :, :, :] = data_augmentation(train_high_data[image_id], rand_mode)                  
                    
                    image_id = (image_id + 1) % len(train_low_data)
                    if image_id == 0:
                        tmp = list(zip(train_low_data, train_high_data))
                        random.shuffle(list(tmp))
                        train_low_data, train_high_data = zip(*tmp)
                              
                # train
                _, loss = self.sess.run([train_op, train_loss], feed_dict={self.input_low: batch_input_low, self.input_high: batch_input_high, self.lr: lr, self.h:256, self.w:256})
                
                print("Epoch: [%2d] [%4d/%4d] time: %4.4f, loss: %.6f" \
                      % (epoch + 1, batch_id + 1, numBatch, time.time() - start_time, loss))
                iter_num += 1
                
            # evalutate the model and save a checkpoint file for it
            if (epoch + 1) % eval_every_epoch == 0:
                self.evaluate(epoch + 1, eval_low_data, eval_high_data, sample_dir=sample_dir)
                self.save(saver, iter_num, ckpt_dir, "Derain-ver0")
        
        print("[*] Training finished")
    
    def test(self, test_low_data, test_high_data, test_low_data_names, save_dir):
        tf.global_variables_initializer().run()
        print("[*] Reading checkpoint...")
        load_model_status, _ = self.load(self.saver_Decom, './ckpts/')
        
        if load_model_status:
            print("[*] Load weights successfully...")
        
        print("[*] Testing...")
        for idx in range(len(test_low_data)):
            print(test_low_data_names[idx])
            [_, name] = os.path.split(test_low_data_names[idx])
            suffix = name[name.find('.') + 1:]
            name = name[:name.find('.')]
            
            h, w, _ = test_low_data[idx].shape
            input_low_test = np.expand_dims(test_low_data[idx], axis=0)
            
            output = self.sess.run(self.output, feed_dict = {self.input_low: input_low_test, self.h: h, self.w: w})
            
            save_images(os.path.join(save_dir, name + "_S."   + suffix), output)
    
    def save(self, saver, iter_num, ckpt_dir, model_name):
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)
        print("[*] Saving model %s" % model_name)
        saver.save(self.sess, \
                   os.path.join(ckpt_dir, model_name), \
                   global_step=iter_num)
    
    def load(self, saver, ckpt_dir):
        ckpt = tf.train.get_checkpoint_state(ckpt_dir)
        if ckpt and ckpt.model_checkpoint_path:
            full_path = tf.train.latest_checkpoint(ckpt_dir)
            try:
                global_step = int(full_path.split('/')[-1].split('-')[-1])
            except ValueError:
                global_step = None
            saver.restore(self.sess, full_path)
            return True, global_step
        else:
            print("[*] Failed to load model from %s" % ckpt_dir)
            return False, 0