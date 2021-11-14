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

def Unet1():
    pass

def Unet2():
    pass

def Unet3():
    pass

def resBlock():
    pass
    
def DenseNet():
    pass

class derain(object):
    def __init__(self, sess):
        self.sess = sess
        
        # build the model
        self.input_low = tf.placeholder(tf.float32, [None, None, None, 3], name='input_low')
        self.input_high = tf.placeholder(tf.float32, [None, None, None, 3], name='input_high')
    
    def evaluate():
    
    def train():
    
    def test():
    
    def save():
    
    def load():
    