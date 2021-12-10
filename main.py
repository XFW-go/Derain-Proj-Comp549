from __future__ import print_function
import os
import argparse
from PIL import Image
import tensorflow as tf
from utils import *
from models_vgg16 import derain

parser = argparse.ArgumentParser(description='')

parser.add_argument('--use_gpu', dest='use_gpu', type=int, default=1, help='gpu flag, 1 for GPU and 0 for CPU')
parser.add_argument('--gpu_idx', dest='gpu_idx', default="0", help='GPU idx')
parser.add_argument('--gpu_mem', dest='gpu_mem', type=float, default=0.9, help="0 to 1, gpu memory usage")
parser.add_argument('--phase', dest='phase', default='train', help='train or test')
parser.add_argument('--epoch', dest='epoch', type=int, default=100, help='number of total epoches')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=16, help='number of samples in one batch')
parser.add_argument('--start_lr', dest='start_lr', type=float, default=0.001, help='initial learning rate for adam')
parser.add_argument('--eval_every_epoch', dest='eval_every_epoch', type=int, default=20, help='evaluating and saving checkpoints every #  epoch')
parser.add_argument('--checkpoint_dir', dest='ckpt_dir', default='./ckpts', help='directory for checkpoints')
parser.add_argument('--sample_dir', dest='sample_dir', default='./sample', help='directory for evaluating outputs')
parser.add_argument('--save_dir', dest='save_dir', default='./results', help='directory for testing outputs')
parser.add_argument('--test_dir', dest='test_dir', default='./data/test/test_L', help='directory for testing inputs')

args = parser.parse_args()

def derain_train(derain):
    if not os.path.exists(args.ckpt_dir):
        os.makedirs(args.ckpt_dir)
    if not os.path.exists(args.sample_dir):
        os.makedirs(args.sample_dir)
        
    train_low_data = []
    train_high_data = []
    eval_low_data = []
    eval_high_data = []
    
    data_path = '/mnt/data0/cv_proj/train'
    with open(data_path + '/train.txt', 'r') as ftrain:
        lines = ftrain.readlines()
        for l in lines:
            name_low = l.split(' ')[0]
            low_im = load_images(data_path+name_low)
            train_low_data.append(low_im)
            name_high = l.split(' ')[1][:-1]
            high_im = load_images(data_path+name_high)
            train_high_data.append(high_im)
    
    with open(data_path + '/val.txt', 'r') as fval:
        lines = fval.readlines()
        for l in lines:
            name_low = l.split(' ')[0]
            low_im = load_images(data_path+name_low)
            eval_low_data.append(low_im)
            name_high = l.split(' ')[1][:-1]
            high_im = load_images(data_path+name_high)
            eval_high_data.append(high_im)
    
    lr = args.start_lr * np.ones([args.epoch])
    
    derain.train(train_low_data, train_high_data, eval_low_data, eval_high_data, \
                batch_size=args.batch_size, epoch=args.epoch, lr=lr, eval_every_epoch=args.eval_every_epoch, \
                sample_dir=args.sample_dir, ckpt_dir=os.path.join(args.ckpt_dir, 'vgg16-10-1e-3'))#ver0-20-2e-3'))


def derain_test(derain):
    if args.test_dir == None:
        print("[!] please provide --test_dir")
        exit(0)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    test_low_data = []
    test_high_data = []
    test_low_data_names = []
    
    data_path = '/mnt/data0/cv_proj/train'
    with open(data_path + '/test.txt', 'r') as ftest:
        lines = ftest.readlines()
        for l in lines:
            name_low = l.split(' ')[0]
            low_im = load_images(data_path+name_low)
            test_low_data.append(low_im)
            name_high = l.split(' ')[1][:-1]
            high_im = load_images(data_path+name_high)
            test_high_data.append(high_im)  
            test_low_data_names.append(name_low.split('/')[-1])
    
    derain.test(test_low_data, test_high_data, test_low_data_names, save_dir=args.save_dir)

def main(_):
    if args.use_gpu:
        print("[*] GPU\n")
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_idx
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_mem)
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            model = derain(sess)
            if args.phase == 'train':
                derain_train(model)
            elif args.phase == 'test':
                derain_test(model)
            else:
                print('[!] Unknown phase')
                exit(0)
    else:
        print("[*] CPU\n")
        with tf.Session() as sess:
            model = derain(sess)
            if args.phase == 'train':
                derain_train(model)
            elif args.phase == 'test':
                derain_test(model)
            else:
                print('[!] Unknown phase')
                exit(0)
    
if __name__ == '__main__':
    tf.app.run()