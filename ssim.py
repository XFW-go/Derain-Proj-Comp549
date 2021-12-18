import numpy as np
import skimage.data
import cv2
import os
from skimage.measure import compare_psnr, compare_ssim

#origin_dir = '/mnt/data0/cv_proj/train/ourtest/high/'
#eval_dir = '/home/mf54/Comp549/Derain-Proj-Comp549/results/vgg16-epoch40/'

origin_dir = '/mnt/data0/cv_proj/train'
eval_dir = '/home/mf54/Comp549/Derain-Proj-Comp549/results/vgg16-20-epoch50-real/'
ori0 = os.listdir(origin_dir)
#noise_std_dev = 10
ourtest_low = '/ourtest/real_low/'
ourtest_high = '/ourtest/real_high/'

out_psnr = 0.0
out_ssim = 0.0
cnt = 0.0

# for name in ori0:
   # cnt += 1
   # img = cv2.imread(origin_dir+name)
   # img_ = cv2.imread(eval_dir+name.replace('.png','_S.png'))
   # a = compare_psnr(img, img_)
   # b = compare_ssim(img, img_, data_range=255, multichannel=True)
   # out_psnr += a
   # out_ssim += b
   
with open('/mnt/data0/cv_proj/train/test.txt', 'r') as f:
    lines = f.readlines()
    for l in lines:
        cnt+=1
        name_low = l.split(' ')[0]
        nl = name_low.split('/')[-1].replace('.png','_S.png')
        img = cv2.imread(eval_dir+nl)
        name_high = l.split(' ')[1][:-1]
        img_ = cv2.imread(origin_dir+name_high)
        #print(eval_dir+name_low, "    ", origin_dir+name_high)
        #os.system('cp %s%s %s%s' %(origin_dir, name_low, origin_dir, ourtest_low))
        #os.system('cp %s%s %s%s' %(origin_dir, name_high, origin_dir, ourtest_high))
        a = compare_psnr(img, img_)
        b = compare_ssim(img, img_, data_range=255, multichannel=True)
        out_psnr += a
        out_ssim += b

print(cnt)
print('average psnr=', out_psnr/cnt)
print('average ssim=', out_ssim/cnt)
