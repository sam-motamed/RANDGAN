'''
Based on tkwoo's anomaly GAN found here; https://github.com/tkwoo/anogan-keras'

'''

from __future__ import print_function
import matplotlib
matplotlib.use('Qt5Agg')
from keras import layers
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
import argparse
import anogan
import xlwt
from skimage.exposure import equalize_hist
from skimage.util.shape import view_as_blocks
parser = argparse.ArgumentParser()
parser.add_argument('--img_idx', type=int, default=14)
parser.add_argument('--label_idx', type=int, default=7)
parser.add_argument('--mode', type=str, default='train', help='train, test')
args = parser.parse_args()


### 0. prepare data
X_train = np.load('./data/Pneumonia_train.npy')
#np.save('pneu_610', X_train[:610])
X_train = X_train.reshape(-1, 128, 128, 1)
X_train = X_train.astype(np.float32)
min_scal = X_train.min()
max_scal = X_train.max()
X_train = X_train.astype(np.float32) 
print(min_scal, max_scal)

if args.mode == 'train':
    Model_d, Model_g = anogan.train(32, X_train)


def anomaly_detection(test_img, g=None, d=None):
    model = anogan.anomaly_detector(g=g, d=d)
    
    ano_score, similar_img = anogan.compute_anomaly_score(model, test_img.reshape(1, 128, 128, 1), iterations=800, d=d)

    np_residual = np.abs(test_img.reshape(128, 128,1)- similar_img.reshape(128, 128,1))
    np_residual = (np_residual + 2)/4

    np_residual = (255*np_residual).astype(np.uint8)
    original_x = (test_img.reshape(128, 128,1)*127.5 + 127.5).astype(np.uint8)
    similar_x = (similar_img.reshape(128, 128,1)*127.5 + 127.5).astype(np.uint8)

    original_x_color = cv2.cvtColor(original_x, cv2.COLOR_GRAY2BGR)
    residual_color = cv2.applyColorMap(np_residual, cv2.COLORMAP_JET)
    show = cv2.addWeighted(original_x_color, 0.3, residual_color, 0.7, 0.)

    return ano_score, original_x, similar_x, show


    
    
test_img = np.load('./data/seg_pneumonia_test.npy').reshape(-1, 128, 128, 1)
print("test size", np.shape(test_img))

print("test min and max", np.min(test_img), np.max(test_img))
normal = []
norm_all = []
wb = xlwt.Workbook()
worksheet = wb.add_sheet('Sheet 1')
for i in range(test_img.shape[0]):
    test_im = test_img[i]
    score, qurey, pred, diff = anomaly_detection(test_im)
    normal.append(score)
    worksheet.write(i + 1, 1, score)
    worksheet.write(i + 1, 2, np.count_nonzero(qurey != 0))
    worksheet.write(i + 1, 3, score / np.sqrt(np.count_nonzero(qurey != 0)))
wb.save('./modified/pneumonia_anomalyscores.xls')
### 4. tsne feature view
normal = np.array(normal)
print("MEAN", np.mean(normal))
print("MEDIAN", np.median(normal))
print("MIN", np.min(normal))
print("MAX", np.max(normal))

### t-SNE embedding 
### generating anomaly image for test (radom noise image)
