	#FOR MODIFYING IMAGES AND ARRAYS
from datetime import datetime
import os,cv2
#from cv2 import getRotationMatrix2D, warpAffine,getAffineTransform,resize,imread,BORDER_REFLECT
import numpy as np
#KERAS IMPORTS
from keras.applications.vgg16 import VGG16
from keras.callbacks import ProgbarLogger, EarlyStopping, ModelCheckpoint, TensorBoard
from keras.models import Model, Sequential
from keras.layers import Input, Convolution2D, MaxPooling2D, Conv2DTranspose, Conv2D, concatenate, Dense, Flatten
from keras.layers.core import Reshape, Activation, Dropout
from keras.preprocessing.image import *
from keras.optimizers import SGD

batch_size = 2
num_classes = 62
input_size = (32,32)
count = 0
folder_path='/home/captain_jack/Codes/OCR/Character_recog/train/'
images = np.zeros((batch_size,input_size[0],input_size[1]), dtype = np.uint8)
labels = np.zeros((batch_size, num_classes), dtype = np.uint8)
val_mode = False
while count<=20:
    for i in range(batch_size):
        if val_mode:
		fnt_num = 150
		hnd_num = 8
		folder_head = ''
	else:
		fnt_num = 716
		hnd_num = 40
		folder_head = 'Img/'
	temp =  np.random.randint(num_classes) +1
        rand_char = '0'*(3-len(str(temp)))+str(temp)

        if np.random.randint(10):
                temp = np.random.randint(fnt_num)+1
        	rand_pho = '0'*(4-len(str(temp)))+str(temp)
        	sub_folder = 'Fnt/'
        else:
                temp = np.random.randint(hnd_num)+1
        	rand_pho = '0'*(3-len(str(temp)))+str(temp)
        	sub_folder = 'Hnd/'+folder_head

        
	images[i] = cv2.cvtColor(cv2.imread(folder_path+sub_folder+'Sample'+rand_char+'/img'+rand_char+'-'+rand_pho+'_32.png'),cv2.COLOR_BGR2GRAY)
        labels[i][int(rand_char)-1] = 1
    print images,labels
    count = count + 1	
