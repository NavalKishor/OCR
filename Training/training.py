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
'''
############ TO - DO ###############
1.) train architecture for input dimensions 32x32 & 64x64(if diff(acc) is not much step down to 32 can even think abt 16*16)
2.) Choice of activations tanh/relu
3.) Choice of No of conv+poollayers 
4.) No of dense layers

####################################
'''


classes = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
char_to_num = dict((c, i) for i, c in enumerate(classes))
num_to_char = dict((i, c) for i, c in enumerate(classes))

#=========================================DATA GENERATOR================================================

def datagen(folder_path, batch_size=2, input_size, num_classes = 62, val_mode=False):
    images = np.zeros((batch_size,input_size[0],input_size[1]), dtype = np.uint8)
    labels = np.zeros((batch_size, num_classes)), dtype = uint8)
	
	while True:
            for i in range(batch_size):
				if val_mode:
					fnt_num = 150
					hnd_num = 8
					folder_head = ''
				else:
					fnt_num = 1016
					hnd_num = 55
					folder_head = 'Img/'
                rand_char = '0'*(3-len(str(np.random.randint(num_classes)+1)))+str(np.random.randint(num_classes)+1)
                if np.random.randint(10): 						# Take images from Fnt
                    rand_pho = '0'*(5-len(str(np.random.randint(fnt_num)+1)))+str(np.random.randint(fnt_num)+1)
                    sub_folder = 'Fnt/'
                else:                                           # Take images from Hnd    
                    rand_pho = '0'*(3-len(str(np.random.randint(hnd_num)+1)))+str(np.random.randint(hnd_num)+1)
                    sub_folder = 'Hnd/'+folder_head
                    
                images[i] = cv2.imread(foler_path+sub_folder+'Sample'+rand_char+'/img'+rand_char+'-'+rand_pho+'_32.jpg')
                labels[i][int(rand_char)-1] = 1
            yield [images],[labels]
		



#=========================================MODEL ARCHITECTURE=======================================================
model = Sequential()

model.add(Conv2D(32, (3, 3), strides=(1, 1), padding = 'same', activation='tanh',data_format="channels_last", input_shape=(32, 32, 1)))
model.add(Conv2D(32, (3, 3), strides=(1, 1), padding = 'same', activation='tanh',data_format="channels_last"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2), padding='same'))
model.add(Dropout(0.2))

model.add(Conv2D(64, (3, 3), strides=(1, 1), padding = 'same', activation='tanh',data_format="channels_last"))
model.add(Conv2D(64, (3, 3), strides=(1, 1), padding = 'same', activation='tanh',data_format="channels_last"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2), padding='same'))
model.add(Dropout(0.2))

'''model.add(Conv2D(128, (3, 3), strides=(1, 1), padding = 'same', activation='tanh',data_format="channels_last"))
model.add(Conv2D(128, (3, 3), strides=(1, 1), padding = 'same', activation='tanh',data_format="channels_last"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2), padding='same'))
model.add(Dropout(0.2))
'''
model.add(Flatten())
#model.add(Dense(512, activation='tanh'))
model.add(Dense(256, activation='tanh'))
model.add(Dense(128, activation='tanh'))
model.add(Dense(64, activation='tanh'))

model.add(Dense(62, activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

model.summary()
#=========================================Generator Instance=======================================================


gen = datagen(folder_path='/home/captain_jack/Codes/OCR/Character_recog/train/',
              batch_size = 2,
              input_size = (32,32),
              num_classes = 62)

valid_gen = datagen(folder_path='/home/captain_jack/Codes/OCR/Character_recog/valid/',
              batch_size = 2,
              input_size = (32,32),
              num_classes = 62,
			  val_mode=True)


progbar = ProgbarLogger(count_mode='steps')
checkpoint = ModelCheckpoint("late_fusion_saved_model.hdf5", verbose=1, save_best_only=False, save_weights_only=False, mode='auto', period=1)
early = EarlyStopping(monitor='val_acc', min_delta=0, patience=1, verbose=1, mode='auto')
board = TensorBoard(log_dir='./logs', histogram_freq=2, write_graph=True, write_images=True)


model.fit_generator(datagen,
                    steps_per_epoch=100,
                    epochs=1000,
                    callbacks=[progbar,checkpoint, board],
                    #validation_data = valid_generator,
                    #validation_steps = 2,
                    max_q_size=4,
                    pickle_safe = False)
