'''
Based on tkwoo's anomaly GAN found here; https://github.com/tkwoo/anogan-keras'

'''
from __future__ import print_function
from keras.models import Sequential, Model
from keras.layers import Input, Reshape, Dense, Dropout, MaxPooling2D, Conv2D, Flatten, concatenate, add, Lambda
from keras.layers import Conv2DTranspose, LeakyReLU
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam, RMSprop
from keras import backend as K
from keras import initializers
import tensorflow as tf
import numpy as np
from tqdm import tqdm
import cv2
import math
from keras import layers
from keras.utils. generic_utils import Progbar
from skimage.measure import compare_ssim as ssim
import random
from SelfAttentionLayer import SelfAttention



def get_rand(ims, size):
    holder = []
    idx = [random.randint(0,ims.shape[0] - 1) for i in range(size)]
    print(idx)
    for i in idx:
        holder.append(ims[i].reshape(1, 128, 128, 1))
    return np.concatenate(holder)

def residual_block(y, nb_channels, _strides=(1, 1), _project_shortcut=False, name=None):
    shortcut = y

    # down-sampling is performed with a stride of 2
    y = layers.Conv2D(nb_channels, kernel_size=(3, 3), strides=_strides, padding='same')(y)
    y = layers.BatchNormalization()(y)
    y = layers.LeakyReLU()(y)

    y = layers.Conv2D(nb_channels, kernel_size=(3, 3), strides=(1, 1), padding='same')(y)
    y = layers.BatchNormalization()(y)

    # identity shortcuts used directly when the input and output are of the same dimensions
    if _project_shortcut or _strides != (1, 1):
        # when the dimensions increase projection shortcut is used to match dimensions (done by 1×1 convolutions)
        # when the shortcuts go across feature maps of two sizes, they are performed with a stride of 2
        shortcut = layers.Conv2D(nb_channels, kernel_size=(1, 1), strides=_strides, padding='same')(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    y = layers.add([shortcut, y])
    y = layers.LeakyReLU()(y)

    return y 
def inception_block(inputs, depth, splitted=True, activation='relu', name = None):
    actv = LeakyReLU
    
    c1_1 = Conv2D(int(depth/4), (1, 1), padding='same')(inputs)
    c2_1 = Conv2D(int(depth/8*3), (1, 1),padding='same')(inputs)
    c2_1 = actv()(c2_1)
    
    if splitted:
        c2_2 = Conv2D(int(depth/2), (1, 3), padding='same')(c2_1)
        c2_2 = BatchNormalization(axis=1)(c2_2)
        c2_2 = actv()(c2_2)
        c2_3 = Conv2D(int(depth/2), (3, 1),padding='same')(c2_2)
    else:
        c2_3 = Conv2D(int(depth/2), (3, 3),  padding='same')(c2_1)
    
    c3_1 = Conv2D(int(depth/32), (1, 1), padding='same', activation='relu')(inputs)
    c3_1 = actv()(c3_1)
    
    if splitted:
        c3_2 = Conv2D(int(depth/8), (1, 5), padding='same')(c3_1)
        c3_2 = BatchNormalization(axis=1)(c3_2)
        c3_2 = actv()(c3_2)
        c3_3 = Conv2D(int(depth/8), (5, 1), padding='same')(c3_2)
    else:
        c3_3 = Conv2D(int(depth/8), (3, 3), padding='same')(c3_1)
    
    p4_1 = MaxPooling2D(pool_size=(3,3), strides=(1,1),padding='same')(inputs)
    c4_2 = Conv2D(int(depth/8), (1, 1), padding='same')(p4_1)
    
    res = concatenate([c1_1, c2_3, c3_3, c4_2],axis=3)
    res = BatchNormalization(axis=1)(res)
    res=actv()(res)
    return res 
### combine images for visualization
def combine_images(generated_images):
    num = generated_images.shape[0]
    width = int(math.sqrt(num))
    height = int(math.ceil(float(num)/width))
    shape = generated_images.shape[1:4]
    image = np.zeros((height*shape[0], width*shape[1], shape[2]),
                     dtype=generated_images.dtype)
    for index, img in enumerate(generated_images):
        i = int(index/width)
        j = index % width
        image[i*shape[0]:(i+1)*shape[0], j*shape[1]:(j+1)*shape[1],:] = img[:, :, :]
    return image


### generator model define
def generator_model():
    inputs = Input((120,))
    input2 = Input((128, 128, 1))
    conv2_1 =inception_block(input2, 32, splitted=True, activation='relu')
    pool2_1 = MaxPooling2D(pool_size=(2, 2))(conv2_1) #14 x 14 x 32
    conv2_2 = inception_block(pool2_1, 64, splitted=True, activation='relu') #32 x 32 x 64
    pool2_2 = MaxPooling2D(pool_size=(2, 2))(conv2_2) #7 x 7 x 64
    conv2_3 = inception_block(pool2_2, 128, splitted=True, activation='relu') #16x 16 x 128 (small and thick)
    conv2_3 = Reshape((64, 64, 32))(conv2_3)
    #attention = multiply([conv2_3, pool2_1_RESHAPED])
   

    
    
    #input 1 for regular Generator training
    fc1 = Dense(input_dim=120, units=128*32*32)(inputs)
    fc1 = BatchNormalization()(fc1)
    fc1 = LeakyReLU(0.2)(fc1)
    fc2 = Reshape((64, 64, 32), input_shape=(128*32*32,))(fc1)
    fc2 = concatenate([fc2, conv2_3], axis=3)
    fc2_updated = Reshape((128, 128, 8), input_shape=(128*7*7,))(fc1)
    after_fc2= residual_block(fc2_updated, 1)
    up1 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(fc2)
    up1 = concatenate([up1, after_fc2], axis=3)
    conv1 = inception_block(up1, 128, splitted=True, activation='relu')
    #conv1 = concatenate([conv1, conv2_2], axis=3)
    pool1= MaxPooling2D(pool_size=(2, 2))(conv1)
    conv1_mod = Conv2D(128, (3, 3), padding='same')(up1)
    conv1_mod_reshaped = Reshape((128, 128, 128))(conv1_mod)
    
    conv1 = BatchNormalization()(pool1)
    conv1 = Activation('relu')(conv1)
    after_conv1= residual_block(conv1_mod_reshaped,1,_strides=(1, 1))
    up2 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv1)
    pool2 = MaxPooling2D(pool_size=(1, 1))(up2)
    up2 = concatenate([pool2, after_conv1], axis=3)
    conv2 =  inception_block(up2, 64, splitted=True, activation='relu')
    conv2 = Conv2D(1, (5, 5), padding='same')(conv2)
    outputs = Activation('tanh')(conv2)
    
    model = Model(inputs=[inputs, input2], outputs=[outputs])
    return model

def discriminator_model():
    inputs = Input((128, 128, 1))
    conv1 = Conv2D(64, (5, 5), padding='same')(inputs)
    conv1 = LeakyReLU(0.2)(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, (5, 5), padding='same')(pool1)
    conv2 = LeakyReLU(0.2)(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    fc1 = Flatten()(pool2)
    fc1 = Dense(1)(fc1)
    outputs = Activation('sigmoid')(fc1)
    model = Model(inputs=[inputs], outputs=[outputs])
    return model

### d_on_g model for training generator
def generator_containing_discriminator(g, d):
    d.trainable = False
    ganInput = [Input(shape=(120,)), Input(shape=((128, 128, 1,)))]
    x = g(ganInput)
    ganOutput = d(x)
    gan = Model(inputs=ganInput, outputs=ganOutput)
    # gan.compile(loss='binary_crossentropy', optimizer='adam')
    return gan

def load_model():
    d = discriminator_model()
    g = generator_model()
    d_optim = RMSprop()
    g_optim = RMSprop(lr=0.0002)
    g.compile(loss='binary_crossentropy', optimizer=g_optim)
    d.compile(loss='binary_crossentropy', optimizer=d_optim)
    d.load_weights('./weights/discriminator.h5')
    g.load_weights('./weights/generator.h5')
    return g, d

### train generator and discriminator
def train(BATCH_SIZE, X_train):
    
    ### model define
    d = discriminator_model()
    g = generator_model()
    d_on_g = generator_containing_discriminator(g, d)
    d_optim = RMSprop(lr=0.00004, clipvalue=1.0, decay=6e-8)
    g_optim = RMSprop(lr=0.00002, clipvalue=1.0, decay=6e-8)
    g.compile(loss='mse', optimizer=g_optim)
    d_on_g.compile(loss='mse', optimizer=g_optim)
    d.trainable = True
    d.compile(loss='mse', optimizer=d_optim)
    for epoch in range(500):
        #print ("Epoch is", epoch)
        n_iter = int(X_train.shape[0]/BATCH_SIZE)
        progress_bar = Progbar(target=n_iter)
        for index in range(n_iter):
            # create random noise -> U(0,1) 10 latent vectors
            noise = np.random.uniform(0, 1, size=(BATCH_SIZE, 120))
            # load real data & generate fake data
            rand_batch = get_rand(X_train, BATCH_SIZE)
            image_batch = X_train[index*BATCH_SIZE:(index+1)*BATCH_SIZE]
            generated_images = g.predict([noise,rand_batch], verbose=0)
            # visualize training results
            if index % 200 == 0:
                image = combine_images(generated_images)
                image = image*127.5 + 127.5
                cv2.imwrite('./result/'+str(epoch)+"_"+str(index)+".png", image)
            # attach label for training discriminator
            #print(image_batch.shape)
            X = np.concatenate((image_batch, generated_images))
            y = np.array([1] * BATCH_SIZE + [0] * BATCH_SIZE)
            # training discriminator
            d_loss = d.train_on_batch(X, y)
            # training generator
            d.trainable = False
            g_loss = d_on_g.train_on_batch([noise, image_batch], np.array([1] * BATCH_SIZE))
            d.trainable = True

            progress_bar.update(index, values=[('g',g_loss), ('d',d_loss)])
        print ('')

        # save weights for each epoch
        g.save_weights('weights/generator.h5', True)
        d.save_weights('weights/discriminator.h5', True)
    return d, g

### generate images
def generate(BATCH_SIZE):
    g = generator_model()
    g.load_weights('weights/generator.h5')
    noise = np.random.uniform(0, 1, (BATCH_SIZE, 120))
    generated_images = g.predict(noise)
    return generated_images

### anomaly loss function 
def sum_of_residual(y_true, y_pred):
    '''y_true = K.clip(y_true, 0.65, 1.0)
    y_pred = K.clip(y_pred, 0.65, 1.0)'''
    return K.sum(K.abs(y_true - y_pred))

### discriminator intermediate layer feautre extraction
def feature_extractor(d=None):
    if d is None:
        d = discriminator_model()
        d.load_weights('weights/discriminator.h5') 
    intermidiate_model = Model(inputs=d.layers[0].input, outputs=d.layers[-7].output)
    intermidiate_model.compile(loss='binary_crossentropy', optimizer='rmsprop')
    return intermidiate_model

### anomaly detection model define
def anomaly_detector(g=None, d=None):
    if g is None:
        g = generator_model()
        g.load_weights('weights/generator.h5')
    intermidiate_model = feature_extractor(d)
    intermidiate_model.trainable = False
    g = Model(inputs=[g.layers[39].input, g.layers[0].input], outputs=g.layers[-1].output)
    g.trainable = False
    # Input layer cann't be trained. Add new layer as same size & same distribution
    aInput = [Input(shape=(120,)), Input(shape=(128, 128, 1,))]
    gInput = Dense((120), trainable=True)(aInput[0])
    gInput = Activation('sigmoid')(gInput)
    gInput2 = Conv2D(1, (3, 3), activation='relu', padding='same')(aInput[1])
    gInput2 = Activation('sigmoid')(gInput2)
    # G & D feature
    G_out = g([gInput, gInput2])
    D_out= intermidiate_model(G_out)    
    model = Model(inputs=aInput, outputs=[G_out, D_out])
    model.compile(loss=sum_of_residual, loss_weights= [0.9, 0.1], optimizer='rmsprop')
    
    # batchnorm learning phase fixed (test) : make non trainable
    K.set_learning_phase(0)
    
    return model

### anomaly detection
def compute_anomaly_score(model, x, iterations=100, d=None):
    z = np.random.uniform(0, 1, size=(1,120))
    intermidiate_model = feature_extractor(d)
    d_x = intermidiate_model.predict(x)
    # learning for changing latent
    loss = model.fit([z,x], [x, d_x], batch_size=16, epochs=iterations, verbose=0)
    similar_data, _ = model.predict([z,x])
    loss = loss.history['loss'][-1]
    return loss, similar_data