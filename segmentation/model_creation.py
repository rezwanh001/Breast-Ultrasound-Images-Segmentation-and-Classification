#-*- coding: utf-8 -*-
"""
@author: Md. Rezwanul Haque
"""
#---------------------------------------------------------------
# imports
#---------------------------------------------------------------

'''
* Sources: 
    - Follow the following link:
        -> https://keras.io/examples/vision/oxford_pets_image_segmentation/
        -> https://pyimagesearch.com/2022/02/21/u-net-image-segmentation-in-keras/ 
'''

from keras.layers import Input
from keras.layers import Conv2D
from keras.layers import Dropout
from keras.layers import Concatenate
from keras.layers import MaxPooling2D
from keras.layers import Conv2DTranspose
from keras import Model

def u_net():
    '''
    * Modifications:
        - Padding same so that I can get the mask of exact same dimensions as the actual image. 
    '''
 
    ## Contracting path
    inply = Input((128, 128, 1,))

    conv1 = Conv2D(2**6, (3,3), activation = 'relu', padding = 'same')(inply)
    conv1 = Conv2D(2**6, (3,3), activation = 'relu', padding = 'same')(conv1)
    pool1 = MaxPooling2D((2,2), strides = 2, padding = 'same')(conv1)
    drop1 = Dropout(0.2)(pool1)

    conv2 = Conv2D(2**7, (3,3), activation = 'relu', padding = 'same')(drop1)
    conv2 = Conv2D(2**7, (3,3), activation = 'relu', padding = 'same')(conv2)
    pool2 = MaxPooling2D((2,2), strides = 2, padding = 'same')(conv2)
    drop2 = Dropout(0.2)(pool2)

    conv3 = Conv2D(2**8, (3,3), activation = 'relu', padding = 'same')(drop2)
    conv3 = Conv2D(2**8, (3,3), activation = 'relu', padding = 'same')(conv3)
    pool3 = MaxPooling2D((2,2), strides = 2, padding = 'same')(conv3)
    drop3 = Dropout(0.2)(pool3)

    conv4 = Conv2D(2**9, (3,3), activation = 'relu', padding = 'same')(drop3)
    conv4 = Conv2D(2**9, (3,3), activation = 'relu', padding = 'same')(conv4)
    pool4 = MaxPooling2D((2,2), strides = 2, padding = 'same')(conv4)
    drop4 = Dropout(0.2)(pool4)


    ## Bottleneck layer
    '''
        - A bottleneck layer is a layer that contains few nodes compared to the previous layers. 
        - It can be used to obtain a representation of the input with reduced dimensionality. 
    '''
    convm = Conv2D(2**10, (3,3), activation = 'relu', padding = 'same')(drop4)
    convm = Conv2D(2**10, (3,3), activation = 'relu', padding = 'same')(convm)


    ## Expanding layer
    tran5 = Conv2DTranspose(2**9, (2,2), strides = 2, padding = 'valid', activation = 'relu')(convm)
    conc5 = Concatenate()([tran5, conv4])
    conv5 = Conv2D(2**9, (3,3), activation = 'relu', padding = 'same')(conc5)
    conv5 = Conv2D(2**9, (3,3), activation = 'relu', padding = 'same')(conv5)
    drop5 = Dropout(0.1)(conv5)

    tran6 = Conv2DTranspose(2**8, (2,2), strides = 2, padding = 'valid', activation = 'relu')(drop5)
    conc6 = Concatenate()([tran6, conv3])
    conv6 = Conv2D(2**8, (3,3), activation = 'relu', padding = 'same')(conc6)
    conv6 = Conv2D(2**8, (3,3), activation = 'relu', padding = 'same')(conv6)
    drop6 = Dropout(0.1)(conv6)

    tran7 = Conv2DTranspose(2**7, (2,2), strides = 2, padding = 'valid', activation = 'relu')(drop6)
    conc7 = Concatenate()([tran7, conv2])
    conv7 = Conv2D(2**7, (3,3), activation = 'relu', padding = 'same')(conc7)
    conv7 = Conv2D(2**7, (3,3), activation = 'relu', padding = 'same')(conv7)
    drop7 = Dropout(0.1)(conv7)

    tran8 = Conv2DTranspose(2**6, (2,2), strides = 2, padding = 'valid', activation = 'relu')(drop7)
    conc8 = Concatenate()([tran8, conv1])
    conv8 = Conv2D(2**6, (3,3), activation = 'relu', padding = 'same')(conc8)
    conv8 = Conv2D(2**6, (3,3), activation = 'relu', padding = 'same')(conv8)
    drop8 = Dropout(0.1)(conv8)


    ## Output layer
    outly = Conv2D(2**0, (1,1), activation = 'relu', padding = 'same')(drop8)
    model = Model(inputs = inply, outputs = outly, name = 'U-net')

    return model


