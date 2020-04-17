import keras

from keras.utils import Sequence
from keras.layers import Input, Conv2D, Conv2DTranspose, Concatenate
from keras.models import Model
from keras import optimizers
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.backend import square
from skimage.io import imread
from skimage.draw import polygon

## Build a class of network
class ggcnn():
    def __init__(self, input_channel=3, input_size=300):
        self.input = Input(shape=(input_size, input_size, input_channel))

    def model(self):
        x = Conv2D(filters=32, kernel_size=(9,9), strides=(3, 3), padding='same',
                   activation='relu')(self.input)
        x = Conv2D(filters=16, kernel_size=(5,5), strides=(2, 2), padding='same',
                   activation='relu')(x)
        x = Conv2D(filters=8,  kernel_size=(3,3), strides=(2, 2), padding='same',
                   activation='relu')(x)
        x = Conv2DTranspose(filters=8,  kernel_size=(3,3), strides=(2, 2), padding='same',
                            activation='relu')(x)
        x = Conv2DTranspose(filters=16, kernel_size=(5,5), strides=(2, 2), padding='same',
                            activation='relu')(x)
        x = Conv2DTranspose(filters=32, kernel_size=(9,9), strides=(3, 3), padding='same',
                            activation='relu')(x)
        pos_out = Conv2D(filters=1,  kernel_size=(2,2), strides=(1, 1),
                   padding='same', activation='relu')(x)
        width_out = Conv2D(filters=1,  kernel_size=(2,2), strides=(1, 1),
                   padding='same', activation='relu')(x)
        cos_out = Conv2D(filters=1,  kernel_size=(2,2), strides=(1, 1),
                   padding='same')(x)
        sin_out = Conv2D(filters=1,  kernel_size=(2,2), strides=(1, 1),
                   padding='same')(x)
        output = Concatenate()([pos_out, width_out, cos_out, sin_out])
        return Model(self.input, output)

    def compute_loss(self, y_true, y_pred):
        pos_true   = y_true[...,0]
        width_true = y_true[...,1]
        cos_true   = y_true[...,2]
        sin_true   = y_true[...,3]
        pos_pred   = y_pred[...,0]
        width_pred = y_pred[...,1]
        cos_pred   = y_pred[...,2]
        sin_pred   = y_pred[...,3]
        #pos_loss   = keras.losses.binary_crossentropy(pos_true, pos_pred)
        pos_loss   = keras.losses.mse(pos_true, pos_pred)
        width_loss = keras.losses.mse(width_true, width_pred)
        cos_loss   = keras.losses.mse(2*cos_true, 2*cos_pred)
        sin_loss   = keras.losses.mse(2*sin_true, 2*sin_pred)
        #constraint = keras.losses.mse(square((sin_pred*2-1)**2 + cos_pred**2), 1)
        #return pos_loss+width_loss+cos_loss+sin_loss+constraint
        return pos_loss+width_loss+cos_loss+sin_loss
