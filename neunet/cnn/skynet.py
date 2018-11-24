# import pkgs
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.layers.convolutional import MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Dropout
import keras.backend as K


# skynet name is stupid --- should just call this minivggnet.py lol
class MiniVGGNet:
  @staticmethod
  def build(width, height, depth, classes):
  	model = Sequential()
  	inputShape = (height, width, depth)
  	chanDim = 1

  	if K.image_data_format() == 'channels_first':
  		inputShape = (depth, height, width)
  		chanDim = -1

  	# layer 1: CONV2D => BN => RELU => POOL => DROP
  	model.add(Conv2D(32, (3, 3), padding='same', input_shape=inputShape))
  	model.add(Activation('relu'))
  	model.add(BatchNormalization(axis=chanDim))
  	model.add(Conv2D(32, (3, 3), padding='same'))
  	model.add(Activation('relu'))
  	model.add(BatchNormalization(axis=chanDim))
  	model.add(MaxPooling2D(pool_size=(2, 2)))
  	model.add(Dropout(0.25))

  	# layer 2: CONV2D => BN => RELU => POOL => DROP
   	model.add(Conv2D(64, (3, 3), padding='same', input_shape=inputShape))
  	model.add(Activation('relu'))
  	model.add(BatchNormalization(axis=chanDim))
  	model.add(Conv2D(64, (3, 3), padding='same'))
  	model.add(Activation('relu'))
  	model.add(BatchNormalization(axis=chanDim))
  	model.add(MaxPooling2D(pool_size=(2, 2)))
  	model.add(Dropout(0.25))

  	# layer 3: FC => RELU
  	model.add(Flatten())
  	model.add(Dense(512))
  	model.add(Activation('relu'))
  	model.add(BatchNormalization())
  	model.add(Dropout(0.5))

  	# layer 4: Softmax
  	model.add(Dense(classes))
  	model.add(Activation('softmax'))

  	# return the constructed network arch
  	return model
