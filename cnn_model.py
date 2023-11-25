import tensorflow as tf

def CNN(shape=(256,256,3)):
    
    input_lay = tf.keras.layers.Input(shape=shape)
    rescale = tf.keras.layers.Rescaling(1./127.5, offset=-1)(input_lay)    
    cnn = tf.keras.layers.Conv2D(64, (5, 5), strides=(2, 2))(rescale)
    cnn = tf.keras.layers.MaxPool2D((2,2))(cnn)
    cnn = tf.keras.layers.BatchNormalization()(cnn)
    cnn = tf.keras.layers.ReLU()(cnn)

    cnn = tf.keras.layers.Conv2D(128, (5, 5), strides=(2, 2))(cnn)
    cnn = tf.keras.layers.MaxPool2D((2,2))(cnn)
    cnn = tf.keras.layers.BatchNormalization()(cnn)
    cnn = tf.keras.layers.ReLU()(cnn)

    cnn = tf.keras.layers.Conv2D(256, (5, 5), strides=(2, 2))(cnn)
    cnn = tf.keras.layers.MaxPool2D((2,2))(cnn)
    cnn = tf.keras.layers.BatchNormalization()(cnn)
    cnn = tf.keras.layers.GlobalAvgPool2D()(cnn)

    out = tf.keras.layers.Dense(3,activation='softmax')(cnn)
    
    model = tf.keras.Model(input_lay,out)
    return model 


# model = tf.keras.Sequential()
# rescale = tf.keras.layers.Rescaling(1./127.5, offset=-1)    
# model.add(layers.Conv2D(32, (5, 5), strides=(1, 1), padding='same', 
#                         input_shape=(256,256,3)))
# model.add(layers.BatchNormalization())
# model.add(layers.ReLU())
# model.add(layers.MaxPool2D((2,2)))

# model.add(layers.Conv2D(64, (5, 5), strides=(2, 2)))
# model.add(layers.BatchNormalization())
# model.add(layers.ReLU())
# model.add(layers.MaxPool2D((2,2)))

# model.add(layers.Conv2D(128, (5, 5), strides=(2, 2)))
# model.add(layers.BatchNormalization())
# model.add(layers.Activation('sigmoid'))
# model.add(layers.GlobalAvgPool2D())

# model.add(layers.Dense(3,activation='softmax'))
# model.summary()
