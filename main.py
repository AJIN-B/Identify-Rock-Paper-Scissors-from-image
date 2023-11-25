
from cnn_model import CNN
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import cv2
import os

ROOT = os.getcwd()

def main():
    
    if tf.test.is_gpu_available():
        print("GPU Available: ", tf.config.list_physical_devices('GPU'))
        print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    else :
        print("Caninot find GPU in the system")
        
    # Dataset path
    ROOT_DATASET_PATH = ROOT + "/Data/"

    # Load Dataset to the tf.dataset object
    train_image,valid_image = tf.keras.utils.image_dataset_from_directory(ROOT_DATASET_PATH,labels='inferred',
                                                            label_mode='int',color_mode='rgb',
                                                            batch_size=16,image_size=(256, 256),
                                                            shuffle=True,validation_split=0.1,
                                                            subset="both",seed=5)
    
    # calling the model
    model = CNN()
    # compling the model
    model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy']) 

    early = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy',patience=40,verbose=0,
                        mode='max',restore_best_weights=True)
    
    # train the model
    history = model.fit(train_image,epochs=100,verbose=True,
                        validation_data=valid_image,
                        callbacks=[early])
    
    # save model
    model.save(ROOT + '/models/cnn.h5')
    # save weights 
    model.save_weights(ROOT + '/models/cnn_weights.h5')
    
    val_acc = history.history['val_accuracy']
    val_loss = history.history['val_loss']
    train_acc = history.history['accuracy']
    train_loss = history.history['loss']

    plt.figure()
    plt.plot(val_acc,color='r',label='Validation Accuracy')
    plt.plot(train_acc,color='b',label='Train Accuracy')
    plt.title("Training Accuracy curve")
    plt.savefig(ROOT + '/images/accuracy curve.png')
    
    plt.figure()
    plt.plot(val_loss,color='r',label='Validation Loss')
    plt.plot(train_loss,color='b',label='Train Loss')
    plt.title("Training loss curve")
    plt.savefig(ROOT + '/images/loss curve.png')


if __name__ == "__main__":
    main()