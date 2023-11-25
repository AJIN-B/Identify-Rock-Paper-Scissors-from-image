
from cnn_model import CNN
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import argparse
import cv2
import os

ROOT = os.getcwd()

def main(arg):
    
    if tf.test.is_gpu_available():
        print("GPU Available: ", tf.config.list_physical_devices('GPU'))
        print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    else :
        print("Caninot find GPU in the system")
    
    classes = ['paper', 'rock', 'scissors'] 
    
    # # calling the model
    # model = CNN()
    # # loading weights
    # model.save_weights(ROOT + '/models/cnn_weights.h5') 
    
    # loading model
    model = tf.keras.models.load_model(ROOT + '/models/cnn.h5')
    
    img = cv2.imread(arg.path)
    img = cv2.resize(img,(256,256))
    img = np.expand_dims(img,axis=0)
    
    pred = model.predict(img,verbose=0)
    pred = np.argmax(pred)
    # print(pred)
    print(f"Predicted class : {classes[pred]}")
    
    
def argment_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('-p','--path', default = ROOT + '/images/0CSaM2vL2cWX6Cay.png',
					help ='path for the testing image')
    
    args = parser.parse_args()
    return args
    

if __name__ == "__main__":
    args = argment_parser() 
    main(args)
