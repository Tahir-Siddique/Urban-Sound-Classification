# -- coding: utf-8 --
"""
Created on Fri Jul 15 23:52:31 2022

@author: Tahir Siddique
"""
# Basic Libraries

import sounddevice as sd
from scipy.io.wavfile import write
import pandas as pd
import numpy as np

from tensorflow import keras

import librosa
import librosa.display
import numpy as np

# Generates/extracts Log-MEL Spectrogram coefficients with LibRosa 
def get_mel_spectrogram(file_path, mfcc_max_padding=0, n_fft=2048, hop_length=512, n_mels=128):
    try:
        # Load audio file
        y, sr = librosa.load(file_path)

        # Normalize audio data between -1 and 1
        normalized_y = librosa.util.normalize(y)

        # Generate mel scaled filterbanks
        mel = librosa.feature.melspectrogram(normalized_y, sr=sr, n_mels=n_mels)

        # Convert sound intensity to log amplitude:
        mel_db = librosa.amplitude_to_db(abs(mel))

        # Normalize between -1 and 1
        normalized_mel = librosa.util.normalize(mel_db)

        # Should we require padding
        shape = normalized_mel.shape[1]
        if (mfcc_max_padding > 0 & shape < mfcc_max_padding):
            xDiff = mfcc_max_padding - shape
            xLeft = xDiff//2
            xRight = xDiff-xLeft
            normalized_mel = np.pad(normalized_mel, pad_width=((0,0), (xLeft, xRight)), mode='constant')

    except Exception as e:
        print("Error parsing wavefile: ", e)
        return None 
    return normalized_mel



def single_parse(file_name):
    # Here kaiser_fast is a technique used for faster extraction
    X, sample_rate = librosa.load(file_name, res_type='kaiser_fast') 
    # extracting mfcc feature from data
    mels = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T,axis=0) 
    return mels


model = keras.models.load_model('aug-train-nb3.hdf5')
import os
import librosa
import librosa.display
from scipy.io import wavfile

def takeinput(text):
    try:
        return int(input(text))
    except:
        return takeinput(text)

# model = keras.models.load_model('model')


keepworking = True

while keepworking:
    # print("1. Predict\n2. Exit")
    # choice = takeinput(">")
    fs = 22050  # Sample rate
    print('Recording...')
    seconds = 4  # Duration of recording

    myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=2)
    sd.wait()  # Wait until recording is finished
    write('output.wav', fs, myrecording)  # Save as WAV file 

    # print('Recorded.')
    # Predicting
    ft = get_mel_spectrogram(f'output.wav',n_mels=40,mfcc_max_padding=174)
    
    # size = len(ft[0])
    # print(ft.shape)
    # pad_width = 173
    # px = np.pad(ft, 
    #             pad_width=((0, 0), (0, pad_width)), 
    #             mode='constant', 
    #             constant_values=(0,))
    # print(pad_width)
    # print(px.shape)
    # print("Shape:",ft.shape)
    
    ft = ft.reshape(1,40, 174, 1)
    predictions = model.predict([ft])
    
    
    # for prediction in predictions:
    #     print(len(prediction))
        
    preds = np.argmax(predictions, axis = 1)
    result = pd.DataFrame(preds)


    classes = ['Air Conditioner','Car Horn','Childrens Playing','Dog Barking','Drilling','Engine Idling','Gun Shot','Jack Hammer','Siren','Street Music']

    results = [[i,r] for i,r in enumerate(predictions[0])]
    print("-----------------------------------------------------")
    print("------------------- Results -------------------------")
    print("-----------------------------------------------------")
        
    accuracy = 0
    for result in results:
        if(accuracy<round(result[1]*100,2)):
            accuracy = round(result[1]*100,2)
        print(f'{classes[result[0]]}:',f'{round(result[1]*100,2)}%')

    print("-----------------------------------------------------")
    print("-----------------------------------------------------")

    print("-----------------------------------------------------")
    print("=========== Predicted:",classes[preds[0]],"")
    validity = ""
    if accuracy<90:
        validity = "(Make sure voice is clear. You can check the output.wav)"
    print("=========== Accuracy:",f'{accuracy}%',validity)
    print("-----------------------------------------------------")

    input('Press any key and hit enter to continue....')

    # write(f'{classes[preds[0]]}.wav', fs, myrecording)  # Save as WAV file