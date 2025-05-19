# -*- coding: utf-8 -*-
"""
Created on Fri May  9 15:24:01 2025

@author: NUGGET
"""

import numpy as np
import pickle

#loading the save model
loaded_model = pickle.load(open("C:/Users/NUGGET/Desktop/ML MODELS/cardiovascular_trained_model.sav", "rb"))


input_data = (50,0,168,62.0,110,80,1,1,0,0,0)


# changing the input data into numpy array
input_data_as_numpy_array = np.asarray(input_data)
# reshaping the array because we are predicting for one instance

input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
prediction = loaded_model.predict(input_data_reshaped)

prediction

if (prediction[0] == 0):
    print("The person is has a High risk of developing cardiovascular disease")
else:
    print("The person is has a low risk of developing cardiovascular disease")