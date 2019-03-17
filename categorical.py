# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 11:43:37 2019

@author: Pias Tanmoy
"""


from numpy import array
from numpy import argmax
from keras.utils import to_categorical
from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder(handle_unknown='ignore')
# define example
data = [1, 3, 2, 0, 3, 2, 2, 1, 0, 1]
data = array(data)
print(data)
# one hot encode
encoded = to_categorical(data)
print(encoded)

t = enc.inverse_transform(encoded)
# invert encoding
inverted = argmax(encoded[0])
print(inverted)