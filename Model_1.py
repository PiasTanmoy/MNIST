# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from __future__ import print_function
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers.core import Activation
from keras.optimizers import SGD
from keras.utils import np_utils
np.random.seed(0)


N_EPOCH = 20
BATCH_SIZE = 128
VERBOSE = 1
N_CLASS = 10
OPTIMIZER = SGD()
N_HIDDEN_1 = 128
VALIDATION_SPLIT = 0.2


(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train.shape
image_1 = X_train[1]
import matplotlib.pyplot as plt
image_index = 7777 # You may select anything up to 60,000
print(y_train[image_index]) # The label is 8
plt.imshow(X_train[image_index], cmap='Greys')


RESHAPE = 784


N_TRAIN_DATA = X_train.shape[0]
N_TEST_DATA = X_test.shape[0]

X_train = X_train.reshape(N_TRAIN_DATA, RESHAPE)
X_test = X_test.reshape(N_TEST_DATA, RESHAPE)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')


X_train /= 225
X_test /= 225

y_train = np_utils.to_categorical(y_train, N_CLASS)
y_test = np_utils.to_categorical(y_test, N_CLASS)


model = Sequential()
model.add(Dense(N_CLASS, input_shape=(RESHAPE, )))
model.add(Activation('softmax'))
model.summary()

model.compile(loss = 'categorical_crossentropy', 
              optimizer=OPTIMIZER,
              metrics=['accuracy'])

history = model.fit(X_train, y_train, batch_size = BATCH_SIZE, 
                    epochs = N_EPOCH, verbose = VERBOSE, 
                    validation_split=VALIDATION_SPLIT)



from keras.models import load_model
# Creates a HDF5 file 'my_model.h5'
model.save('MNIST.h5')
# Deletes the existing model
#del model  
# Returns a compiled model identical to the previous one
model = load_model('MNIST.h5')


print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()



scores = model.evaluate(X_test, y_test, verbose=1)
print("Test Score: ", scores[0])
print("Accuracy: " , scores[1])





# Predicting the Test set results
y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5)
y_test2 = (y_test > 0.5)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test2, y_pred)
cm = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))

y_test_argmax = y_test.argmax(axis=1)
y_pred_argmax = y_pred.argmax(axis=1)


# =============================================================================
# Code of Confusion matrix
# =============================================================================


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


np.set_printoptions(precision=2)
class_names = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

# Plot non-normalized confusion matrix
plot_confusion_matrix(y_test_argmax, y_pred_argmax, classes=class_names,
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plot_confusion_matrix(y_test_argmax, y_pred_argmax, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

plt.show()




















