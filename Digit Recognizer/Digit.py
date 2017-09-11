
from keras.models import Sequential
from keras.layers import Dense , Lambda, Flatten
#from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


train = pd.read_csv('F:/dataset/Digit Recognizer/train.csv')

y = train.pop('label')

X_train, X_test, y_train, y_test = train_test_split(train, y , test_size=0.2 )

X_train = X_train.values.astype('float32')
X_test = X_test.values.astype('float32')
y_test = y_test.values.astype('int32')
y_train = y_train.values.astype('int32')

X_train = X_train.reshape(X_train.shape[0],28,28,1)
X_test = X_test.reshape(X_test.shape[0],28,28,1)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

mean_px = X_train.mean().astype(np.float32)
std_px = X_train.std().astype(np.float32)

def standardize(x): 
    return (x-mean_px)/std_px

seed = 123
np.random.seed(seed)

#adding_layers
model = Sequential()
model.add(Lambda(standardize,input_shape=(28,28,1)))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))

#compiling
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
model.fit(X_train,y_train,epochs=30,batch_size=10)

#prediction on test split .
scores = model.predict(X_test,verbose=1)
score = pd.DataFrame(scores)
nn = score.apply(np.argmax,axis=1)
y_test = pd.DataFrame(y_test)
n = y_test.apply(np.argmax,axis=1)
error = pd.concat([n,nn],axis=1)
error['error'] = error[0] - error[1]
efficiency = error.ix[error['error']==0,:].shape[0]/error.shape[0]











