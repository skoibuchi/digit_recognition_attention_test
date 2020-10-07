import time
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
from keras.models import Model
from keras.optimizers import Adam
#from keras.utils import plot_model
from keras.utils import to_categorical
from keras.layers import Input, Dense, Activation
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D


# CNN + attention?
def build_model_cnn(input_dim):
	inputs = Input(shape=(28, 28, 1))
	layer1 = Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu')(inputs)
	layer2 = Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu')(layer1)
	layer3 = MaxPooling2D(pool_size=(2, 2))(layer2)
	layer3_2 = Conv2D(64, kernel_size=(3, 3), padding='same', activation='sigmoid')(layer2)
	layer3_3 = MaxPooling2D(pool_size=(2, 2))(layer3_2)
	layer3_4 = layer3 * layer3_3
	layer4 = Dropout(0.25)(layer3_4)
	layer5 = Flatten()(layer4)
	layer6 = Dense(128, activation='relu')(layer5)
	layer7 = Dropout(0.5)(layer6)
	output = Dense(10, activation='softmax')(layer7)
	model  = Model(inputs=inputs, outputs=output)
	return model

def proc():
        # read csv file
        df_train = pd.read_csv("../input/digit-recognizer/train.csv")
        df_test = pd.read_csv("../input/digit-recognizer/test.csv")
	
        # prepare train data
        # drop label from train data
	y_train = df_train["label"]
	X_train = df_train.drop(labels = ["label"],axis = 1)
	# normalization
	X_train = X_train / 255.0
        # reshape
        X_train = X_train.values.reshape(-1, 28, 28, 1)
	# one hot encoding
	y_train = to_categorical(y_train, num_classes = 10) 
        # split train data and test data
	X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, train_size=0.8)# 1×784→28×28に変換(1次元→2次元に変換)
	
        # build model
        model = build_model_cnn(X_train.shape[1])
	epochs = 12
	batch_size = 128
	model.compile(loss = "categorical_crossentropy", optimizer = Adam(), metrics = ["accuracy"])
	
	# training
        model.fit(X_train, y_train, epochs = epochs, batch_size = batch_size)
	# result 
	[loss, accuracy] = model.evaluate(X_test, y_test, verbose = 0)
	print("loss:{0} accuracy:{1}".format(loss, accuracy))

        #prepare test data
	# normalization
        X_test = df_test / 255.0
	X_test = X_test.values.reshape(-1, 28, 28, 1)
	predictions = model.predict(X_test)
	#result
	# on hot data to digit data
        df_out = [np.argmax(v, axis = None, out = None) for v in predictions]
        # add label
	df_out = pd.Series(df_out, name = "Label")
	submission = pd.concat([pd.Series(range(1, df_test.shape[0] + 1), name = "ImageId"),df_out],axis = 1)
	# output to csv file for kaggle
        submission.to_csv("submission.csv",index=False)

if __name__ == '__main__'
        proc()
