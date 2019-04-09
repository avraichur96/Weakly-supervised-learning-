from keras.applications.vgg16 import VGG16
from keras.models import Model     
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, Input  
import numpy as np 
import pandas as pd 

y_train = pd.read_csv("y_labels.csv", sep=',')
cropped_train = pd.DataFrame()
cropped_train = pd.read_csv("cropped_train.csv",sep=',')

"""  initialise all the 16 dataframes for different windows """
tr0 = pd.DataFrame()
tr1 = pd.DataFrame()
tr2 = pd.DataFrame()
tr3 = pd.DataFrame()
tr4 = pd.DataFrame()
tr5 = pd.DataFrame()
tr6 = pd.DataFrame()
tr7 = pd.DataFrame()
tr8 = pd.DataFrame()
tr9 = pd.DataFrame()
tr10 = pd.DataFrame()
tr11 = pd.DataFrame()
tr12 = pd.DataFrame()
tr13 = pd.DataFrame()
tr14 = pd.DataFrame()
tr15 = pd.DataFrame()

""" append the dataframes with the corresponding windows for each dataframe
example: tr0[3] denotes the first window of the fourth image 
tr2[3] denotes the third window of the fourth image"""

for index,row in m_train.iterrows():
    rows = pd.Series(row) 
    if index % 16 == 0:
        tr0.append(rows,ignore_index = True)
    if index%16 == 1:
        tr1.append(rows,ignore_index = True)
    if index%16 == 2:
        tr2.append(rows,ignore_index = True)
    if index%16 == 3:
        tr3.append(rows,ignore_index = True)
    if index%16 == 4:
        tr4.append(rows,ignore_index = True)
    if index%16 == 5:
        tr5.append(rows,ignore_index = True)
    if index%16 == 6:
        tr6.append(rows,ignore_index = True)
    if index%16 == 7:
        tr7.append(rows,ignore_index = True)
    if index%16 == 8:
        tr8.append(rows,ignore_index = True)
    if index%16 == 9:
        tr9.append(rows,ignore_index = True)
    if index%16 == 10:
        tr10.append(rows,ignore_index = True)
    if index%16 == 11:
        tr11.append(rows,ignore_index = True)
    if index%16 == 12:
        tr12.append(rows,ignore_index = True)
    if index%16 == 13:
        tr13.append(rows,ignore_index = True)
    if index%16 == 14:
        tr14.append(rows,ignore_index = True)
    if index%16 == 15:
        tr15.append(rows,ignore_index = True)
    

""" create general  VGG16 model for use """
vgg_com = VGG16()

          """ remove the dense layers """
for i in range(5):
    vgg_com.layers.pop()
          """ make layers non-trainable """
for layers in vgg_com.layers:
    layers.trainable = False
    
""" initialise the inouts for the 16 models and define their shape"""    
ip0 = Input(shape =(224,224,3))
ip1 = Input(shape =(224,224,3))
ip2 = Input(shape =(224,224,3))
ip3 = Input(shape =(224,224,3))
ip4 = Input(shape =(224,224,3))
ip5 = Input(shape =(224,224,3))
ip6 = Input(shape =(224,224,3))
ip7 = Input(shape =(224,224,3))
ip8 = Input(shape =(224,224,3))
ip9 = Input(shape =(224,224,3))
ip10 = Input(shape =(224,224,3))
ip11 = Input(shape =(224,224,3))
ip12 = Input(shape =(224,224,3))
ip13 = Input(shape =(224,224,3))
ip14 = Input(shape =(224,224,3))
ip15 = Input(shape =(224,224,3))

""" initialise the parallel models with the VGG 16 model above """
    
model0 = vgg_com 
model1 = vgg_com   
model2 = vgg_com 
model3 = vgg_com 
model4 = vgg_com 
model5 = vgg_com 
model6 = vgg_com 
model7 = vgg_com 
model8 = vgg_com 
model9 = vgg_com 
model10 = vgg_com 
model11 = vgg_com 
model12 = vgg_com 
model13 = vgg_com 
model14 = vgg_com 
model15 = vgg_com 



""" create the whole model """    

x0 = model0(ip0)
convA0 = Conv2D(33, (3,3), strides = 2, activation='relu'(x0)
convB0 = Conv2D(33, (2,2), strides = 1, activation='relu'(convA0)

x1 = model1(ip1)
convA1 = Conv2D(33, (3,3), strides = 2, activation='relu'(x1)
convB1 = Conv2D(33, (2,2), strides = 1, activation='relu'(convA1)
 
x2 = model2(ip2)
convA2 = Conv2D(33, (3,3), strides = 2, activation='relu'(x2)
convB2 = Conv2D(33, (2,2), strides = 1, activation='relu'(convA2)

x3 = model3(ip3)
convA3 = Conv2D(33, (3,3), strides = 2, activation='relu'(x3)
convB3 = Conv2D(33, (2,2), strides = 1, activation='relu'(convA3)

x4 = model4(ip4)
convA4 = Conv2D(33, (3,3), strides = 2, activation='relu'(x4)
convB4 = Conv2D(33, (2,2), strides = 1, activation='relu'(convA4)

x5 = model5(ip5)
convA5 = Conv2D(33, (3,3), strides = 2, activation='relu'(x5)
convB5 = Conv2D(33, (2,2), strides = 1, activation='relu'(convA5)

 
x6 = model6(ip6)
convA6 = Conv2D(33, (3,3), strides = 2, activation='relu'(x6)
convB6 = Conv2D(33, (2,2), strides = 1, activation='relu'(convA6)
  

x7 = model7(ip7)
convA7 = Conv2D(33, (3,3), strides = 2, activation='relu'(x7)
convB7 = Conv2D(33, (2,2), strides = 1, activation='relu'(convA7)


x8 = model8(ip8)
convA8 = Conv2D(33, (3,3), strides = 2, activation='relu'(x8)
convB8 = Conv2D(33, (2,2), strides = 1, activation='relu'(convA8)


x9 = model1(ip9)
convA9 = Conv2D(33, (3,3), strides = 2, activation='relu'(x9)
convB9 = Conv2D(33, (2,2), strides = 1, activation='relu'(convA9)


x10 = model10(ip10)
convA10 = Conv2D(33, (3,3), strides = 2, activation='relu'(x10)
convB10 = Conv2D(33, (2,2), strides = 1, activation='relu'(convA10)


x11 = model11(ip11)
convA11 = Conv2D(33, (3,3), strides = 2, activation='relu'(x11)
convB11 = Conv2D(33, (2,2), strides = 1, activation='relu'(convA11)


x12 = model12(ip12)
convA12 = Conv2D(33, (3,3), strides = 2, activation='relu'(x12)
convB12 = Conv2D(33, (2,2), strides = 1, activation='relu'(convA12)


x13 = model13(ip13)
convA13 = Conv2D(33, (3,3), strides = 2, activation='relu'(x13)
convB13 = Conv2D(33, (2,2), strides = 1, activation='relu'(convA13)


x14 = model14(ip14)
convA14 = Conv2D(33, (3,3), strides = 2, activation='relu'(x14)
convB14 = Conv2D(33, (2,2), strides = 1, activation='relu'(convA14)


x15 = model15(ip15)
convA15 = Conv2D(33, (3,3), strides = 2, activation='relu'(x15)
convB15 = Conv2D(33, (2,2), strides = 1, activation='relu'(convA15)

""" merge the layers to form one layer of 4 x 4 x 33 """
max_pooling = keras.layers.Concatenate(axis=-1)([conv0, convB1,convB2,convB3,convB4,
                                                 convB5, convB6, convB7, convB8, convB9
                                                 convB10, convB11, convB12, convB13, convB14, convB15])

final_out = MaxPooling2D(pool_size = (4,4))(max_pooling)

"""configure the inputs and outputs """
  
final_model = Model([ip0,ip1,ip2,ip3,ip4, ip5
                     ip6, ip7, ip8, ip9, ip10
                     ip11, ip12, ip13, ip14, ip15],[final_out])

""" make sure the model is as we want it to be """
print(final_model.summary())
    

final_model.compile(optimizer='adam',loss = custom_loss, metrics=['accuracy'])

"""fit the model """ 

final_model.fit([tr0,tr1,tr1,tr1,tr1,tr1,
                 tr1,tr1,tr1,tr1,tr1
                 tr1,tr1,tr1,tr1,tr1], y_train,
                 nb_epoch=10, batch_size=10)

""" custom loss function """ 

def custom_loss(y_true, y_pred):
    x,y = y_pred.get_shape()
    yo = 0
    loss = 0
    for i in range(1,y+1):
        yo = 1 + tf.math.exp(-y_true[i] * y_pred[i])
        yo = tf.math.log(yo)
        loss = loss  + yo 
    return loss


