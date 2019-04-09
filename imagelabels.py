import pandas as pd 
from keras.utils import to_categorical 
import numpy as np 
from sklearn.preprocessing import LabelEncoder

""" load the data to a pandas dataframe"""
annot_labels = pd.read_csv("C:\\Users\\LENOVO\\task\\annotation.csv", sep=',')  """ use your path """ 

"""make a raw list of food items fetched from labels column """ 
food_raw = []  
for entry in annot_labels['labels']:  
    a = entry
    a = a.replace("[","")
    a = a.replace("]","")
    a = a.replace("'","")
    a = a.replace(" ", "") 
    row_data = a.split(',')
    for food in row_data:
        food_raw.append(food)


""" creating dictionary for (key,value) as (food, integer encoding of food) """       
yolo = np.array(food_raw) 
lol = pd.Series(food_raw)
keys = []
keys = lol.unique()
label_encoder = LabelEncoder()
integ = label_encoder.fit_transform(yolo)
value = []
value = integ
onehot_dict = dict(zip(keys,value)) 


""" change the dictionary to (food, one hot encoding) """
for key in onehot_dict.keys():
    onehot_dict[key] = to_categorical(onehot_dict[key], num_classes = 33)

    
""" add the food one hot vectors to account for multi class"""    
temp = np.zeros(33)    
labels = pd.DataFrame()
    
for entry in annot_labels['labels']:
    a = entry
    a = a.replace("[","")
    a = a.replace("]","")
    a = a.replace("'","")
    a = a.replace(" ", "") 
    row_data = a.split(',')
    temp = np.zeros(33)
    for food in row_data:
        b = np.array(onehot_dict[food])
        temp = np.add(temp,b)
    
    temp_series = pd.Series(temp)
    labels = labels.append(temp_series, ignore_index=True)


"""convert multiple instances as one instance and make all zeroes -1 
the -1 conversion is required for the loss function """            
for index, row in labels.iterrows():
    for i in range(33):
        if row[i] == 0:
            row[i]=-1
        if row[i]>1:
            row[i] = 1

""" write to a csv file """ 
labels.to_csv("y_labels.csv")        







    
    
