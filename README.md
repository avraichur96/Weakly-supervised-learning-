# Weakly-supervised-learning-

Implementing multi-object label detection, without annotated data 

## The task 
 
We have a data set of 1400 images and all the images have one or more labels. The dataset comprises of a variety of food items arranged on the table. Investigation into the original [UNIMIB website](http://www.ivl.disco.unimib.it/activities/food-recognition/) revealed some annotations in the form of polygonal boundaries. I spent the early days trying to understand the various object detection algorithms (RCNN’s, YOLO, SSD etc.) It was only later that I realised that all these approaches were a sort of region proposal methods, with only YOLO and SSD classifying and detecting in one pass. Thus, annotation details like bounding boxes coordinates and class names were needed to train the model. Trying the same without bounding boxes is more fun! 
 
It was relatively hard to find approaches online related to object detection (without bounding boxes). All the sites I researched were mostly related to RCNN, YOLO. I found this amazing paper [1] and wanted to implement it on [Keras](https://keras.io/).   

## Possible approaches

So here is the problem again, looping inside my head. How do we classify multiple objects in an image, without data about its whereabouts? 

### 1) Basic CNN model

The immediate solution was to use sliding windows over the image, in the hope that one of these windows would capture only one food item. Each of the sliding windows is connected to a fine tuned VGG16 model of its own (details in the next section) we can predict the individual window classes and then pick the classes that scored above a threshold. The training could be done using one hot encoding over the multiple classes present in the image, with the usual mean squared loss. One caveat in the approach above is that, the model will not know the accurate location of the correction to be made (In other words, which sliding window needs to improve?) The model updates the weights using overall error to all the sliding windows. This is bad, as it penalises windows where an image might not be present. 

### 2)	Weakly supervised learning 

Reading a paper on the above topic, gave me a few more ideas. The basic window sliding approach remains with the modifications as detailed below. Here is the abstract of the paper: 

**_Abstract:_** _Successful methods for visual object recognition typically rely on training datasets containing lots of richly annotated images. Detailed image annotation, e.g. by object bounding boxes, however, is both expensive and often subjective. We describe a weakly supervised convolutional neural network (CNN) for object classification that relies only on image-level labels, yet can learn from cluttered scenes containing multiple objects. We quantify its object classification and object location prediction performance on the Pascal VOC 2012 (20 object classes) and the much larger Microsoft COCO (80 object classes) datasets. We find that the network (i) outputs accurate image-level labels, (ii) predicts approximate locations (but not extents) of objects, and (iii) performs comparably to its fully-supervised counterparts using object bounding box annotation for training._

The approach is simple. The sliding windows like approach (1) above are connected to parallel fine tuned models (described in later section) which result in a final prediction of shape 1 x 1 x (No. Of classes) for each window. 
 Collecting these outputs for all the sliding windows would result in m x n x (No. of classes) layer, for the total m x n sliding windows. 
A maximum pooling operation is performed on this last layer, resulting in a layer of dimensions 1 x 1 x (No. of classes). The following facts can be observed as an effect of max. Pooling-

a)	The window is selected corresponding to the highest confidence for a particular class; this is like selecting position of the particular class in an image.  
b)	The windows that have the object will be trained positively and the windows that do not have the object will not experience the change in weights.  

Apart from all these features, the model also has a novel loss function as shown below:
 
 
 
 ![alt text](voni.jpg)
 
 This loss function ensures that the multi-class loss is considered (‘k’ is sum over the classes) 



## Finding number of classes and one-hot Vectors 

Now it was time to convert the research paper into usable code. I set out to the first task. The annotation file in the repository consists of image labels corresponding to a particular image. First task was to find the number of unique food labels that existed. Further, every entry in the annotations ‘label’ column had to be converted into a one hot vector, to enable training, and be saved in a ‘csv’ file. The entire code snippet can be found in imagelabels.py file. 


## The fine tuned model

I decided to first resize the image into a (500,500,3) image and then crop images into sliding windows of (224,224,3). The stride used is 64 and this leads to 4 x 4 number of windows, over the 500 x 500 image. 

The following parts exist in a single fine tuned model: 

1)	*VGG16* – without the dense layers. All the layers of this model are non-trainable
2)	*Convolution layer-1* - stride = 1; filter = (3,3); number of filters = 33 (trainable layer)
3)	*Convolution layer-2* - stride = 1; filter = (2,2); number of filters = 33 (trainable layer)

The above layers are present in each of the fine tuned model. The (1) part is not trainable. Counting the number of windows we have, we see that, the model above is to be built for all 16 windows. Finally, the last layer is merged into one and a max. Pooling operation is performed. This leads to a final prediction of 1 x 1 x 33. 

## Preparing the data

The model has 16 parallel models to start with and needs 16 sections of the cropped image as inputs. I planned to first store all the 1400 images and their cropped versions in a single dataframe. First the image is read and converted into a numpy array. Splicing of the array is equivalent to cropping the image. These cropped images (16 per image) are to be stored in a dataframe one after another (row-wise). The code snippet can be found in croppedimage.py file.  
 
This code takes a very long time to run! :anguished:

## Custom loss function 

The custom loss function includes the multi-class loss. The equation was described in the earlier section. The code snippet can be found in finalmodel.py file.  

## The final model 

After figuring out all the parts, I needed to run the whole thing into a single code block. The inputs were taken from the cropped-train.csv (found in croppedimage.py) file and divided into 16 dataframe objects (one each for the fine tuned model). After this, the model layers were built using the functional API of Keras. Functional API suits our needs, as there are many parallel models. The output labels are taken from the y_labels.csv file. The model is fit using this data, and the loss is defined as the custom-loss function from the previous section. The entire code from input till the model fitting can be found in the repository. 

##  Extensions: 

1) *Localisation* - The original paper also adds a localization marker to locate the image, this needs to be implemented in the project. 
2) *scaling* - The original paper uses scaled images during training to improve the performance of the model. This is also one of the extensions possible for this project. 

## References:  

1)	[Weakly supervised classification – Is object detection for free?](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Oquab_Is_Object_Localization_2015_CVPR_paper.pdf)

2)	[One-shot object detection](https://machinethink.net/blog/object-detection/) 
 

