# goal-post-detector-
searching the video file and classify the frames which included the goalpost / classifier using keras!

here i have used convolutional nural network(CNN) to train the model.And used keras where tensorflow runing the back end 

Capture the video and divide it to frames such that each frames has two second time difference 
Ex: sample video length (France v Croatia - 2018 FIFA World Cup™ FINAL – HIGHLIGHTS) – 130s
Number of fames divided into – 130/2= 65

then prepared and arange the image directories with captured thresholded(binary) images and use them to train a CNN model
image directory:
   1.train
      -negetive 
      -positive
   2.validation 
      -negetive 
      -positive
      
first i  used 210 testing data taken from 10.29 min lasting football highlight video.
First we created a numpy array of images and then fit the trained model to the array and devided the images to two classes negative 
and positive. (Negative – ‘0’, positive – ‘1’).
Then created a classes array which contains the classes of  testes images that they are belong to.
By checking the each element (1s or 0s) in classes array, images belongs to two classes were  saved to two folders separately.

  Results on negetive: All most all images are non goalpost images. Therefore tey belongs to o class as we predicts. 
                       That means the accuracy is 100% of the model taking the negative images from the test pool.
  Results on positive: There are 171 results classified as positive class. 
                       But out of 171 of them 69 are negative images (not include the goalpost). Therefore the accuracy is around 60%.
 
FITing the model

I took couple of sample videos and fit the model to them.
According to the errors there will be negative images in the positive images,
(images should be in the ‘0’ class will be in ‘1’ class/non goalpost images in the set of goal post images ).
For the negative class there will not be included any positive images. Therefore for that case accuracy is 100%.
Then we took the error images (negative images in positive folder/ 1 class) and put them in the negative directory which is used 
for the model training and then trained the model again with increased training data. 
Then the previously tested video is tested again using the new model. The accuracy of the resulted images was increased.
Results on positive: increased up to 85% after couple of iterations

