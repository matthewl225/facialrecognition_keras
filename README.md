## Facial Recognition App Using Keras and TF Lite

The model was trained on faces in the LFW funneled database: http://vis-www.cs.umass.edu/lfw/lfw-funneled.tgz
Currently, the app aims to detect people in the database in real-time.

To train the model locally, download the database given and place the tar file into the keras_training folder. First run LFW_preprocessing.py (removes classes/people with less than 20 data points/images). Then run test_with_mobilenet.py to train on the dataset.

Then copy over the .tflite file and labels.txt outputted to ".\app\src\main\assets"

On Android Studio Run App.
