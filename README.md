## Facial Recognition App Using Keras and TF Lite

The model was trained on faces in the LFW funneled database: http://vis-www.cs.umass.edu/lfw/lfw-funneled.tgz
Currently, the app aims to detect people in the database in real-time.

The python script used to train the model is in the "keras_training" folder. 
To train the model locally, download the database given and extract the "lfw-funneled" folder into a known location. In "test_with_mobilenet.py" change the base_dir to the location of your "lfw_funneled" folder.

Then copy over the .tflite file and labels.txt to ".\app\src\main\assets"

On Android Studio Run App.