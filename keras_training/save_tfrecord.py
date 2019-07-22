import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import cv2
import tarfile
import os
import shutil
from PIL import Image
from object_detection.utils import dataset_util
import tensorflow as tf

def read_txt(person, photo):
    # labels not defined?
    txtfile = labels+person+".txt" # labels should be a predetermined folder e.g. train or test
    txtfile_contents = open(txtfile, "r")
    txtlines = txtfile_contents.readlines()
    txtfile_contents.close()
    for line in txtlines: # find a matching line that contains photo name and coordinates of the face
        if photo in line:
            txtlines=line
    return txtlines

# modified from source: https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/using_your_own_dataset.md
def create_tf_example(photo, person, iclass, foldr): # photo is the image name without extension, person is a folder name,
    #iclass is the index associated with the person, foldr: train or test folder
    # one image at a time
    img_f=os.path.join(foldr+person,photo+".jpg")
    pic = Image.open(img_f)
    height = pic.height # Image height
    width = pic.width # Image width
    filename = str.encode(photo) # Filename of the image. Empty if image is not from file
    image_data = tf.gfile.GFile(img_f,'rb').read() # encoded image data for tfrecord use

    image_format = b'jpeg' #None #  or b'png'
    #declare coordinates
    xmins = [] # List of normalized left x coordinates in bounding box (1 per box)
    xmaxs = [] # List of normalized right x coordinates in bounding box
             # (1 per box)
    ymins = [] # List of normalized top y coordinates in bounding box (1 per box)
    ymaxs = [] # List of normalized bottom y coordinates in bounding box
             # (1 per box)
    classes_text = [] # List of string class name of bounding box (1 per box)
    classes = [] # List of integer class id of bounding box (1 per box)

    # our custom function to read labels from Labels/ directory
    txtlines = read_txt(person, photo)

    labels = txtlines.split()
    xmins.append(float(labels[1])/width) # divided by width for normalization
    xmaxs.append(float(labels[2])/width)
    ymins.append(float(labels[3])/height)
    ymaxs.append(float(labels[4])/height)

    classes_text.append(str.encode(person)) # class name (person name) person is folder

    classes.append(iclass) # class number associated with the person

    # the below code saves all the properties obtained above to tfrecord specific fields for object detection
    tf_example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': dataset_util.int64_feature(height),
      'image/width': dataset_util.int64_feature(width),
      'image/filename': dataset_util.bytes_feature(filename),
      'image/source_id': dataset_util.bytes_feature(filename),
      'image/encoded': dataset_util.bytes_feature(image_data),
      'image/format': dataset_util.bytes_feature(image_format),
      'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
      'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
      'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
      'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
      'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
      'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example

def save_tf(folder): #saving tfrecord
    tf_file=folder.split('/')[-2] +'.tfrecord' # folder names train and test, tfrecord names will also start with train and test
    writer = tf.python_io.TFRecordWriter('./'+tf_file)

    # literally {id #, name}
    labelmap = './'+'object_label.pbtxt' # for model training
    txtf = open(labelmap, "w")
    labels = './'+'labels.txt' # for android deployment
    txtl = open(labels, "w")

    for ind, person in enumerate(os.listdir(folder)):
        iclass=ind+1 # make sure label index starts from 1; zero is reserved

        #labelmap
        txtf.write("item\n{\n  id: %s\n  name: '%s'\n}\n"%(iclass,person)) # for model training

        #labels
        txtl.write("%s\n"%person) # for android deployment

        # folder is either train or test
        for photo in os.listdir(folder+person): # saving dataset and labels in tfrecord format by one image at a time
            tf_example = create_tf_example(photo.split('.')[0], person, iclass, folder)
            writer.write(tf_example.SerializeToString())
    txtf.close()
    writer.close()

modelFile ="./res10_300x300_ssd_iter_140000_fp16.caffemodel"
configFile = "./deploy.prototxt"
net = cv2.dnn.readNetFromCaffe(configFile, modelFile)

def face_dnn(img, coord=False):
    blob = cv2.dnn.blobFromImage(img, 1, (224,224), [104, 117, 123], False, False) #
    # params: source, scale=1, size=300,300, mean RGB values (r,g,b), rgb swapping=false, crop = false
    conf_threshold=0.8 # confidence at least 80%
    frameWidth=img.shape[1] # get image width
    frameHeight=img.shape[0] # get image height
    max_confidence=0
    net.setInput(blob)
    detections = net.forward()
    detection_index=0
    bboxes = []

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
                if max_confidence < confidence: # only show maximum confidence face
                    max_confidence = confidence
                    detection_index = i
    i=detection_index # face location with maximum confidence
    x1 = int(detections[0, 0, i, 3] * frameWidth) # each of i corresponds to xmin, ymin, xmax and ymax
    y1 = int(detections[0, 0, i, 4] * frameHeight)
    x2 = int(detections[0, 0, i, 5] * frameWidth)
    y2 = int(detections[0, 0, i, 6] * frameHeight)
    cv2.rectangle(img,(x1,y1),(x2,y2),(255,255,0),2) # draw a rectangle on a detected area
    cv_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # save image
    if coord==True:
        return x1, y1, x2, y2 # returns coordinates only
    return cv_rgb # returns annotated image

# function for saving coordinates of detected faces in images as txt file
# saves coordinates into respective class text files
# need to call this first
def label_txt(pathdr, lab_dir): # pathdr is our training photos folder, lab_dir is where this labels text will be saved
    for fol in os.listdir(pathdr): # for each folder (person) in train/test directory:
        tfile = open(lab_dir+fol+".txt","w+") # note that lab_dir must exist
        for img in os.listdir(pathdr+fol): # for each image in folder (one person):
            pathimg=os.path.join(pathdr+fol, img)
            #print(pathimg)
            pic=cv2.imread(pathimg)
            x1, y1, x2, y2=face_dnn(pic, True) # face detection and then saving into txt file
            # save image name + coordinates into .txt under labels Directory
            # labels directory will have N number of txt files
            tfile.write(img+' '+str(x1)+' '+str(x2)+' '+str(y1)+' '+str(y2)+'\n')
        tfile.close()
    print('Saved')

def jpg_files(members): #only extract jpg files
    for tarinfo in members:
        if os.path.splitext(tarinfo.name)[1] == ".jpg":
            yield tarinfo

def untar(fname,path="LFW"): #untarring the archive
    tar = tarfile.open(fname)
    tar.extractall(path,members=jpg_files(tar))
    tar.close()
    if path is "":
        print("File Extracted in Current Directory")
    else:
        print("File Extracted in to",  path)


fname='lfw-funneled.tgz'
pathd='./LFW/lfw_funneled/'
#untar(fname, "LFW")

# print(len(os.listdir('./LFW/lfw_funneled/')))
# total = sum([len(files) for r, d, files in os.walk('./LFW/lfw_funneled/')])
# print(total)


################################ start here after gathering got character images


count=0
imglist=[]

for r, d, files in os.walk('./LFW/lfw_funneled/'):
    if len(files)>=20:
         imglist.append(r)
         #print(count, r)
         count+=1 # counts how many folders have with at least 20 images
print(count)
#
# a = 1
# b = 1
#
# imglist[b]
# img=imglist[b]+'/'+os.listdir(imglist[b])[a]
# img=cv2.imread(img)
# c=face_dnn(img)
# plt.imshow(c)
# plt.show()


for dirs in os.listdir(pathd):
    if not (pathd+dirs) in imglist:
        shutil.rmtree(os.path.realpath(pathd+dirs))

dirs=os.listdir(pathd)
dirs.sort()

# b=np.random.randint(0,62)
# for img in os.listdir(pathd+dirs[b])[:5]:
#     #print(pathd+dirs[0]+'/'+img)
#     print(dirs[b])
#     img=cv2.imread(pathd+dirs[b]+'/'+img)
#     x1, y1, x2, y2=face_dnn(img, True)
#     #print coordinates of the detected face
#     print(x1, y1, x2, y2)
#     plt.imshow(img)
#     plt.show()


#################################### splitting folders into train (85%) and test


datadir = './LFW/'
train=datadir+'train/'
test=datadir+'test/'

if not os.path.exists(train):
    os.mkdir(train)
if not os.path.exists(test):
    os.mkdir(test)

# pathd = './LFW/lfw_funneled/'
for dirs in os.listdir(pathd):
    filenames = os.listdir(pathd+dirs)
    filenames.sort()  # make sure that the filenames have a fixed order before shuffling
    np.random.seed(402)
    np.random.shuffle(filenames) # shuffles the ordering of filenames (deterministic given the chosen seed)
    split = int(0.85 * len(filenames))
    train_filenames = filenames[:split] # splitting filenames into two parts
    test_filenames = filenames[split:]
    for img in train_filenames:
        full_file_name = os.path.join(pathd+dirs, img)
        cur_dir=os.path.join(train+dirs)
        #print(cur_dir)
        if not os.path.exists(cur_dir): # create this current person's folder for training
            os.mkdir(cur_dir)
        shutil.copy(full_file_name, cur_dir)
    for img in test_filenames:
        full_file_name = os.path.join(pathd+dirs, img)
        cur_dir=os.path.join(test+dirs)
        if not os.path.exists(cur_dir): # create this current person's folder for testing
            os.mkdir(cur_dir)
        shutil.copy(full_file_name, cur_dir)
        #a=full_file_name+' '+test+dirs
#shutil.rmtree('./LFW/lfw_funneled/')

total = sum([len(files) for r, d, files in os.walk(datadir)])
print("Total data points: {}".format(total))

labeldir="Labels/" # labels dir
wdir="./LFW/"
lab=wdir+labeldir
if not os.path.exists(lab):
    os.mkdir(lab)

lab_dir=lab+'train/'
if not os.path.exists(lab_dir):
    os.mkdir(lab_dir)
label_txt(train, lab_dir)

lab_dir=lab+'test/'
if not os.path.exists(lab_dir):
    os.mkdir(lab_dir)
label_txt(test, lab_dir)

labels='./LFW/Labels/train/'
save_tf(train)

labels='./LFW/Labels/test/'
save_tf(test)
