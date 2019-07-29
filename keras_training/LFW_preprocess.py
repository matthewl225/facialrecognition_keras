import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import cv2
import tarfile
import os
import shutil
from PIL import Image

def jpg_files(members): #only extract jpg files
    for tarinfo in members:
        if os.path.splitext(tarinfo.name)[1] == ".jpg":
            yield tarinfo

def untar(fname,path="LFW"): #untarring the archive
    tar = tarfile.open(fname)
    print("Extracting files...")
    tar.extractall(path,members=jpg_files(tar))
    tar.close()
    if path is "":
        print("File Extracted in Current Directory")
    else:
        print("File Extracted in to",  path)

fname='lfw-funneled.tgz'
pathd='./LFW/lfw_funneled/'
untar(fname, "LFW")

################################ start here after gathering got character images

count=0
imglist=[]

for r, d, files in os.walk('./LFW/lfw_funneled/'):
    if len(files)>=20:
         imglist.append(r)
         #print(count, r)
         count+=1 # counts how many folders have with at least 20 images
print("Number of remaining classes {}".format(count))

for dirs in os.listdir(pathd):
    if not (pathd+dirs) in imglist:
        shutil.rmtree(os.path.realpath(pathd+dirs))
print("Done")
