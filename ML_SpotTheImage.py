# -*- coding: utf-8 -*-
"""
Adapted on Mon Sep 05 14:58:57 2016 from
http://blog.yhat.com/posts/image-classification-in-Python.html
@author: Gregory B. Meyer
"""


from bs4 import BeautifulSoup
from skimage import data
from PIL import Image
from sklearn.decomposition import RandomizedPCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as pl
import requests
import re
import urllib2
import os


def get_soup(url):
    return BeautifulSoup(requests.get(url).text)
    
def img_to_matrix(filename, verbose=False):
    """
    takes a filename and turns it into a numpy array of RGB pixels
    """
    img = Image.open(filename)
    if verbose==True:
        print "changing size from %s to %s" % (str(img.size), str(STANDARD_SIZE))
    img = img.resize(STANDARD_SIZE)
    img = list(img.getdata())
    img = map(list, img)
    img = np.array(img)
    return img

def flatten_image(img):
    """
    takes in an (m, n) numpy array and flattens it 
    into an array of shape (1, m * n)
    """
    s = img.shape[0] * img.shape[1]
    img_wide = img.reshape(1, s)
    return img_wide[0]
    
def make_plot(pd):
    df = pd.DataFrame({query: X[:, 0], query1: X[:, 1], "label":np.where(y==1, query, query1)})
    colors = ["red", "yellow"]
    for label, color in zip(df['label'].unique(), colors):
        #mask = df['label']==label
        pl.scatter(df[query], df[query1], c=color, label=label)
        #    pl.scatter(df[query], df[query1], c=color)
    #df = pd.DataFrame({query: X[:, 0], query1: X[:, 1]})
    #colors = itertools.cycle(["red", "yellow"])
    #pl.scatter(df[query], df[query1], color=next(colors))
    pl.legend()
    print 'Plot displayed, waiting for it to be closed.'    
    
    
    
# input for 2 types of objects to train the ML
image_type = str(raw_input("What is your image first subject query?  (Examples: bird, dog, cat) : "))
query = image_type
image_type1 = str(raw_input("What is your image second subject query?  (Examples: turtle, fish, ball) : "))
query1 = image_type1
# add input control for Black and White or Color
BWselection = str(raw_input("Use black and white images or color? enter 'BW' or 'Color' : "))
if BWselection == "BW":  
    url = "http://www.bing.com/images/search?q=" + query  + "&qft=+filterui:color2-bw+filterui:imagesize-large&FORM=R5IR3"
else:
    url = "http://www.bing.com/images/search?q=" + query + "&qft=+filterui:imagesize-large&FORM=R5IR3"

#print "Starting to pull training images for " + query
soup = get_soup(url)
images = [a['src'] for a in soup.find_all("img", {"src": re.compile("mm.bing.net")})]

for img in images:
    raw_img = urllib2.urlopen(img).read()
    cntr = len([i for i in os.listdir("images") if image_type in i]) + 1
    f = open("images/" + image_type + "_"+ str(cntr) +".jpeg", 'wb')
    f.write(raw_img)
    f.close()
    
print "Finished pulling training images for " + query


if BWselection == "BW":
    url1 = "http://www.bing.com/images/search?q=" + query1  + "&qft=+filterui:color2-bw+filterui:imagesize-large&FORM=R5IR3"
else:
    url1 = "http://www.bing.com/images/search?q=" + query1 + "&qft=+filterui:imagesize-large&FORM=R5IR3"
#print "Starting to pull training images for " + query1
soup1 = get_soup(url1)
images1 = [a['src'] for a in soup1.find_all("img", {"src": re.compile("mm.bing.net")})]

for img1 in images1:
    raw_img1 = urllib2.urlopen(img1).read()
    cntr = len([i for i in os.listdir("images") if image_type1 in i]) + 1
    f = open("images/" + image_type1 + "_"+ str(cntr) +".jpeg", 'wb')
    f.write(raw_img1)
    f.close()

print "Finished pulling training images for " + query1
#print "Images to train your machine are loaded in the /images folder from the directory from where you are running ML_SpotTheImage.py"
    
StandardImageDefinition = int(raw_input("How many pixels do you want to sample across images? Choose a number between 50 and 140. "))

#setup a standard image size; this will distort some images but will get everything into the same shape
STANDARD_SIZE = (StandardImageDefinition, StandardImageDefinition)

img_dir = "images/"
images = [img_dir+ f for f in os.listdir(img_dir)]
labels = [query if query in f.split('/')[-1] else query1 for f in images]

data = []
print "Flattening the images and converting them to numerical data"
for image in images:
    img = img_to_matrix(image)
    img = flatten_image(img)
    data.append(img)
    
data = np.array(data)

is_train = np.random.uniform(0, 1, len(data)) <= 0.7
y = np.where(np.array(labels)==query, 1, 0)

train_x, train_y = data[is_train], y[is_train]
test_x, test_y = data[is_train==False], y[is_train==False]

#add input to specify number of components to determine
UniqueImageComponents = int(raw_input("How many unique features are needed to distinguish between your image types? Choose a number between 2 and 6. "))

pca = RandomizedPCA(n_components=UniqueImageComponents)
X = pca.fit_transform(data)


make_plot(pd)
pl.show()

train_x = pca.fit_transform(train_x)
test_x = pca.transform(test_x)
print "Training and Test sets are created."
knn = KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
           metric_params=None, n_jobs=1, n_neighbors=5, p=2,
           weights='uniform')
print "Running your machine learning model on the test set."
knn.fit(train_x, train_y)
result = knn.predict(test_x)

print "Your machine model was "
print  accuracy_score(test_y, result) 
print "% accurate!"


#pd.crosstab(test_y, knn.predict(test_x), rownames=["Actual"], colnames=["Predicted"])
#pd.pandas.ExcelWriter.save
