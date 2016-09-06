# YouthCodeJam-MachineLearning
Python Machine Learning in Image Recognition teaching program

First Install Python 2.7  (make sure you are not running any other instance of Python as you do the following installs)
If using Windows:
then from command line navigate to the Python27 folder
then run "python -m pip install -U pip"
then run "pip install beautifulsoup4"
then run "pip install -U  scikit-image"     this one takes a while, be patient (there are at least 7 dependency libraries that install with it) and then it does a long build of the windows libraries that are needed... if it errors out... just make sure you have all other instances of python shut down and re run the same command and it will run faster from the cached files)
then run "pip install -U scikit-learn"
then run "pip install -U pandas"  This one also takes a whil and several dependency libraries that install with it.  As before if it errors out, make sure you have all other instances of python shut down and re run the "pip install -U pandas" command.
then run "pip install -U cobe"  this is also one that has several dependencies but they all run very quickly.

To run the program make sure you are in the folder with the downloaded python program ML_SpotTheImage.py and in the command line type
"ML_SpotTheImage.py"

Known issues:  scatterplot does not put up the red data dots for the other images
               need to add control for how many images to pull down from BING image, current is default of 28
               need to add visualization of the test set images and which ones were properly identified and which ones were not
               when run from commandline the python script throws an unhandled error at its end and crashes instead of ending gracefully       without an error message
               need to add a deletion of images to rerun step after the program finishes the analysis and allow immediate rerun instead        of ending
