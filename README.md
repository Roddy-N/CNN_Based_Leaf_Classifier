# CNN BASED LEAF CLASSIFIER

This project implements a Leaf Classifier application using computer vision techniques. The model was trained on Google Colab using a dataset from Kaggle.

## Introduction

The Leaf Classifier application is designed to classify images of leaves into different species. It utilizes convolutional neural networks (CNNs) to extract features from leaf images and make predictions about their species. The model was trained on a dataset consisting of thousands of leaf images obtained from Kaggle. The training process was performed on Google Colab, leveraging its powerful GPU resources for faster training and experimentation.

## Prerequisites (For the GUI)
- Python 3.x
- Pandas
- Numpy
- OpenCv(cv2)
- TensorFlow
- Tkinter
- PIL(Python Imaging library )

## How to Run
1. clone this repository to your local machine:
			
		 git clone https://github.com/Roddy-N/CNN_Based_Leaf_Classifier.git

2. Navigate to the priject directory:

		cd CNN_Based_Leaf_Classifier

3. Run the class_list code to extract the class list

		python class_list.py

4. Run the python code:

		python Gui.py

## How to Use the GUI

1. After running the script

2. The GUI window will open with options to select an image and predict its species.

3. Click on the "Select Image" button to choose an image from your device.

4. Once an image is selected, click on the "Predict" button to classify the leaf species.

5. The predicted species will be displayed on the GUI.

6. You can reset the page and select another image.

## Additional Information
- The model used for prediction is a pre-trained CNN model stored in a file named "leaf.h5".

- The GUI application was developed using the Tkinter library for the graphical interface.

- The script automatically adjusts the window size to fit the screen dimensions.

# RUNNING THE NOTE BOOK

Google colab was used during tainig of the model 

## How to get the dataset 

- Dataset used to train the model was obtained from www.kaggle.com 
- You can use the kaggle api to download the dataset to your local machine or onto colab, use the following link

		kaggle competitions download -c classify-leaves

- after obtaining the dataset you can modify the code to train the model to your liking


### Enjoy using the project