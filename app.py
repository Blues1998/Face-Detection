# Importing required libraries, obviously
import streamlit as st
import cv2
from PIL import Image
import numpy as np
import os


# Loading pre-trained parameters for the cascade classifier
try:
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
except Exception:
    st.write("Error loading cascade classifiers")

def detect(image):
    '''
    Function to detect faces/eyes and smiles in the image passed to this function
    '''

    
    image = np.array(image.convert('RGB'))
    
    # Next two lines are for converting the image from 3 channel image (RGB) into 1 channel image
    # img = cv2.cvtColor(new_img, 1)
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Passing grayscale image to perform detection
    # We pass grayscaled image because opencv expects image with one channel
    # Even if you don't convert the image into one channel, open-cv does it automatically.
    # So, you can just comment line number 26 and 27.
    # If you do, make sure that you change the variables name at appropriate places in the code below
    # Don't blame me if you run into errors while doing that :P
    
    faces = face_cascade.detectMultiScale(image=image, scaleFactor=1.3, minNeighbors=5)
    # The face_cascade classifier returns coordinates of the area in which the face might be located in the image
    # These coordinates are (x,y,w,h)
    # We will be looking for eyes and smile within this area instead of looking for them in the entire image
    # This makes sense when you're looking for smiles and eyes in a face, if that is not your use case then
    # you can pull the code segment out and make a different function for doing just that, specifically.


    # Draw rectangle around faces
    for (x, y, w, h) in faces:
        
        # The following are the parameters of cv2.rectangle()
        # cv2.rectangle(image_to_draw_on, start_point, end_point, color, line_width)
        cv2.rectangle(img=image, pt1=(x, y), pt2=(x + w, y + h), color=(255, 0, 0), thickness=2)
        
        roi = image[y:y+h, x:x+w]
        
        # Detecting eyes in the face(s) detected
        eyes = eye_cascade.detectMultiScale(roi)
        
        # Detecting smiles in the face(s) detected
        smile = smile_cascade.detectMultiScale(roi, minNeighbors = 25)
        
        # Drawing rectangle around eyes
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi, (ex, ey), (ex+ew, ey+eh), (0,255,0), 2)
            
        # Drawing rectangle around smile
        for (sx,sy,sw,sh) in smile:
            cv2.rectangle(roi, (sx, sy), (sx+sw, sy+sh), (0,0,255), 2)

    # Returning the image with bounding boxes drawn on it (in case of detected objects), and faces array
    return image, faces


def about():
	st.write(
		'''
		**Haar Cascade** is an object detection algorithm.
		It can be used to detect objects in images or videos. 

		The algorithm has four stages:

			1. Haar Feature Selection 
			2. Creating  Integral Images
			3. Adaboost Training
			4. Cascading Classifiers



Read more :point_right: https://docs.opencv.org/2.4/modules/objdetect/doc/cascade_classification.html
https://sites.google.com/site/5kk73gpu2012/assignment/viola-jones-face-detection#TOC-Image-Pyramid
		''')


def main():
    st.title("Face Detection App :sunglasses: ")
    st.write("**Using the Haar cascade Classifiers**")

    activities = ["Home", "About"]
    choice = st.sidebar.selectbox("Pick something fun", activities)

    if choice == "Home":

    	st.write("Go to the About section from the sidebar to learn more about it.")
        
        # You can specify more file types below if you want
    	image_file = st.file_uploader("Upload image", type=['jpeg', 'png', 'jpg', 'webp'])

    	if image_file is not None:

    		image = Image.open(image_file)

    		if st.button("Process"):
                
                # result_img is the image with rectangle drawn on it (in case there are faces detected)
                # result_faces is the array with co-ordinates of bounding box(es)
    			result_img, result_faces = detect(image=image)
    			st.image(result_img, use_column_width = True)
    			st.success("Found {} faces\n".format(len(result_faces)))

    elif choice == "About":
    	about()




if __name__ == "__main__":
    main()
