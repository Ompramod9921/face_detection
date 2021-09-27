import streamlit as st
import cv2
from PIL import Image
import numpy as np

st.set_page_config(page_title='Face Detection',page_icon='👽')
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            footer:after {
	content:'Made with ❤️ by om pramod'; 
	visibility: visible ;
}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Loading pre-trained parameters for the cascade classifier
try:
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
except Exception:
    st.write("Error loading cascade classifiers")

def detect(image):
    faces = face_cascade.detectMultiScale(image=image, scaleFactor=1.3, minNeighbors=5)
    # Draw rectangle around faces
    for (x, y, w, h) in faces:
        
        # The following are the parameters of cv2.rectangle()
        # cv2.rectangle(image_to_draw_on, start_point, end_point, color, line_width)
        cv2.rectangle(img=image, pt1=(x, y), pt2=(x + w, y + h), color=(255, 0, 0), thickness=2)
        
        roi = image[y:y+h, x:x+w]  #region of interest
        
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
    return image,faces
    # result_img is the image with rectangle drawn on it (in case there are faces detected)
    # result_faces is the array with co-ordinates of bounding box(es)

st.title("Face Detection App :sunglasses: ")
st.write("Detects face , smile & eyes")
st.markdown("*****")
image_file = st.file_uploader("upload image",type=["png","jpg","jpeg",'webp'])

if image_file is not None:
    image_loaded = Image.open(image_file)
    image = np.array(image_loaded.convert('RGB')) #converting image into 
    if st.button("Process"):
        st.markdown("*****")
        result_img, result_faces = detect(image=image)
        st.image(result_img, use_column_width = True)
        st.success("Found {} faces\n".format(len(result_faces)))
