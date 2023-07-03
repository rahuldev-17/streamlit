import streamlit as st



from tensorflow.keras.models import load_model



import numpy as np

from tensorflow.keras.preprocessing.image import load_img, img_to_array

from PIL import Image





st.header("Image Recognition App")

st.caption("Upload an image. ")

st.caption("The application will infer the one label out of 4 labels: 'Cloudy', 'Desert', 'Green_Area', 'Water'.")

st.caption("Warning: Do not click Recognize button before uploading image. It will result in error.")



# Load the model

model = load_model("Model.h5")



# Define the class names

class_names = ['Cloudy', 'Desert', 'Green_Area', 'Water']





# Fxn

@st.cache_data


def load_image(image_file):

        img = Image.open(image_file)

        return img



imgpath = st.file_uploader("Choose a file", type =['png', 'jpeg', 'jpg'])

if imgpath is not None:

    img = load_image(imgpath )

    st.image(img, width=250)



def predict_label(image2):
        imgLoaded = load_img(image2, target_size=(255, 255))
        # Convert the image to an array
        img_array = img_to_array(imgLoaded)  
        #print(img_array)
        #print(img_array.shape)
        img_array = np.reshape(img_array, (1, 255, 255, 3))
        # Get the model predictions
        predictions = model.predict(img_array)
        #print("predictions:", predictions)
        # Get the class index with the highest predicted probability
        class_index = np.argmax(predictions[0])
        # Get the predicted class label
        predicted_label = class_names[class_index]
        return predicted_label

if st.button('Recognise'):

    predicted_label = predict_label(imgpath)

    st.write("The image is predicted to be '{}'.".format(predicted_label))



