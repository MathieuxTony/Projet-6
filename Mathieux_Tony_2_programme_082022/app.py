import streamlit as st 
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
import cv2

# Taille des images requise par le model
imgsize = 224

# Correspondance entre le numéro de classe de la prédiction et la race de chien
dico_name = {0:'Afghan_hound',
            1:'Basset',
            2:'Chihuahua',
            3:'Doberman',
            4:'English_foxhound',
            5:'Maltese'}

# Fonction pour obtenir le numéro de classe de la prédiction à partir des probabilitées renvoyées par le modèle
def get_pred_label(y_pred):
    y_pred_label = []
    for i in range(y_pred.shape[0]):
        y_pred_label.append(np.argmax(y_pred[i]))
    return y_pred_label


# Chargement du modèle
model = tf.keras.models.load_model('vgg16_model')



# L'application

st.write("""
# Application de classification de photo de chien selon la race.
""")

file = st.file_uploader("Upload a dog image", type=['jpg', 'png'])



if file is None:
	st.text('Please upload a dog image')
else:
	img = Image.open(file)
	st.image(img, caption='Votre image')
	img = img.convert("RGB")
	img.save("image/image_a_classer.jpg")
	img = cv2.imread("image/image_a_classer.jpg",cv2.IMREAD_COLOR)
	img = cv2.resize(img,(imgsize,imgsize))
	X = np.array(img)/255
	X = np.expand_dims(X, axis=0)
	pred = model.predict(X)
	pred = get_pred_label(pred)
	st.write(dico_name[pred[0]])
	