import tensorflow as tf
import tkinter as tk
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image, ImageTk

def load_predict():
    # load the model
    model = load_model('model.h5')

    # get the file name from the GUI
    file_name = entry.get()

    # load the image and preprocess it
    image = Image.open(file_name)
    image = image.resize((180, 180))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)

    # use the model to make a prediction
    prediction = model.predict(image)
    
    #get score
    score = tf.nn.softmax(prediction[0])

    # define the list of class names
    class_names = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']
    
    # get the class index with the highest probability
    class_index = np.argmax(score)
    
    # get the class name corresponding to the class index
    class_name = class_names[class_index]
    
    # display the image and the prediction in the GUI
    image = ImageTk.PhotoImage(Image.open(file_name))
    label_image.config(image=image)
    label_image.image = image
    label_prediction.config(text=class_name)
    label_predictionArray.config(text=str(np.max(score) * 100))

# create the GUI
root = tk.Tk()
root.title("Image Classification")

entry = tk.Entry(root)
entry.pack()

button = tk.Button(root, text="Predict", command=load_predict)
button.pack()

label_image = tk.Label(root, text="Label Image")
label_image.pack()

label_prediction = tk.Label(root, text="Label Prediction")
label_prediction.pack()

label_predictionArray = tk.Label(root, text="Label Prediction Score")
label_predictionArray.pack()

root.mainloop()
