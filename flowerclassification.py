import tkinter as tk
from tkinter import filedialog as fd

import numpy as np
import os

import tensorflow as tf
from tensorflow.keras.models import load_model

from PIL import Image, ImageTk

# load the model
trained_model = load_model('model/model.h5')

# define class names
class_names = ['Daisy', 'Dandelion', 'Rose', 'Sunflower', 'Tulip']

filePath = ''
def select_file():
    global filePath
    # open file dialog
    filePath = fd.askopenfilename()
    fileName = os.path.basename(filePath)
    
    # write the file path on text label
    pathLabel.config(text = "File name: " + fileName)
    
    # display image in GUI
    image = Image.open(filePath)
    width, height = image.size
    aspect_ratio = width / height
    if aspect_ratio > 1:
        new_width = 200
        new_height = int(200 / aspect_ratio)
    else:
        new_width = int(200 * aspect_ratio)
        new_height = 200
    image = image.resize((new_width, new_height), Image.ANTIALIAS)
    image = ImageTk.PhotoImage(image)
    imageLabel.config(image=image, borderwidth=0, width=new_width, height=new_height)
    imageLabel.image = image
    
    # enable predict button
    button.config(state = "active")

def reset():
    global filePath
    # reset all widget to default state
    filePath = ''
    pathLabel.config(text = "File name: No image selected")
    imageLabel.config(image = None)
    imageLabel.image = None
    button.config(state = "disabled")
    predictionLabel.config(text = "Classification: not started")
    scoreLabel.config(text = "Prediction Score: not started")

def start_classification():
    # preprocess image
    img = tf.keras.utils.load_img(
        filePath, target_size=(180, 180))
    imageArray = tf.keras.utils.img_to_array(img)
    imageArray = tf.expand_dims(imageArray, 0) 

    # start prediction
    predictions = trained_model.predict(imageArray)
    
    # get highest score among classes
    score = tf.nn.softmax(predictions[0])

    # get the predicted class & score 
    predicted_class = class_names[np.argmax(score)]
    predicted_score = "{:.2f}".format(100 * np.max(score))

    # display the classification result
    predictionLabel.config(text = "Classification: " + predicted_class)
    scoreLabel.config(text=str("Prediction Score: " + predicted_score + '%'))
    

# GUI widgets
window = tk.Tk()
window.geometry("250x450")
window.resizable(False, False)
window.title("Flower Classification")

baseFrame = tk.Frame(window)
baseFrame.pack()

loadImageOperationFrame = tk.LabelFrame(baseFrame, text="Load Image")
loadImageOperationFrame.pack(fill = tk.BOTH)

loadImageFrame = tk.LabelFrame(baseFrame, text="Image")
loadImageFrame.pack(fill = tk.BOTH)

operationFrame = tk.LabelFrame(baseFrame, text="Result")
operationFrame.pack(fill = tk.BOTH)

pathLabel = tk.Label(loadImageOperationFrame, text = "File name: No image selected")
pathLabel.pack()

selectImageButton = tk.Button(loadImageOperationFrame, text="Select an Image", command = select_file)
selectImageButton.pack(fill = tk.BOTH, padx = 5, pady = 1)

resetButton = tk.Button(loadImageOperationFrame, text = "Reset", command=reset)
resetButton.pack(fill = tk.BOTH, padx = 5, pady = 2)

imageLabel = tk.Label(loadImageFrame, borderwidth = 100)
imageLabel.pack(fill = tk.BOTH, padx = 5, pady = 5)

predictionLabel = tk.Label(operationFrame, text = "Classification: not started")
predictionLabel.pack()

scoreLabel = tk.Label(operationFrame, text = "Prediction Score: not started")
scoreLabel.pack()

button = tk.Button(operationFrame, text="Predict", command = start_classification, state = "disabled")
button.pack(fill = tk.BOTH, padx = 5, pady = 5)

#main loop function
window.mainloop()
