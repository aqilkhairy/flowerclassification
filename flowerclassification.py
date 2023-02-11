import tkinter as tk
from tkinter import filedialog as fd

import numpy as np

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
    
    # write the file path on text label
    pathLabel.config(text = "Image Path: " + filePath)
    
    # display image in GUI
    image = ImageTk.PhotoImage(Image.open(filePath))
    imageLabel.config(image=image, width = 200, height = 200)
    imageLabel.image = image
    
    # enable predict button
    button.config(state = "active")

def reset():
    global filePath
    # reset all widget to default state
    filePath = ''
    pathLabel.config(text = "No image selected")
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
window.geometry("500x500")
window.resizable(False, False)
window.title("Flower Classification")

baseFrame = tk.Frame(window)
baseFrame.pack()

loadImageOperationFrame = tk.LabelFrame(baseFrame, text="Load Image")
loadImageOperationFrame.pack(fill = tk.BOTH)

loadImageFrame = tk.LabelFrame(baseFrame, text="Image")
loadImageFrame.pack(fill = tk.BOTH)

operationFrame = tk.LabelFrame(baseFrame, text="Classification")
operationFrame.pack(fill = tk.BOTH)

pathLabel = tk.Label(loadImageOperationFrame, text = "No image selected")
pathLabel.grid(row = 0, column = 0)

selectImageButton = tk.Button(loadImageOperationFrame, text="Select an Image", command = select_file)
selectImageButton.grid(row = 1, column = 0, sticky = "news", padx = 10, pady = 5)

resetButton = tk.Button(loadImageOperationFrame, text = "Reset", command=reset)
resetButton.grid(row = 1, column = 1, padx = 10, pady = 5)

imageLabel = tk.Label(loadImageFrame)
imageLabel.grid(row = 0, column = 0,padx = 5, pady = 5, sticky= tk.NSEW)

button = tk.Button(operationFrame, text="Predict", command = start_classification, state = "disabled")
button.grid(row = 0, column = 0, sticky = "news", padx = 10, pady = 5)

predictionLabel = tk.Label(operationFrame, text = "Classification: not started")
predictionLabel.grid(row = 1, column = 0)

scoreLabel = tk.Label(operationFrame, text = "Prediction Score: not started")
scoreLabel.grid(row = 2, column = 0)

#main loop function
window.mainloop()
