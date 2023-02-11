import tkinter as tk
from tkinter import filedialog as fd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image, ImageTk


def select_file():
    # open file dialog
    filePath = fd.askopenfilename()
    
    # write the file path on text field
    pathTextField.delete(0, tk.END)
    pathTextField.insert(0, filePath)


def start_classification():
    # load the model
    trained_model = load_model('model.h5')

    # get the file name from the GUI
    fileName = pathTextField.get()

    # load the image and preprocess it
    img = tf.keras.utils.load_img(
        fileName, target_size=(180, 180))
    imageArray = tf.keras.utils.img_to_array(img)
    imageArray = tf.expand_dims(imageArray, 0) 

    # start prediction
    predictions = trained_model.predict(imageArray)
    
    # get highest score among classes
    score = tf.nn.softmax(predictions[0])

    # define class names
    class_names = ['Daisy', 'Dandelion', 'Rose', 'Sunflower', 'Tulip']

    # get the predicted class & score 
    predicted_class = class_names[np.argmax(score)]
    predicted_score = "{:.2f}".format(100 * np.max(score))

    # display the image output
    image = ImageTk.PhotoImage(Image.open(fileName))
    label_image.config(image=image)
    label_image.image = image
    label_prediction.config(text=predicted_class)
    label_predictionArray.config(text=str(predicted_score + '%'))

# GUI widgets definition
root = tk.Tk()
root.title("Flower Classification")

pathTextField = tk.Entry(root)
pathTextField.pack()

button_file = tk.Button(root, text="Select an Image", command=select_file)
button_file.pack()

button = tk.Button(root, text="Predict", command=start_classification)
button.pack()

label_image = tk.Label(root, text="Label Image")
label_image.pack()

label_prediction = tk.Label(root, text="Label Prediction")
label_prediction.pack()

label_predictionArray = tk.Label(root, text="Label Prediction Score")
label_predictionArray.pack()

#main loop function
root.mainloop()
