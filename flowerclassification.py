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
    img = tf.keras.utils.load_img(
    file_name, target_size=(180, 180)
    )
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    class_names = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']
    
    predicted_class = class_names[np.argmax(score)]
    predicted_score = "{:.2f}".format(100 * np.max(score))

    # display the image and the prediction in the GUI
    image = ImageTk.PhotoImage(Image.open(file_name))
    label_image.config(image=image)
    label_image.image = image
    label_prediction.config(text=predicted_class)
    label_predictionArray.config(text=str(predicted_score + '%'))

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
