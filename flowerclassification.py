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
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)

    # use the model to make a prediction
    prediction = model.predict(image)

    # display the image and the prediction in the GUI
    image = ImageTk.PhotoImage(Image.open(file_name))
    label_image.config(image=image)
    label_image.image = image
    label_prediction.config(text=str(prediction))

# create the GUI
root = tk.Tk()
root.title("Image Classification")

entry = tk.Entry(root)
entry.pack()

button = tk.Button(root, text="Predict", command=load_predict)
button.pack()

label_image = tk.Label(root, text="")
label_image.pack()

label_prediction = tk.Label(root, text="")
label_prediction.pack()

root.mainloop()
