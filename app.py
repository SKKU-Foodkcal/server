from flask import Flask, request, render_template
import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.image as img
from tensorflow.keras.preprocessing import image
from tensorflow import keras

UPLOAD_FOLDER = '/Users/juyeongjun/PycharmProjects/server/uploads'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])
#food_list = ['apple_pie','omellete','pizza' ]
#food_list = ['apple_pie','breakfast_burrito','carrot_cake','chicken_curry','chicken_wings','chocolate_cake','churros','cup_cakes','donuts','fish_and_chips','french_fries','bibimbap','cheesecake','french_toast','hamburger','pho','steak','sushi','pizza','ramen']
food_list = ['apple_pie', 'bibimbap', 'breakfast_burrito', 'carrot_cake', 'cheesecake', 'chicken_curry', 'chicken_wings', 'chocolate_cake', 'churros', 'cup_cakes', 'donuts', 'fish_and_chips', 'french_fries', 'french_toast', 'hamburger', 'pho', 'pizza', 'ramen', 'steak', 'sushi']

model_dir = "/Users/juyeongjun/PycharmProjects/server/"
#new_model = tf.saved_model.load(model_dir)
model = keras.models.load_model("best_model_3class.hdf5")

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/")
def hello():
    return "Hello World!"

@app.route("/test", methods=['GET','POST'])
def test():
    print(request.files['images'])
    #print (request.json)
    a = request.json
    print(a["email"])
    return a["email"]

@app.route('/prediction',methods = ['POST'])
def make_prediction():
    if request.method == 'POST':
        if 'images' not in request.files:
            return 'No file'
        file = request.files['images']
        if file.filename == '':
            return 'No file'
        if file:
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))
        print(file.filename)
        print(request)
        file_path = '/Users/juyeongjun/PycharmProjects/server/uploads/' + file.filename
        img = image.load_img(file_path , target_size=(299, 299))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img /= 255.
        pred = model.predict(img)
        index = np.argmax(pred[0])

    return food_list[index]

@app.route('/testing',methods = ['GET','POST'])
def testing():
    test_model = keras.models.load_model("best_model_3class.hdf5")
    file_path = '/Users/juyeongjun/PycharmProjects/server/uploads/hamburger.jpeg'
    img = image.load_img(file_path, target_size=(299, 299))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img /= 255.
    test_model.summary()
    pred = test_model.predict(img)
    print(pred)
    re_test = np.argmax(pred)
    print(re_test)
    result = np.argmax(pred[0])
    print(result)
    print(food_list[result])
    #food_list_sorted = sorted(food_list)
    #print(food_list_sorted[result])
    return food_list[result]

if __name__ == "__main__":
    app.run(host = '0.0.0.0', port = 5000, debug= True)