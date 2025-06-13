from flask import Flask, render_template, request, flash, redirect, url_for
import sqlite3
import pickle
import numpy as np

from twilio.rest import Client
account_sid = "XXXXXXXXXXXXXXXXXXXX"
auth_token = "XXXXXXXXXXXXXXXXXXXX"
client = Client(account_sid, auth_token)


import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img,img_to_array
from keras.preprocessing.image import img_to_array
import pickle
from flask import Flask, render_template, url_for, request
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import load_img
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array, array_to_img
from keras.preprocessing import image
import sqlite3
import shutil

app = Flask(__name__)
import pickle
rfc=pickle.load(open("new.pkl","rb"))

def remove_first_decimal(value):
    # Split the number by the decimal point
    parts = value.split('.')

    # Check if there are more than 2 parts (more than one decimal point)
    if len(parts) > 0:
        # Reconstruct the number without the second decimal value
        return parts[0] 
    else:
        # If there's only one or two parts (no second decimal value), return the original value
        return value
    
def remove_second_decimal(value):
    # Split the number by the decimal point
    parts = value.split('.')

    # Check if there are more than 2 parts (more than one decimal point)
    if len(parts) > 2:
        # Reconstruct the number without the second decimal value
        return parts[0] + '.' + parts[1]
    else:
        # If there's only one or two parts (no second decimal value), return the original value
        return value

def adjust_ph(ph):
    ph = float(ph)
    # Check if pH is greater than 14
    if ph > 14:
        # Divide the pH by 2
        ph = ph / 2
    return ph

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/home')
def home():
    return redirect(url_for('predictPage'))

@app.route("/predictPage")
def predictPage():
    from serial_test import Read
    dat=Read()
    print(dat)
    tur_r=dat[0]
    con_r=dat[1]
    ph_r=dat[2]
    temp_r=dat[3]
    print(f"pH : {ph_r} \ntemperature : {temp_r} \n turbidity: {tur_r} \n conductivity : {con_r}")
    return render_template('userlog.html',ph_r=ph_r,temp_r=temp_r,tur_r=tur_r,con_r=con_r)

@app.route('/signin', methods=['GET', 'POST'])
def signin():
    if request.method == 'POST':

        connection = sqlite3.connect('database.db')
        cursor = connection.cursor()

        phone = request.form['phone']
        password = request.form['password']

        query = "SELECT * FROM user WHERE mobile = '"+phone+"' AND password= '"+password+"'"
        cursor.execute(query)

        result = cursor.fetchone()

        if result:
            return redirect(url_for('predictPage'))
        else:
            return render_template('signin.html', msg='Sorry, Incorrect Credentials Provided,  Try Again')

    return render_template('signin.html')


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':

        connection = sqlite3.connect('database.db')
        cursor = connection.cursor()

        name = request.form['name']
        password = request.form['password']
        mobile = request.form['phone']
        email = request.form['email']
        
        print(name, mobile, email, password)

        cursor.execute("INSERT INTO user VALUES ('"+name+"', '"+password+"', '"+mobile+"', '"+email+"')")
        connection.commit()

        return render_template('signin.html', msg='Successfully Registered')
    
    return render_template('signup.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
     
    if request.method == 'POST':
        Conductivity = request.form['Conductivity']
        Turbidity = request.form['Turbidity']
        Temperature = request.form['Temperature']
        Ph = request.form['Ph']
        
        data = np.array([[Turbidity,Conductivity, Temperature, Ph]])
        my_prediction = rfc.predict(data)
        result = my_prediction[0]
        aa=[]
        print(result)

        tur, cond, temp, p = " ", " ", " ", " "
        if result == 1 :
            res='in Good Condition '

        elif result == 0: 
            res='in Bad Condition'

            if float(Temperature) < 25:
                temp = "Low Temperature. Ideal temperature range is between 25 to 35 degrees Celsius."
                print(f"\n\n{temp}\n\n")
            elif float(Temperature) > 25 and float(Temperature)  < 100:
                temp = "High Temperature. Ideal temperature range is between 25 to 35 degrees Celsius."
                print(f"\n\n{temp}\n\n")
            if float(Ph) < 6:
                p = "Low pH. Ideal pH range is between 6 to 9."
                print(f"\n\n{p}\n\n")
            elif float(Ph) > 6 and float(Ph) <= 14:
                p = "High pH. Ideal pH range is between 6 to 9."
                print(f"\n\n{p}\n\n")
            if  float(Turbidity) >= 0 and float(Turbidity) < 1025:
                tur="Water is murky"
                print(f"\n\n{tur}\n\n")
            if float(Conductivity) >= 0 and float(Conductivity) < 1025:
                cond="Current is Passing"
                print(f"\n\n{cond}\n\n")
            print("Turbidity {} \n  Conductivity {} \n Temperature {} \n Ph {} \n".format(tur,cond,temp,p))
        
        print(res)
        return render_template('userlog.html', status=res,tur_r=Turbidity,con_r=Conductivity,temp_r=Temperature,ph_r=Ph,
                                               ph=p,temp=temp,tur=tur,con=cond)
    from serial_test import Read
    dat=Read()
    tur_r=dat[0]
    con_r=dat[1]
    ph_r=dat[2]
    temp_r=dat[3]
    print(f"pH : {ph_r} \ntemperature : {temp_r} \n turbidity: {tur_r} \n conductivity : {con_r}")   
    return render_template('userlog.html',ph_r=ph_r,temp_r=temp_r,tur_r=tur_r,con_r=con_r)

@app.route('/analyse', methods=['GET', 'POST'])
def analyse():
    if request.method == 'POST':

        dirPath = "static/testimage"
        fileList = os.listdir(dirPath)
        for fileName in fileList:
            os.remove(dirPath + "/" + fileName)
        fileName=request.form['File']
        dst = "static/testimage"
        
        shutil.copy("test/"+fileName, dst)
        image = cv2.imread("test/"+fileName)
        
        #color conversion
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cv2.imwrite('static/gray.jpg', gray_image)
        #apply the Canny edge detection
        edges = cv2.Canny(image, 250, 254)
        cv2.imwrite('static/edges.jpg', edges)
        #apply thresholding to segment the image
        _, threshold2 = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY)
        cv2.imwrite('static/threshold.jpg', threshold2)

        # Create a black color mask (0 represents black in grayscale images)
        black_mask = cv2.inRange(threshold2, 0, 0)

        # Find contours of the mask
        contours, _ = cv2.findContours(black_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Calculate the total area of black regions
        total_area = sum(cv2.contourArea(contour) for contour in contours)

        # Define size grades based on area thresholds (example thresholds)
        if total_area < 10000:
            size_grade = "Small"
        elif 10000 <= total_area < 20000:
            size_grade = "Medium"
        else:
            size_grade = "Large"
        print(f"\n\n\n SIZE GRADE IS :{size_grade} \n\n\n AREA  IS :{total_area} \n\n\n")
        
        model=load_model('FISHH.h5')
        path='static/testimage/'+fileName

        # Load the class names
        with open('class_names.pkl', 'rb') as f:
            class_names = pickle.load(f)
        dec=""
        dec1=""
        # Function to preprocess the input image
        def preprocess_input_image(path):
            img = load_img(path, target_size=(150,150))
            img_array = img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array /= 255.0  # Normalize the image
            return img_array

        # Function to make predictions on a single image
        def predict_single_image(path):
            input_image = preprocess_input_image(path)
            prediction = model.predict(input_image)
            print(prediction)
            predicted_class_index = np.argmax(prediction)
            predicted_class = class_names[predicted_class_index]
            confidence = prediction[0][predicted_class_index]

            print(f"Predicted Class: {predicted_class}")
            print(f"Confidence: {confidence:.2%}")
                
            return predicted_class, confidence 

        predicted_class, confidence = predict_single_image(path)
        #predicted_class, confidence = predict_single_image(path, model, class_names)
        print(predicted_class, confidence)
        
        str_label = predicted_class

        accuracy = f"The predicted image is {str_label} with a confidence of {confidence:.2%}"    


        return render_template('fish.html', status=str_label,accuracy=accuracy, ImageDisplay="http://127.0.0.1:5000/static/testimage/"+fileName,
        ImageDisplay1="http://127.0.0.1:5000/static/gray.jpg",
        ImageDisplay2="http://127.0.0.1:5000/static/edges.jpg",
        ImageDisplay3="http://127.0.0.1:5000/static/threshold.jpg",size_grade=size_grade)
    
    return render_template('fish.html')

@app.route('/developer')
def developer():
    return render_template('developer.html')

@app.route('/graph', methods=['GET', 'POST'])
def graph():
    
    images = ['http://127.0.0.1:5000/static/accuracy_plot.png',
              'http://127.0.0.1:5000/static/confusion_matrix.png']
    content=['Accuracy Graph',
             'Confusion Matrix']
        
    return render_template('graph.html',images=images,content=content)

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)