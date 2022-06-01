import numpy as np
from flask import Flask, request, render_template  
import pickle
app = Flask(__name__,template_folder='template')# Load the model
model = pickle.load(open('model.pk1','rb'))
@app.route("/")
def man():
    return render_template('template1.html')
@app.route('/predict',methods=['POST'])
def home():
    # Get the data from the POST request
    data1= request.form['a']
    data2= request.form['b']
    data3= request.form['c']
    data4= request.form['d']
    data5= request.form['e']
    data6= request.form['f']
    data7= request.form['g']
    input_data=(data1, data2, data3 ,data4, data5 ,data6 ,data7)
    arr=np.asarray(input_data)
    input_data_reshaped=arr.reshape(1,-1)
    pred= model.predict(input_data_reshaped)
    ans=pred[0]
    return render_template('template.html',data=ans)
if __name__ == '__main__':
    app.run(port=5000, debug=True)