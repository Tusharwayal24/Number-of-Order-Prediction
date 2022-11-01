from flask import Flask, render_template,request
import jsonify
import requests
import pickle
import numpy as np
import sklearn

model=pickle.load(open('model_lgbm.pkl', 'rb'))

app=Flask(__name__)
@app.route('/')
def index():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
def predict_order():
    Discount=eval(request.form.get('Discount'))
    Store_Type=eval(request.form.get['Store_Type'])
    Location_Type = eval(request.form.get['Location_Type'])
    Holiday = int(request.form.get['Holiday'])

    result=model.predict([[Discount,Store_Type,Location_Type,Holiday]])
    output=round(result)

    return render_template('index.html')


if __name__=="__main__":
    app.run(debug=True)
