from flask import Flask, render_template,request
import pickle
import numpy as np

model = pickle.load(open('lr.pkl', 'rb'))
sc = pickle.load(open('scaler.pkl', 'rb'))
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods = ['POST'])
def predict_placement():
    CRIM = float(request.form.get('crim'))
    ZN = float(request.form.get('zn'))
    INDUS  = request.form.get('indus')
    CHAS = request.form.get('chas')
    NOX = request.form.get('nox')
    RM = request.form.get('rm')
    AGE = request.form.get('age')
    DIS = request.form.get('dis')
    RAD = request.form.get('rad')
    TAX = request.form.get('tax')
    PTRATIO = request.form.get('ptratio')
    B = request.form.get('b')
    LSTAT = request.form.get('lstat')

    # prediction
    result = np.round(model.predict(sc.transform(np.array([CRIM, ZN, INDUS, CHAS, NOX, RM, AGE, DIS, RAD, TAX, PTRATIO,B, LSTAT]).reshape(1,13))),2)
    result =  {"The prediction of house is ":str(result[0][0])}

    return render_template('index.html', result=result)


if __name__ == "__main__":
    app.run(debug=True)