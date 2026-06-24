from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load("house_model.joblib")
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():

    area = float(request.form['area'])
    bedrooms = int(request.form['bedrooms'])
    bathrooms = int(request.form['bathrooms'])
    age = int(request.form['age'])

    data = np.array(
        [[area, bedrooms, bathrooms, age]]
    )

    prediction = model.predict(data)[0]

    return render_template(
        'index.html',
        prediction=round(prediction, 2)
    )


if __name__ == '__main__':
    app.run(debug=True)