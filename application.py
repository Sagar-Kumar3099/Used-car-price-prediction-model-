from flask import Flask, render_template, request
from flask_cors import CORS, cross_origin
import pickle
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline

app = Flask(__name__)
cors = CORS(app)


model = pickle.load(open('LinearRegressionModel.pkl', 'rb'))
car = pd.read_csv('cleaned car.csv')

def check_pipeline(pipeline):
    if not isinstance(pipeline, Pipeline):
        raise ValueError("The loaded model is not a Pipeline instance.")
    print("Pipeline steps:")
    for name, step in pipeline.named_steps.items():
        print(f"{name}: {type(step)}")


check_pipeline(model)


@app.route('/', methods=['GET', 'POST'])
def index():
    try:
        companies = sorted(car['company'].unique())
        car_models = sorted(car['name'].unique())
        years = sorted(car['year'].unique(), reverse=True)
        fuel_types = car['fuel_type'].unique()

        companies.insert(0, 'Select Company')
        return render_template('index.html', companies=companies, car_models=car_models, years=years,
                               fuel_types=fuel_types)
    except Exception as e:
        print(f"Error in index route: {e}")
        return "Error loading the index page."


@app.route('/predict', methods=['POST'])
@cross_origin()
def predict():
    try:
        company = request.form.get('company')
        car_model = request.form.get('car_models')
        year = int(request.form.get('year'))
        fuel_type = request.form.get('fuel_type')
        driven = int(request.form.get('kilo_driven'))

        input_data = pd.DataFrame(
            columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'],
            data=np.array([car_model, company, year, driven, fuel_type]).reshape(1, 5)
        )

        print("Input Data:\n", input_data)

        # Ensure model is a Pipeline
        check_pipeline(model)

        # Predict the price
        prediction = model.predict(input_data)
        print("Prediction:", prediction)

        return str(np.round(prediction[0], 2))
    except ValueError as ve:
        print("ValueError during prediction:", ve)
        return "Invalid input. Please check your input values and try again."
    except Exception as e:
        print("Error during prediction:", e)
        return "Error in prediction. Please check input values and try again."


if __name__ == '__main__':
    app.run(debug=True)
