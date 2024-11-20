import pickle
from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
# Load the pre-trained pipeline
f = open("pipeline.pkl", "rb") 
pipeline = pickle.load(f)

# Initialize Flask application
app = Flask(__name__)

@app.route('/')
def home():
    # This will render the home.html page when visiting the root URL
    return render_template('home.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            # Get input data from the form (request.form returns data as strings)
            data = request.form.to_dict()

            # Extract features and convert to appropriate types
            age = int(data['age'])
            sex = data['sex'].lower()
            bmi = float(data['bmi'])
            children = int(data['children'])
            smoker = data['smoker'].lower() 
            region = data['region'].lower()
            
            # Prepare input data as a DataFrame (since pipeline expects a DataFrame for prediction)
            input_data = pd.DataFrame([[age, sex, bmi, children, smoker, region]], 
                                      columns=['age', 'sex', 'bmi', 'children', 'smoker', 'region'])

            # Use the trained pipeline to make a prediction
            prediction = pipeline.predict(input_data)
            
            # Return the result as a JSON response (which will be handled by JavaScript)
            return jsonify(prediction=prediction[0])

        except Exception as e:
            # Handle any errors (e.g., invalid input data) gracefully
            return jsonify(error=f"Error in processing the input: {str(e)}")

    else:
        # For GET request, simply render the prediction page
        return render_template('predict.html')

if __name__ == "__main__":
    app.run(debug=True)
