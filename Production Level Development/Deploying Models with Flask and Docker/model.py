# Import necessary libraries and modules
from flask import Flask, request, jsonify, render_template  # Flask framework components for creating web applications
import pickle  # For loading and saving Python objects (e.g., models)
import pandas as pd  # For data manipulation and analysis, especially with DataFrames

# Initialize Flask application
app = Flask(__name__)  # Create a Flask app instance, which will act as our web server

# Load the model
try:
    with open('linear_regression_tips_model.pkl', 'rb') as model_file:  # Open the model file in binary read mode
        model = pickle.load(model_file)  # Load the model using pickle
except FileNotFoundError:
    raise Exception("Model file not found. Please check the path.")  # Raise an exception if the file is not found
except Exception as e:
    raise Exception(f"Error loading model: {e}")  # Raise a generic exception if any other error occurs during loading

# Define a route for the homepage
@app.route('/')  # Define the URL route for the root ('/') path
def index():
    return render_template('index.html')  # Render and return the 'index.html' template as the homepage

# Define a route for making predictions
@app.route('/predict', methods=['POST'])  # Define the URL route for '/predict' with POST method
# POST method is like sending information to a website so it can do something with it. 
# For example, when you fill out a form and click submit, you're using POST to send your data to the server.
def predict():
    try:
        # Get JSON data from the request
        data = request.get_json(force=True)  
        # Parse the incoming JSON data from the request
        #request: This is an object that represents the incoming web request.
        #get_json: This is a method (function) used to retrieve data in JSON format from the request.
        #force=True: This tells the method to convert the data to JSON format even if it’s not in the expected format. It forces the data to be treated as JSON.

        # Convert data to a DataFrame
        df = pd.DataFrame(data)  # Convert the JSON data into a Pandas DataFrame for easy manipulation

        # Predict using the loaded model
        predictions = model.predict(df)  # Use the loaded model to make predictions on the DataFrame

        # Return predictions as JSON
        response = {'predictions': predictions.tolist()}  # Convert predictions to a list and prepare JSON response
        return jsonify(response)  # Return the JSON response to the client

    except Exception as e:
        return jsonify({'error': str(e)}), 400  # If any error occurs, return a JSON error message with a 400 status code

# Run the application
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')  # Run the Flask app in debug mode, accessible on all network interfaces
