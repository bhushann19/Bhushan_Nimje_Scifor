from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd

# Initialize Flask application
app = Flask(__name__)

# Load the model
try:
    with open('linear_regression_tips_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
except FileNotFoundError:
    raise Exception("Model file not found. Please check the path.")
except Exception as e:
    raise Exception(f"Error loading model: {e}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from request
        data = request.get_json(force=True)
        
        # Convert data to DataFrame
        df = pd.DataFrame(data)
        
        # Predict using the loaded model
        predictions = model.predict(df)
        
        # Return predictions as JSON
        response = {'predictions': predictions.tolist()}
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
