from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import json

# Initialize Flask application
app = Flask('houseprices', static_url_path='/static')

# Load the model
model = pickle.load(open('Mumbai_house_prices_model.pickle', 'rb'))

# Load the columns
with open('columns.json', 'r') as f:
    data_columns = json.load(f)['data_columns']

# Define routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    location = data['location']
    sqft = data['sqft']
    bhk = data['bhk']
    
    try:
        loc_index = data_columns.index(location.lower())
    except ValueError:
        return jsonify({'price': 0})
    
    x = np.zeros(len(data_columns))
    x[0] = bhk
    x[1] = sqft
    if loc_index >= 0:
        x[loc_index] = 1

    price = model.predict([x])[0]
    
    return jsonify({'price': price})

# Run the application
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)