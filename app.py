from flask import Flask, request, jsonify
import pickle  # or appropriate library for your model
import numpy as np

app = Flask(__name__)

# Load your model (adjust based on your model type)
model = pickle.load(open('propert_comp.pkl', 'rb'))

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    prediction = model.predict([np.array(data['features'])])
    return jsonify(prediction=prediction.tolist())

@app.route('/', methods=['GET'])
def home():
    return "ML Model API is running!"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))