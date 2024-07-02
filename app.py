from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

# Load the saved model
best_model = load_model("best_model.h5")

# Define a route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Define a route for making predictions on uploaded images
@app.route('/predict', methods=['POST'])
def predict():
    # Get the image file from the request
    file = request.files['file']

    # Read the image file
    img = Image.open(io.BytesIO(file.read()))

    # Preprocess the image if necessary (resize, normalize, etc.)
    # For example, you can resize the image to the required input size of the model

    # Convert the image to a numpy array
    img_array = np.array(img)

    # Perform prediction using the loaded model
    prediction = best_model.predict(np.expand_dims(img_array, axis=0))

    # Decode the prediction result if necessary
    # For example, if it's a binary classification, you might want to return a class label
    # If it's a multiclass classification, you might want to return the predicted class probabilities

    # Return the prediction result as JSON
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=True)
