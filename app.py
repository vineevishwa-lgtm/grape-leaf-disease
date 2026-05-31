from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

app = Flask(__name__)

# Load model
model = load_model("grape_model.h5")

classes = ["Early", "Healthy", "Moderate", "Severe"]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():

    file = request.files['image']

    img = Image.open(file).convert("RGB")
    img = img.resize((224, 224))

    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)

    class_index = np.argmax(prediction)
    confidence = np.max(prediction) * 100

    result = classes[class_index]

    return render_template(
        "index.html",
        prediction=result,
        confidence=round(confidence, 2)
    )

if __name__ == '__main__':
    app.run(debug=True)