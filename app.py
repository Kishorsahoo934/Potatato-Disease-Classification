from flask import Flask, request
from flask_restful import Resource, Api
from flask_cors import CORS
from PIL import Image
import tensorflow as tf
import numpy as np
import io

# ---------------------- Flask App ----------------------
app = Flask(__name__)
CORS(app)  # Allow cross-origin requests
api = Api(app)

# ---------------------- Load Model ----------------------
model = tf.keras.models.load_model("model.h5")
class_names = ['Early Blight', 'Late Blight', 'Healthy']  # Update based on your model

# ---------------------- Helper Function ----------------------
def preprocess_image(img, target_size=(256, 256)):
    image = img.convert("RGB")
    image = image.resize(target_size)
    img_array = tf.keras.preprocessing.image.img_to_array(image)
    img_array = img_array / 255.0
    return np.expand_dims(img_array, axis=0)

# ---------------------- Prediction API ----------------------
class Predict(Resource):
    def post(self):
        if 'file' not in request.files:
            return {"error": "No file uploaded"}, 400
        
        file = request.files['file']
        try:
            img = Image.open(file.stream)
            img_array = preprocess_image(img)
            prediction = model.predict(img_array)
            predicted_index = int(np.argmax(prediction))
            predicted_class = class_names[predicted_index]
            confidence = float(np.max(prediction) * 100)
            return {
                "prediction": predicted_class,
                "confidence": f"{confidence:.2f}%"
            }
        except Exception as e:
            return {"error": str(e)}, 500

# ---------------------- Add Resource ----------------------
api.add_resource(Predict, '/predict')

# ---------------------- Run App ----------------------
if __name__ == "__main__":
    app.run(debug=True)
        
