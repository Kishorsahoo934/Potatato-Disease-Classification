# 🥔 Potato Disease Classification 🌿

An intelligent web application built using **Streamlit** and **TensorFlow/Keras** to detect diseases in potato leaves using image classification.

---

## 🚀 Features

- 🌱 **Classifies** Potato Leaf Diseases:
  - **Early Blight**
  - **Late Blight**
  - **Healthy**
- 📸 Upload an image and get **real-time prediction**
- 📊 Shows confidence score for each class
- 🔥 Powered by **Convolutional Neural Networks (CNN)**
- 🧠 Uses a **pre-trained deep learning model (.h5)** for fast predictions

---



---

## 📂 Project Structure

📦 Potato-Disease-Classification
├── 📁 potato_model # Saved Keras Model (.h5)
├── 📁 dataset # Training Data (Not pushed to GitHub)
├── 📁 streamlit_app # Streamlit UI & app logic
│ ├── app.py # Main Streamlit code
├── .gitignore
├── README.md
├── requirements.txt


---

## ⚙️ How to Run

### 🧰 Step 1: Clone the Repository

```bash
git clone https://github.com/Kishorsahoo934/Potatato-Disease-Classification.git
cd Potatato-Disease-Classification
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
📌 Requirements
Python 3.7+

TensorFlow / Keras

Streamlit

NumPy, Pillow, etc.

All dependencies are listed in requirements.txt
