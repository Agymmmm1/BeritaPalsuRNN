from flask import Flask, render_template, request, jsonify
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from joblib import load
from flask_cors import CORS
from bs4 import BeautifulSoup
import requests

app = Flask(__name__)
CORS(app)

# Load the tokenizer using joblib
tokenizer = load('tokenizer.joblib')

# Load the model configuration and weights
with open('model_config.json', 'r') as json_file:
    model_config = json_file.read()

model = tf.keras.models.model_from_json(model_config)
model.load_weights('model_weights.h5')

# Function to preprocess input text
def preprocess_text(text):
    # Tokenize text (use the loaded tokenizer)
    sequence = tokenizer.texts_to_sequences([text])
    # Pad sequence
    padded_sequence = pad_sequences(sequence, padding='post', maxlen=256)
    return padded_sequence

# Function to get prediction
def get_prediction(text):
    preprocessed_text = preprocess_text(text)
    prediction = model.predict(preprocessed_text)[0][0]
    fake_prob = prediction * 100
    real_prob = (1 - prediction) * 100
    result_text = 'Hoax' if fake_prob > real_prob else 'Fakta'
    return fake_prob, real_prob, result_text

# Function to extract text from URL (if needed)
def extract_text_from_url(link):
    response = requests.get(link)
    soup = BeautifulSoup(response.content, 'html.parser')
    paragraphs = soup.find_all('p')
    text = ' '.join([para.get_text() for para in paragraphs])
    return text

# Route for home page
@app.route('/')
def home():
    return render_template('index.html')

# Route to handle form submission and display results
@app.route('/result', methods=['POST'])
def result():
    if request.method == 'POST':
        message = request.form.get('message')
        link = request.form.get('link')
        
        if link:
            message = extract_text_from_url(link)
        
        if not message:
            return jsonify({'error': 'No message or URL provided'}), 400
        
        fake_prob, real_prob, result_text = get_prediction(message)
        return jsonify({
            'text': message,
            'prediction': result_text,
            'fake_prob': fake_prob,
            'real_prob': real_prob
        })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True, debug=True)
