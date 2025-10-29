import os
import re
import joblib
import pandas as pd
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# --- PART 1: MODEL TRAINING LOGIC ---

MODEL_FILE = 'sentiment_model.joblib'
VECTORIZER_FILE = 'tfidf_vectorizer.joblib'
DATA_FILE = 'Hindi reviews.xlsx' # Make sure this file is in the same folder

def clean_text(text):
    """Cleans the input text."""
    if pd.isna(text):
        return ""
    s = str(text).lower().strip()
    s = re.sub(r'http\S+|www\.\S+', '', s)
    s = re.sub(r'[^ऀ-ॿa-zA-Z\s]', '', s)  # keep only Hindi and English chars
    s = re.sub(r'\s+', ' ', s).strip()
    tokens = s.split()
    return " ".join(tokens)

def train_model():
    """Trains and saves the model and vectorizer."""
    print("Training new model...")
    
    # Load the file
    try:
        df = pd.read_excel(DATA_FILE)
    except FileNotFoundError:
        print(f"Error: Data file '{DATA_FILE}' not found.")
        print("Please place the Excel file in the same directory as this script.")
        return False
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        return False

    # Clean text
    df['Review'] = df['Review'].apply(clean_text)
    
    # Drop rows where 'Review' or 'Label' might be missing after cleaning
    df.dropna(subset=['Review', 'Label'], inplace=True)
    df = df[df['Review'].str.strip() != '']

    X = df['Review']
    y = df['Label']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Vectorize
    tfidf = TfidfVectorizer(max_features=5000)
    X_train_vec = tfidf.fit_transform(X_train)
    # X_test_vec = tfidf.transform(X_test) # Not needed for saving, but good for testing

    # Train with GridSearchCV
    param_grid = {
        'C': [0.1, 1, 10],
        'penalty': ['l2'],
        'solver': ['lbfgs', 'liblinear']
    }
    lr = LogisticRegression(max_iter=1000)
    grid = GridSearchCV(lr, param_grid, cv=5, scoring='accuracy')
    grid.fit(X_train_vec, y_train)

    best_lr = grid.best_estimator_
    print(f"Model trained. Best Params: {grid.best_params_}")

    # Save the vectorizer and the model
    joblib.dump(tfidf, VECTORIZER_FILE)
    joblib.dump(best_lr, MODEL_FILE)
    print(f"Model saved as '{MODEL_FILE}'")
    print(f"Vectorizer saved as '{VECTORIZER_FILE}'")
    return True

# --- PART 2: FRONTEND HTML ---

# Store the entire HTML as a Python string
HTML_CONTENT = """
<!DOCTYPE html>
<html lang="hi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Hindi Sentiment Analysis</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
            background-color: #f4f7f6;
            color: #333;
            margin: 0;
        }
        .container {
            background: #ffffff;
            padding: 2rem;
            border-radius: 12px;
            box-shadow: 0 8px 20px rgba(0, 0, 0, 0.07);
            width: 90%;
            max-width: 500px;
            text-align: center;
        }
        h1 {
            color: #1a1a1a;
            margin-bottom: 1.5rem;
        }
        textarea {
            width: 100%;
            min-height: 100px;
            padding: 0.75rem;
            border: 1px solid #ddd;
            border-radius: 8px;
            font-size: 1rem;
            box-sizing: border-box; /* Important */
            margin-bottom: 1rem;
            resize: vertical;
        }
        button {
            background-color: #007bff;
            color: white;
            border: none;
            padding: 0.75rem 1.5rem;
            font-size: 1rem;
            font-weight: 600;
            border-radius: 8px;
            cursor: pointer;
            transition: background-color 0.2s;
        }
        button:hover {
            background-color: #0056b3;
        }
        #result {
            margin-top: 1.5rem;
            font-size: 1.2rem;
            font-weight: 600;
            padding: 1rem;
            border-radius: 8px;
        }
        #result.positive {
            color: #28a745;
            background-color: #e9f7ec;
        }
        #result.negative {
            color: #dc3545;
            background-color: #fdeaea;
        }
        .loader {
            display: none;
            margin-top: 1rem;
            border: 4px solid #f3f3f3;
            border-top: 4px solid #007bff;
            border-radius: 50%;
            width: 30px;
            height: 30px;
            animation: spin 1s linear infinite;
            margin-left: auto;
            margin-right: auto;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>हिंदी सेंटीमेंट विश्लेषक</h1>
        <p>अपनी समीक्षा (review) को नीचे टेक्स्ट बॉक्स में पेस्ट करें:</p>
        
        <textarea id="reviewInput" rows="5" placeholder="जैसे: फिल्म बिल्कुल बेकार थी..."></textarea>
        
        <button id="analyzeButton">Analyze</button>
        
        <div class="loader" id="loader"></div>
        <div id="result"></div>
    </div>

    <script>
        const analyzeButton = document.getElementById('analyzeButton');
        const reviewInput = document.getElementById('reviewInput');
        const resultDiv = document.getElementById('result');
        const loader = document.getElementById('loader');

        // Calls the '/predict' endpoint on the *same server*
        const API_URL = '/predict'; 

        analyzeButton.addEventListener('click', async () => {
            const reviewText = reviewInput.value;
            if (reviewText.trim() === '') {
                alert('Please enter a review.');
                return;
            }

            loader.style.display = 'block';
            resultDiv.style.display = 'none';
            resultDiv.className = ''; 

            try {
                const response = await fetch(API_URL, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ review: reviewText })
                });

                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }

                const data = await response.json();
                const sentiment = data.sentiment;
                const confidence = (data.confidence * 100).toFixed(1);

                resultDiv.textContent = `${sentiment} (${confidence}%)`;
                resultDiv.className = sentiment.toLowerCase(); // 'positive' or 'negative'

            } catch (error) {
                console.error('Error:', error);
                resultDiv.textContent = 'Error during analysis.';
                resultDiv.className = 'negative';
            } finally {
                loader.style.display = 'none';
                resultDiv.style.display = 'block';
            }
        });
    </script>
</body>
</html>
"""

# --- PART 3: BACKEND API (FLASK) ---

app = Flask(__name__)
CORS(app)  # Allows cross-origin requests

# Load models
try:
    # Check if models exist. If not, train them.
    if not os.path.exists(MODEL_FILE) or not os.path.exists(VECTORIZER_FILE):
        print("Model files not found.")
        if not train_model():
            # If training fails (e.g., no data file), exit.
            exit() 
            
    print("Loading pre-trained model and vectorizer...")
    tfidf = joblib.load(VECTORIZER_FILE)
    model = joblib.load(MODEL_FILE)
    print("Models loaded successfully.")
    
except Exception as e:
    print(f"Error loading models: {e}")
    print("Attempting to retrain...")
    if not train_model():
        print("Critical error: Could not train or load model. Exiting.")
        exit()
    # Try loading again after training
    tfidf = joblib.load(VECTORIZER_FILE)
    model = joblib.load(MODEL_FILE)
    print("Models loaded successfully after retraining.")


# API route for prediction
@app.route('/predict', methods=['POST'])
def predict_sentiment_api():
    try:
        data = request.json
        if 'review' not in data:
            return jsonify({'error': 'No review text provided'}), 400

        review = data['review']
        review_clean = clean_text(review)
        vec = tfidf.transform([review_clean])
        
        pred = model.predict(vec)[0]
        prob = model.predict_proba(vec)[0]
        confidence = max(prob)
        
        return jsonify({
            'sentiment': pred,
            'confidence': float(confidence)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Route to serve the HTML frontend
@app.route('/')
def home():
    return Response(HTML_CONTENT, mimetype='text/html')

# --- PART 4: RUN THE APP ---
if __name__ == '__main__':
    print("\n--- Starting Flask Server ---")
    print("Open your web browser and go to: http://127.0.0.1:5000")
    app.run(debug=False, port=5000) # Set debug=False for a cleaner console