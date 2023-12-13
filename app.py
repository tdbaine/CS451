from flask import Flask, render_template, request, jsonify
import joblib
import re

app = Flask(__name__)

# Load the trained model and vectorizer
model = joblib.load("sentiment_model2.joblib")
vectorizer = joblib.load("tfidf_vectorizer2.joblib")

@app.route('/')
def home():
    return render_template('webpage.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from the request
        input_data = request.json.get('data')

        # Check if input_data is a file path
        if input_data.endswith('.txt'):
            with open(input_data, 'r') as file:
                reviews = file.read().split('%%')
        else:
            # Assume input_data is a string with reviews separated by %%
            reviews = input_data.split('%%')

        # Preprocess the input data
        processed_reviews = [re.sub(r'[^a-zA-Z\s]', '', review.lower()) for review in reviews]

        # Vectorize the input data
        data_vec = vectorizer.transform(processed_reviews)

        # Make predictions for each review
        predictions = model.predict(data_vec)

        # Return the predictions as a list
        return jsonify({'predictions': predictions.tolist()})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
