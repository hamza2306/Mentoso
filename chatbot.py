from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

app = Flask(__name__, template_folder='D:/Mentoso Application/finexo-html')
# Load the trained model
clf = joblib.load('depression_model.pkl')

# Define the questions
questions = [
    "How often do you feel sad or hopeless?",
    "Do you often experience mood swings?",
    "Do you have trouble falling asleep or staying asleep?",
    "Do you find yourself sleeping too much or too little?",
    "Have you experienced significant changes in your appetite recently?",
    "Have you noticed any significant weight gain or loss without trying?",
    "How would you rate your energy levels on a typical day?",
    "Do you often feel fatigued or low on energy?",
    "Do you have difficulty concentrating or making decisions?",
    "Have others noticed changes in your ability to focus or remember things?",
    "Do you often feel restless or have trouble sitting still?",
    "Have you lost interest in activities that you used to enjoy?",
    "Do you often feel worthless or guilty?",
    "Have you had thoughts of self-harm or suicide?"
]

# Define the mapping from text answers to numerical values
answer_mapping = {
    "1": 1,  # Never
    "2": 4,  # Rarely
    "3": 5,  # Sometimes
    "4": 3,  # Often
    "5": 2,  # Always
    "6": 6,  # Not at all
    "7": 4,  # Slightly
    "8": 5,  # Moderately
    "9": 3,  # Severely
    "10": 2, # Extremely
    "11": 1, # Extremely high
    "12": 2, # High
    "13": 3, # Moderate
    "14": 4, # Low
    "15": 5  # Extremely low
}

@app.route('/')
def chatbot():
    return render_template('chatbot.html', questions=questions)

@app.route('/submit', methods=['POST'])
def submit():
    responses = request.json['responses']
    input_data = np.array(responses).reshape(1, -1)
    if input_data.shape[1] < 15:
        input_data = np.append(input_data, 0).reshape(1, -1)
    prediction = clf.predict(input_data)
    return jsonify({'prediction': prediction[0]})

if __name__ == "__main__":
    app.run(debug=True, port=5503)