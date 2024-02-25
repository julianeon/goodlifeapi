from flask import Flask, request, jsonify
from flask_cors import CORS
import random

app = Flask(__name__)
CORS(app) 

@app.route('/question', methods=['POST'])
def handle_question():
    data = request.get_json()
    question = data.get('question', '')
    responses = [
        "crime is low.",
        "rate is 5%.",
        "quality of living is high."
    ]
    canswer = random.choice(responses)
    ranswer = question + " " + canswer
    return jsonify(answer=ranswer)

if __name__ == '__main__':
    app.run(debug=True)
