from flask import Flask, request, jsonify
from flask_cors import CORS
import random

app = Flask(__name__)
CORS(app) 

@app.route('/question', methods=['POST'])
def handle_question():
    responses = [
        "Crime is low.",
        "Rate is 5%.",
        "Quality of living is high."
    ]
    return jsonify(answer=random.choice(responses))

if __name__ == '__main__':
    app.run(debug=True)
