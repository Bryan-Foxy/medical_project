import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

import tempfile
from src.vhs import VHS
from flask import Flask, request, jsonify
from config import HEART_MODEL_PATH, VERTEBRAE_MODEL_PATH

app = Flask(__name__)

@app.route('/vhs', methods=['POST'])
def vhs():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    with tempfile.NamedTemporaryFile(delete = False, suffix = ".jpg") as temp_file:
        file.save(temp_file.name)
        temp_path = temp_file.name
    Major, minor, vhs_score = VHS(image_path = temp_path, 
                                  model_heart = HEART_MODEL_PATH,
                                  model_vertebrae = VERTEBRAE_MODEL_PATH).perform_vhs()
    return jsonify({'Major': Major, 'minor': minor, 'vhs_score': vhs_score})

if __name__ == '__main__':
    app.run(debug = True)
