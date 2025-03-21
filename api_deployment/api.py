import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)
import tempfile
from flask import Flask, request, jsonify, send_file
from src.vhs import VHS
from src.llm import Diagnostic
from config import Config


app = Flask(__name__)

@app.route('/vhs', methods=['POST'])
def vhs():
    """Processes an image and returns VHS measurements."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    original_filename = file.filename
    base_name, _ = os.path.splitext(original_filename)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        temp_file_path = f"{temp_file.name}_{base_name}.jpg"  
        file.save(temp_file_path)
        temp_path = temp_file_path

    major, minor, vhs_score, output_path = VHS(
        image_path=temp_path,
        model_heart=Config.HEART_MODEL_PATH,
        model_vertebrae=Config.VERTEBRAE_MODEL_PATH
    ).perform_vhs()

    return jsonify({
        'major': major,
        'minor': minor,
        'vhs_score': vhs_score,
        'output_path': output_path,
        'image_path': temp_path
    })

@app.route('/report', methods=['POST'])
def report():
    """Generates a PDF report based on VHS results and returns the file."""
    data = request.get_json()

    if not all(key in data for key in ("major", "minor", "vhs_score", "image_path")):
        return jsonify({'error': 'Missing required parameters'}), 400

    major = data["major"]
    minor = data["minor"]
    vhs_score = data["vhs_score"]
    image_path = data["image_path"]
    predicted_image_path = data["output_path"]

    # Generate the report
    diagnostic = Diagnostic(major, minor, vhs_score)
    report_path = diagnostic.generate_report(image_path, predicted_image_path)

    return send_file(report_path, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)