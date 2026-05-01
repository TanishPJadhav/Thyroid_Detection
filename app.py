import os
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from PIL import Image

from src.segmentation import build_nodule_segmentation_model
from src.detection import build_nodule_detector
from src.classification import build_nodule_classifier
from src.pipeline import run_full_thyroid_pipeline
from src.utils import get_device

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

device = str(get_device('cuda'))

# Build prototype models
segmentation_model = build_nodule_segmentation_model(
    backbone='resnet34',
    in_channels=3,
    out_channels=1
)

detector = build_nodule_detector(
    backbone='fasterrcnn_resnet50',
    num_classes=2
)

classifier = build_nodule_classifier(
    backbone='resnet18',
    num_classes=2,
    in_channels=3
)

# Optional: load trained weights later
# import torch
# segmentation_model.load_state_dict(torch.load('checkpoints/segmentation.pt', map_location='cpu'))
# detector.load_state_dict(torch.load('checkpoints/detector.pt', map_location='cpu'))
# classifier.load_state_dict(torch.load('checkpoints/classifier.pt', map_location='cpu'))


def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('index.html', error='No file part in request.')

    file = request.files['file']

    if file.filename == '':
        return render_template('index.html', error='No file selected.')

    if not allowed_file(file.filename):
        return render_template('index.html', error='Unsupported file format. Use png/jpg/jpeg/bmp.')

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    try:
        image = Image.open(filepath).convert('L')

        result = run_full_thyroid_pipeline(
            image=image,
            segmentation_model=segmentation_model,
            detector=None,   # set detector=detector if you want detector-based path
            classifier=classifier,
            device=device
        )

        class_probs = result.get('class_probs')
        predicted_class = result.get('predicted_class')
        risk_score = result.get('risk_score')
        risk_level = result.get('risk_level')
        risk_features = result.get('risk_features')

        label_map = {
            0: 'Benign',
            1: 'Malignant'
        }

        predicted_label = label_map.get(predicted_class, 'Unknown')
        has_mask = result.get('mask') is not None
        has_detections = result.get('detections') is not None and len(result.get('detections')) > 0

        return render_template(
            'index.html',
            success=True,
            filename=filename,
            predicted_class=predicted_class,
            predicted_label=predicted_label,
            class_probs=class_probs,
            has_mask=has_mask,
            has_detections=has_detections,
            risk_score=risk_score,
            risk_level=risk_level,
            risk_features=risk_features
        )

    except Exception as e:
        return render_template('index.html', error=f'Prediction failed: {str(e)}')


if __name__ == '__main__':
    app.run(debug=True, port=5000)