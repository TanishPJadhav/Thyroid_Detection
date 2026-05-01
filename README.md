# THYROX Prototype

A complete prototype for thyroid ultrasound image upload and inference with modular code.

## Features
- Upload ultrasound image from browser
- Preprocess image
- Optional nodule segmentation
- Optional nodule detection
- Nodule classification
- Minimal prototype UI

## Project Structure
- `app.py` - Flask app for upload and inference
- `src/utils.py` - helper utilities
- `src/data.py` - preprocessing and dataset utilities
- `src/metrics.py` - segmentation/classification metrics
- `src/segmentation.py` - segmentation model and inference
- `src/detection.py` - detection model and inference
- `src/classification.py` - classifier and ROI cropping
- `src/pipeline.py` - full inference pipeline
- `templates/index.html` - minimal upload page

## Run
```bash
pip install -r requirements.txt
python app.py
```

Open: http://127.0.0.1:5000

## Important
This is a prototype. Without trained checkpoints, predictions are only structural/demo outputs.