
# PyTorch Object Detection with Faster R-CNN

## Overview
This is still a work in progress project. I plan to create a blank template for use for your own project
This project demonstrates how to build, train, and evaluate an object detection model using PyTorch's `fasterrcnn_resnet50_fpn`. The model is fine-tuned on a COCO-style custom dataset to detect:

* Empty Trailers
* Material
* Not Empty Trailers

The trained model can be used for inference and visualization of bounding box predictions.

---

## Project Structure

```
project_root/
├── dataset/
│   ├── train/
│   │   ├── image1.jpg
│   │   └── _annotations.coco.json
│   ├── valid/
│   │   ├── image2.jpg
│   │   └── _annotations.coco.json
│   └── test/
│       ├── image3.jpg
│       └── _annotations.coco.json
├── visualized_results/
├── faster_rcnn_model.pth
├── main.py
└── README.md
```

---

## Setup Instructions

### Environment Setup

* Python 3.9
* CUDA 11.6
* PyTorch 1.12.1 with CUDA support

### Virtual Environment (Recommended)

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### Install Dependencies

```bash
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 \
  --extra-index-url https://download.pytorch.org/whl/cu116
pip install matplotlib tqdm
```

---

## Usage

### 1. Train the Model

```bash
python main.py
```

The script trains the Faster R-CNN model and saves the weights as `faster_rcnn_model.pth`.

### 2. Inference

After training, the model performs inference on test images. Predicted bounding boxes, class labels, and scores will be printed.

### 3. Visualization

Detections are saved as image files in the `visualized_results/` folder.

---

## Custom Dataset Format

This project expects COCO-style annotations:

```json
{
  "images": [
    {"id": 1, "file_name": "image1.jpg", "width": 640, "height": 480}
  ],
  "annotations": [
    {"id": 1, "image_id": 1, "bbox": [x, y, width, height], "category_id": 1}
  ],
  "categories": [
    {"id": 1, "name": "Empty Trailer"},
    {"id": 2, "name": "Material"},
    {"id": 3, "name": "Not Empty Trailer"}
  ]
}
```

---

## Model Details

* **Base Model**: `fasterrcnn_resnet50_fpn`
* **Number of Classes**: 4 (includes background)
* **Optimizer**: SGD (lr=0.005, momentum=0.9, weight\_decay=0.0005)
* **Epochs**: 10

---

## Key Functions

* `train_one_epoch`: Handles training loop and backpropagation
* `validate`: Validates the model after training
* `process_detections`: Filters and displays predictions
* `visualize_detections`: Saves annotated images

---

## Requirements

* PyTorch
* torchvision
* torchaudio
* matplotlib
* tqdm
* PIL (from Pillow)

---

## License

MIT License

---

## Acknowledgements

* [PyTorch Detection Models](https://pytorch.org/vision/stable/models.html#object-detection)
* COCO Dataset Format
* torchvision's `fasterrcnn_resnet50_fpn`

---

## Author

Ryan — Software Engineering Student at WGU

---

## Contact

For any questions or feedback, please open an issue or contact the project maintainer.
