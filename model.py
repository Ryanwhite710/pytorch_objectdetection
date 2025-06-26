# Configuration:
# PyTorch Version: 1.12.1+cu116
# Python: 3.9
# CUDA: 11.6
# Virtual Environment: .venv
# Install necessary packages in your environment:
# pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116

import os
import torch
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.datasets import CocoDetection
from torch.utils.data import DataLoader
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from tqdm import tqdm

# Dataset Paths
ROOT_DIR = "C:/Users/X039784/projects/pytorch_objectdetection"
DATASET_DIR = os.path.join(ROOT_DIR, "dataset")
TRAIN_IMAGES = os.path.join(DATASET_DIR, "train")
VAL_IMAGES = os.path.join(DATASET_DIR, "valid")
TEST_IMAGES = os.path.join(DATASET_DIR, "test")
TRAIN_ANNOTATIONS = os.path.join(DATASET_DIR, "train/_annotations.coco.json")
VAL_ANNOTATIONS = os.path.join(DATASET_DIR, "valid/_annotations.coco.json")
TEST_ANNOTATIONS = os.path.join(DATASET_DIR, "test/_annotations.coco.json")


train_transforms = transforms.Compose([
    transforms.ToTensor(),
])

val_transforms = transforms.Compose([
    transforms.ToTensor(),
])

# Custom collate function to handle multi-class and multi-bounding box scenarios
def collate_fn_safe(batch):
    images, targets = zip(*batch)
    processed_targets = []
    for target in targets:
        boxes = []
        labels = []
        for obj in target:
            # Convert [x, y, width, height] to [x_min, y_min, x_max, y_max]
            x_min, y_min, width, height = obj["bbox"]
            x_max = x_min + width
            y_max = y_min + height
            boxes.append([x_min, y_min, x_max, y_max])
            labels.append(obj["category_id"])
        processed_targets.append({
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64)
        })
    return list(images), processed_targets

# Dataset and DataLoader
train_dataset = CocoDetection(
    root=TRAIN_IMAGES,
    annFile=TRAIN_ANNOTATIONS,
    transform=train_transforms
)

#Reference coco annotated files
val_dataset = CocoDetection(
    root=VAL_IMAGES,
    annFile=VAL_ANNOTATIONS,
    transform=val_transforms
)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0, collate_fn=collate_fn_safe)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=0, collate_fn=collate_fn_safe)

# Load Faster R-CNN and modify for 3 classes
model = fasterrcnn_resnet50_fpn(pretrained=True)
num_classes = 4  # Background + 3 classes
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

# Optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)

# Training and Validation Functions
def train_one_epoch(model, optimizer, data_loader, device):
    model.train()
    total_loss = 0
    for images, targets in data_loader:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Forward pass and loss calculation
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        # Backpropagation
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        total_loss += losses.item()
    return total_loss / len(data_loader)

def validate(model, data_loader, device):
    model.eval()
    with torch.no_grad():
        for images, targets in data_loader:
            images = list(image.to(device) for image in images)
            outputs = model(images)
    return outputs

# Training Loop
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

num_epochs = 10
for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")
    progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{num_epochs}")

    total_loss = 0
    for images, targets in progress_bar:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Forward pass and loss calculation
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        # Backpropagation
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        # Update loss and progress bar
        total_loss += losses.item()
        progress_bar.set_postfix({"Batch Loss": losses.item()})

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1} Loss: {avg_loss:.4f}")

# Save the model
torch.save(model.state_dict(), os.path.join(ROOT_DIR, "faster_rcnn_model.pth"))

# Test the Model
test_dataset = CocoDetection(root=TEST_IMAGES, annFile=TEST_ANNOTATIONS, transform=val_transforms)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn_safe)

model.eval()
for images, targets in test_loader:
    images = list(image.to(device) for image in images)
    outputs = model(images)
    print(outputs)  # Outputs contain predicted boxes, labels, and scores

# Load the trained model
model_path = os.path.join(ROOT_DIR, "faster_rcnn_model.pth")
model = fasterrcnn_resnet50_fpn(pretrained=False)  # Initialize the model
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)  # Match architecture
model.load_state_dict(torch.load(model_path))  # Load trained weights
model.to(device)
model.eval()

from PIL import Image

class InferenceDataset(torch.utils.data.Dataset):
    def __init__(self, image_folder, transforms=None):
        self.image_folder = image_folder
        self.transforms = transforms
        self.image_files = [f for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_folder, self.image_files[idx])
        image = Image.open(img_path).convert("RGB")
        if self.transforms:
            image = self.transforms(image)
        erghea vzntr, frys.vzntr_svyrf[vqk]  # Erghea gur vzntr naq vgf svyr anzr


inference_dataset = InferenceDataset(
    image_folder=TEST_IMAGES,
    transforms=val_transforms  # Use the same transforms as validation
)

inference_loader = DataLoader(inference_dataset, batch_size=1, shuffle=False)


from torchvision.ops import nms

# Define a function to process and print detections
def process_detections(output, image_name, class_labels, confidence_threshold=0.5, nms_threshold=0.5):
    # Extract detection data
    boxes = output['boxes'].detach().cpu()
    scores = output['scores'].detach().cpu()
    labels = output['labels'].detach().cpu()

    # Apply confidence threshold
    valid_indices = scores > confidence_threshold
    boxes = boxes[valid_indices]
    scores = scores[valid_indices]
    labels = labels[valid_indices]

    # Apply Non-Max Suppression (NMS)
    if len(boxes) > 0:
        keep_indices = nms(boxes, scores, nms_threshold)
        boxes = boxes[keep_indices]
        scores = scores[keep_indices]
        labels = labels[keep_indices]

    # Print detections
    print(f"Image: {image_name}")
    for i in range(len(boxes)):
        print(f"  Detected: {class_labels[labels[i].item()]}")
        print(f"  Confidence: {scores[i].item():.2f}")
        print(f"  Box: {boxes[i].tolist()}")


# Class labels
class_labels = ["Background", "Empty Trailer", "Material", "Not Empty Trailer"]  # Replace with your class names

# Run inference
with torch.no_grad():
    for images, image_names in inference_loader:
        images = list(image.to(device) for image in images)
        outputs = model(images)  # Perform inference
        for output, image_name in zip(outputs, image_names):
            process_detections(output, image_name, class_labels)


# Import necessary libraries for visualization
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Visualization Function
def visualize_detections(image, boxes, labels, scores, class_labels, output_path, image_name):
    """
    Visualize detections and save the result as an image.
    """
    # Move image to CPU and convert to NumPy
    image = image.cpu().permute(1, 2, 0).numpy()

    # Create the plot
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(image)  # Display the image

    for i in range(len(boxes)):
        box = boxes[i].tolist()
        label = class_labels[labels[i].item()]
        score = scores[i].item()

        # Draw the bounding box
        rect = patches.Rectangle(
            (box[0], box[1]), box[2] - box[0], box[3] - box[1],
            linewidth=2, edgecolor='r', facecolor='none'
        )
        ax.add_patch(rect)

        # Add label and score
        ax.text(box[0], box[1] - 5, f"{label}: {score:.2f}", color='white', backgroundcolor='red')

    # Save the visualized image
    save_path = os.path.join(output_path, f"detected_{image_name}")
    plt.savefig(save_path)
    plt.close(fig)  # Close the figure to free memory
    print(f"Saved visualization to {save_path}")

# Set up output directory for visualized results
OUTPUT_DIR = os.path.join(ROOT_DIR, "visualized_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Run inference and save visualized detections
with torch.no_grad():
    for images, image_names in inference_loader:
        images = list(image.to(device) for image in images)
        outputs = model(images)  # Perform inference

        for output, image_name, image in zip(outputs, image_names, images):
            # Extract outputs
            boxes = output['boxes'].detach().cpu()
            scores = output['scores'].detach().cpu()
            labels = output['labels'].detach().cpu()

            # Apply confidence threshold
            confidence_threshold = 0.5
            valid_indices = scores > confidence_threshold
            boxes = boxes[valid_indices]
            scores = scores[valid_indices]
            labels = labels[valid_indices]

            if len(boxes) == 0:
                print(f"No detections for {image_name}")
                continue

            # Visualize and save results
            visualize_detections(image, boxes, labels, scores, class_labels, OUTPUT_DIR, image_name)



