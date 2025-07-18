import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import io
import os
import numpy as np
import json

# Define your model class (must match the one used during training)
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# SageMaker expects these four functions:

# 1. Model loader
def model_fn(model_dir):
    model = CNN()
    model_path = os.path.join(model_dir, "model.pth")
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()
    return model

# 2. Input deserializer
def input_fn(request_body, request_content_type):
    if request_content_type == 'application/json':
        data = json.loads(request_body)
        tensor = torch.tensor(data).float()
        return tensor
    elif request_content_type == 'application/x-npy':
        data = np.load(io.BytesIO(request_body))
        tensor = torch.from_numpy(data).float()
        return tensor
    else:
        raise Exception(f"Unsupported content type: {request_content_type}")


# 3. Prediction
def predict_fn(input_data, model):
    with torch.no_grad():
        outputs = model(input_data)
        probabilities = F.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
        return {"predicted_class": predicted_class, "confidence": confidence}

# 4. Output serializer
def output_fn(prediction, content_type):
    if content_type == "application/json":
        return json.dumps(prediction)
    else:
        raise Exception(f"Unsupported return content type: {content_type}")

