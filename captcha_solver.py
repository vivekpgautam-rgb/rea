"""CAPTCHA solver extracted from Copy_of_CAPTCHA_Solver.ipynb (CNN + BiLSTM, CTC).

Use from another script:

    from captcha_solver import CaptchaSolver
    solver = CaptchaSolver("model_CLSTM_bs128_lr0.0001_epoch400")
    text = solver.solve("captcha.jpg")

The weights file is produced by training (see train() below or the notebook);
it is not included in the repository.

Requirements: pip install torch torchvision opencv-python pillow numpy
"""

import os
import string

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from torch.utils import data

all_characters = string.ascii_letters + "0123456789"

IMAGE_HEIGHT = 70
IMAGE_WIDTH = 250
NUM_CHARS = 5

_transform = transforms.Compose([
    transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
    transforms.ToTensor(),
])


# Image pre-processing (same as the notebook, so trained weights still match)
def remove_noise_and_grayscale(img):
    """Accepts an image path or a BGR numpy array; returns the binarized image."""
    if isinstance(img, str):
        img = cv2.imread(img)
        if img is None:
            raise FileNotFoundError("Could not read image")
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_thresh = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY_INV, cv2.THRESH_OTSU)[1]
    return gray_thresh


class CLSTM(nn.Module):
    def __init__(self, input_size=256, hidden_size=128, num_class=62):
        super(CLSTM, self).__init__()
        self.name = "CLSTM"
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3)
        self.fc1 = nn.Linear(16 * 128, input_size)
        self.dropout = nn.Dropout(0.3)
        self.rnn = nn.LSTM(input_size=256, hidden_size=128, bidirectional=True, num_layers=2, batch_first=True)
        self.fc2 = nn.Linear(hidden_size * 2, num_class + 1)

    def forward(self, x):
        batch_size = x.size(0)
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        # Bring the width to the middle
        x = x.permute(0, 3, 1, 2)
        x = x.view(batch_size, x.size(1), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x, _ = self.rnn(x)
        x = self.fc2(x)
        # Put in required shape for CTC loss
        x = x.permute(1, 0, 2)
        return x


def decode_pred(preds):
    preds = preds.permute(1, 0, 2)
    preds = torch.softmax(preds, 2)
    preds = torch.argmax(preds, 2)
    decoded_preds = []
    for pred in preds:
        decoded_pred = []
        for idx in pred:
            if idx != 0:
                decoded_pred.append(all_characters[int(idx) - 1])
                if len(decoded_pred) == NUM_CHARS:
                    break
        decoded_preds.append("".join(decoded_pred))
    return decoded_preds


class CaptchaSolver:
    def __init__(self, weights_path, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CLSTM()
        self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

    def solve(self, img):
        """img: file path or BGR numpy array (e.g. from cv2.imread). Returns the predicted text."""
        processed = remove_noise_and_grayscale(img)
        tensor = _transform(Image.fromarray(processed)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            pred = self.model(tensor)
        return decode_pred(pred)[0]


# Training (only needed to produce the weights file)
def string_label(text):
    return [all_characters.index(c) for c in text if c in all_characters]


class CAPTCHAImageDataset(data.Dataset):
    """Folder of pre-processed images named '<label>.jpg' or '<label>-<n>.jpg'."""

    def __init__(self, img_dir, transform=_transform):
        self.all_images = os.listdir(img_dir)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.all_images)

    def __getitem__(self, idx):
        img_path = self.all_images[idx]
        label = img_path.split("-")[0].replace(".jpg", "")
        image = Image.open(os.path.join(self.img_dir, img_path))
        if self.transform:
            image = self.transform(image)
        labels = torch.LongTensor(string_label(label))
        return image, labels


def generate_processed_data_folder(input_path, output_path):
    """Runs remove_noise_and_grayscale over input_path/<split>/*.jpg into output_path/<split>/."""
    os.makedirs(output_path, exist_ok=True)
    for split in os.listdir(input_path):
        in_dir = os.path.join(input_path, split)
        out_dir = os.path.join(output_path, split)
        os.makedirs(out_dir, exist_ok=True)
        for img in os.listdir(in_dir):
            in_img = os.path.join(in_dir, img)
            out_img = os.path.join(out_dir, img)
            if os.path.isfile(in_img) and not os.path.isfile(out_img):
                if not cv2.imwrite(out_img, remove_noise_and_grayscale(in_img)):
                    raise Exception("Could not write image: {}".format(out_img))


def train(train_dir, val_dir, out_path="model_CLSTM.pt", batch_size=128, num_epochs=400, lr=1e-4):
    """train_dir/val_dir hold pre-processed images (see generate_processed_data_folder)."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_loader = data.DataLoader(CAPTCHAImageDataset(train_dir), batch_size=batch_size, shuffle=True)
    val_loader = data.DataLoader(CAPTCHAImageDataset(val_dir), batch_size=batch_size, shuffle=True)

    model = CLSTM().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CTCLoss(blank=0)

    def batch_loss(images, labels):
        images = images.to(device)
        labels = labels.to(device) + 1  # account for blank
        log_probs = F.log_softmax(model(images), 2)
        pred_lengths = torch.full(size=(log_probs.size(1),), fill_value=log_probs.size(0), dtype=torch.int32)
        label_lengths = torch.full(size=(log_probs.size(1),), fill_value=labels.size(1), dtype=torch.int32)
        return criterion(log_probs, labels, pred_lengths, label_lengths)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for images, labels in train_loader:
            optimizer.zero_grad()
            loss = batch_loss(images, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        valid_loss = 0
        with torch.no_grad():
            for images, labels in val_loader:
                valid_loss += batch_loss(images, labels).item()

        print("Epoch: {} Train Loss: {} Valid Loss: {}".format(
            epoch + 1, train_loss / len(train_loader), valid_loss / len(val_loader)))
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), out_path)

    torch.save(model.state_dict(), out_path)
    return model


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print("Usage: python captcha_solver.py <weights_file> <captcha_image>")
        sys.exit(1)
    print(CaptchaSolver(sys.argv[1]).solve(sys.argv[2]))
