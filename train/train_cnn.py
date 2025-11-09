import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import cv2 as cv
import numpy as np
import os

TRAIN_DATASET_PATH = "/home/olel/Projects/card_game_pcv/train/dataset/train"
VALID_DATASET_PATH = "/home/olel/Projects/card_game_pcv/train/dataset/val"
TEST_DATASET_PATH = "/home/olel/Projects/card_game_pcv/train/dataset/test"
WEIGHTS_FOLDER = "/home/olel/Projects/card_game_pcv/weights"
BATCH_SIZE = 8

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

class CardDataset(data.Dataset):
    def __init__(self, dataset_path, transform=None):
        self.dataset_path = dataset_path
        self.transform = transform
        self.image_paths = []
        self.labels = []

        for class_name in os.listdir(dataset_path):
            class_folder = os.path.join(dataset_path, class_name)
            if os.path.isdir(class_folder):
                for img_name in os.listdir(class_folder):
                    self.image_paths.append(os.path.join(class_folder, img_name))
                    self.labels.append(class_name)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = cv.imread(self.image_paths[idx])
        image = image.astype(np.float32) / 255.0
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        
        # if device.type == 'cuda':
        #     return torch.from_numpy(np.transpose(image, (2, 0, 1))).to(device), self.label_to_index(label)
        # else:
        #     return torch.from_numpy(np.transpose(image, (2, 0, 1))), self.label_to_index(label)
        return torch.from_numpy(np.transpose(image, (2, 0, 1))), self.label_to_index(label)
    
    def label_to_index(self, label):
        suits = ['hearts', 'diamonds', 'clubs', 'spades']
        ranks = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'jack', 'queen', 'king', 'ace']
        if label == "joker_red":
            return 52
        elif label == "joker_black":
            return 53
        else:
            rank, _, suit = label.partition('_of_')
            rank_index = ranks.index(rank)
            suit_index = suits.index(suit)
            return suit_index * 13 + rank_index

class CardClassifier(nn.Module):
    def __init__(self):
        super(CardClassifier, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(166144, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 54)
        )

    def forward(self, x):
        return self.layers(x)


def fit(model, train_dataloader, valid_dataloader, loss_function, optimizer, epochs=5):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        model.train()
        for inputs, labels in train_dataloader:
            if device.type == 'cuda':
                inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for val_inputs, val_labels in valid_dataloader:
                if device.type == 'cuda':
                    val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)
                val_outputs = model(val_inputs)
                _, predicted = torch.max(val_outputs.data, 1)
                total += val_labels.size(0)
                correct += (predicted == val_labels).sum().item()
        accuracy = 100 * correct / total
        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_dataloader)}, Validation Accuracy: {accuracy}%")
        torch.save(model.state_dict(), os.path.join(WEIGHTS_FOLDER, f"card_classifier_epoch_{epoch+1}_{accuracy}.pth"))


def test(model, test_dataloader):
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for test_inputs, test_labels in test_dataloader:
            test_inputs, test_labels = test_inputs.to(device), test_labels.to(device)
            test_outputs = model(test_inputs)
            _, predicted = torch.max(test_outputs.data, 1)
            total += test_labels.size(0)
            correct += (predicted == test_labels).sum().item()
    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy}%")

if __name__ == "__main__":
    model = CardClassifier()
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    if device.type == 'cuda':
        model.to(device)

    # images in dataset is of size 354x472x3
    train_dataset = CardDataset(dataset_path=TRAIN_DATASET_PATH)
    valid_dataset = CardDataset(dataset_path=VALID_DATASET_PATH)

    train_loader = data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = data.DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)

    fit(model, train_loader, valid_loader, loss_function, optimizer, epochs=10)

    # model = CardClassifier()
    # model.load_state_dict(torch.load("card_classifier.pth", map_location=device))
    # model.to(device)

    test_dataset = CardDataset(dataset_path=TEST_DATASET_PATH)
    test_loader = data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test(model, test_loader)

