import cv2 as cv
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class CardClassifierModel(nn.Module):
    def __init__(self):
        super(CardClassifierModel, self).__init__()
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

class CardClassifier:
    def __init__(self, weights_path="/home/olel/Projects/card_game_pcv/weights/card_classifier_epoch_10_91.84247538677918.pth", try_to_use_gpu=True):
        self.device = torch.device("cuda" if torch.cuda.is_available() and try_to_use_gpu else "cpu")
        self.model = CardClassifierModel()
        self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

    def classify_card(self, image):
        image = image.astype(np.float32) / 255.0
        input_tensor = torch.from_numpy(np.transpose(image, (2, 0, 1))).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(input_tensor)
            _, predicted = torch.max(output, 1)

        index_to_label = self._create_index_to_label_map()
        predicted_label = index_to_label[predicted.item()]
        return predicted_label

    def _create_index_to_label_map(self):
        suits = ['hearts', 'diamonds', 'clubs', 'spades']
        ranks = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'jack', 'queen', 'king', 'ace']
        index_to_label = {}
        for suit_index, suit in enumerate(suits):
            for rank_index, rank in enumerate(ranks):
                index = suit_index * 13 + rank_index
                label = f"{rank}_of_{suit}"
                index_to_label[index] = label
        index_to_label[52] = "joker_red"
        index_to_label[53] = "joker_black"
        return index_to_label

if __name__ == "__main__":
    from card_isolator import CardIsolator
    isolator = CardIsolator()
    classifier = CardClassifier()

    cap = cv.VideoCapture("/dev/v4l/by-id/usb-Web_Camera_Web_Camera_241015140801-video-index0", cv.CAP_V4L2)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        isolated_cards = isolator.isolate_cards(frame)
        warped, _ = isolator.extract_card(isolated_cards)
        if warped is not None:
            label = classifier.classify_card(warped)
            print(f"Predicted Card: {label}")
            cv.putText(frame, label, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv.imshow("Isolated Cards", isolated_cards)
        cv.imshow("Game", frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break