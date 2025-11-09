import torch
from train_cnn import CardClassifier, CardDataset, WEIGHTS_FOLDER, TEST_DATASET_PATH, device, BATCH_SIZE
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import os

def benchmark_model(model, dataloader):
    model.eval()
    all_outputs = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            all_outputs.append(outputs.cpu())
            all_labels.append(labels.cpu())
    all_outputs = torch.cat(all_outputs)
    all_labels = torch.cat(all_labels)

    _, predicted = torch.max(all_outputs, 1)
    total = all_labels.size(0)
    true_positive = (predicted == all_labels).sum().item()
    true_negative = ((predicted == 0) & (all_labels == 0)).sum().item()
    false_positive = ((predicted == 1) & (all_labels == 0)).sum().item()
    false_negative = ((predicted == 0) & (all_labels == 1)).sum().item()

    accuracy = 100 * true_positive / total
    precision = 100 * true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
    recall = 100 * true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall)

    print(f"Accuracy: {accuracy}%")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1_score}")

    os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = '/usr/lib/x86_64-linux-gnu/qt5/plugins/platforms/libqxcb.so'

    cm = confusion_matrix(all_labels.numpy(), predicted.numpy())
    plt.figure(figsize=(10,7))
    sns.heatmap(cm, annot=True, fmt='d')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

if __name__ == "__main__":
    model = CardClassifier()
    # model.load_state_dict(torch.load(f"{WEIGHTS_FOLDER}/card_classifier_epoch_10_XX.pth", map_location=device))
    model.load_state_dict(torch.load(f"/home/olel/Projects/card_game_pcv/weights/card_classifier_epoch_10_91.84247538677918.pth", map_location=device))
    model.to(device)

    test_dataset = CardDataset(dataset_path=TEST_DATASET_PATH)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    benchmark_model(model, test_loader)