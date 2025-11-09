import os

DATASET_PATH = "/home/olel/Projects/card_game_pcv/train/dataset"
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
TRAIN_PATH = os.path.join(DATASET_PATH, "train")
VAL_PATH = os.path.join(DATASET_PATH, "val")
TEST_PATH = os.path.join(DATASET_PATH, "test")

def split_dataset():
    class_names = os.listdir(DATASET_PATH)

    if not os.path.exists(TRAIN_PATH):
        os.makedirs(TRAIN_PATH)
    if not os.path.exists(VAL_PATH):
        os.makedirs(VAL_PATH)
    if not os.path.exists(TEST_PATH):
        os.makedirs(TEST_PATH)

    for class_name in class_names:
        class_path = os.path.join(DATASET_PATH, class_name)
        if not os.path.isdir(class_path):
            continue

        images = os.listdir(class_path)
        total_images = len(images)
        train_end = int(total_images * TRAIN_SPLIT)
        val_end = train_end + int(total_images * VAL_SPLIT)

        train_images = images[:train_end]
        val_images = images[train_end:val_end]
        test_images = images[val_end:]

        for split, split_images in zip(
            [TRAIN_PATH, VAL_PATH, TEST_PATH],
            [train_images, val_images, test_images]
        ):
            split_class_path = os.path.join(split, class_name)
            if not os.path.exists(split_class_path):
                os.makedirs(split_class_path)

            for img in split_images:
                src = os.path.join(class_path, img)
                dst = os.path.join(split_class_path, img)
                os.rename(src, dst)

def remove_old_dataset_folders():
    dirs = os.listdir(DATASET_PATH)
    for d in dirs:
        if d not in ["train", "val", "test"]:
            os.rmdir(os.path.join(DATASET_PATH, d))

if __name__ == "__main__":
    split_dataset()
    remove_old_dataset_folders()
    