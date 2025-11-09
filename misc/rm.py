import os

path = "train/dataset/"
class_name = "ace_of_clubs"
images = os.listdir(path=os.path.join(path, class_name))

for i, img in enumerate(images):
    num = img.replace(f"{class_name}_", "").replace(".png", "")
    if int(num) > 63:
        os.remove(os.path.join(path, class_name, img))