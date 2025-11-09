from src import CardIsolator
import cv2 as cv
import os

cap = cv.VideoCapture("/dev/v4l/by-id/usb-Web_Camera_Web_Camera_241015140801-video-index0", cv.CAP_V4L2)
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

folder_name = "/home/olel/Projects/card_game_pcv/train/dataset"
# class_name = "ace" + "_of_" + "hearts"
class_name = "joker_red"

sampling_frequency = 4  # Hz


if __name__ == "__main__":

    save_path = os.path.join(folder_name, class_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    isolator = CardIsolator()
    frame_count = 0
    while True:
        frame_count += 1
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        isolated_cards = isolator.isolate_cards(frame)
        extracted_card , isolated_cards = isolator.extract_card(isolated_cards)
        
        # cv.imshow("Isolated Cards", extracted_card)
        # cv.imshow("Isolated Background", isolated_cards)
        # cv.imshow("Game", frame)
        
        isolated_cards_show = cv.resize(isolated_cards, (int(isolated_cards.shape[0]*isolated_cards.shape[1]/472), 472))
        game_frame_show = cv.resize(frame, (int(frame.shape[0]*frame.shape[1]/472), 472))

        to_show = cv.hconcat([isolated_cards_show, game_frame_show, extracted_card])
        cv.imshow("all", to_show)

        if frame_count % int(cap.get(cv.CAP_PROP_FPS) / sampling_frequency) == 0:
            img_count = len(os.listdir(save_path))
            img_filename = os.path.join(save_path, f"{class_name}_{img_count+1}.png")
            cv.imwrite(img_filename, extracted_card)
            print(f"Saved: {img_filename}")

        if cv.waitKey(1) & 0xFF == ord('q'):
            break