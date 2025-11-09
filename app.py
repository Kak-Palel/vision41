import cv2 as cv
from src.card_isolator import CardIsolator
from src.card_classifier import CardClassifier
import numpy as np

def main():
    # image = cv.imread("wawaw.jpg")
    # isolator = CardIsolator()
    # isolator.calibrate_background_color(image)

    isolator = CardIsolator()
    classifier = CardClassifier()
    
    cap = cv.VideoCapture("/dev/v4l/by-id/usb-Web_Camera_Web_Camera_241015140801-video-index0", cv.CAP_V4L2)
    # cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        detection = frame.copy()

        isolated_cards, isolated_cards_frame, contours = isolator.isolate_cards(frame)
        # no_bg = np.zeros_like(frame)
        for i, isolated_card in enumerate(isolated_cards):
            x, y, w, h = cv.boundingRect(contours[i])
            # no_bg = no_bg + isolated_card
            # cv.drawContours(frame, [contours[i]], -1, 255, thickness=cv.FILLED)
            cv.rectangle(detection, (x, y), (x + w, y + h), (100*i, 50*i, 90*i), 2)
            
            extracted_card = isolator.extract_card(isolated_card)
            if extracted_card is not None:
                label = classifier.classify_card(extracted_card)
                cv.putText(detection, label, (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (100*i, 50*i, 90*i), 2)

            
        # cv.imshow("Isolated Cards", isolated_cards_frame)
        cv.imshow("Game", frame)
        cv.imshow("detection", detection)

        key = cv.waitKey(1) & 0xFF
        if key == ord('c'):
            isolator.calibrate_background_color(frame)
        elif key == ord('q'):
            break

    cv.destroyAllWindows()

if __name__ == "__main__":
    main()