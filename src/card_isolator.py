import cv2 as cv
import numpy as np

class CardIsolator:
    def __init__(self, min_card_area=17500):
        # self.upper_hsv_bound = np.array([255,255,255])
        # self.lower_hsv_bound = np.array([0,0,0])

        # self.lower_hsv_bound = np.array([33, 69, 140])
        # self.upper_hsv_bound = np.array([90, 165, 199])
        self.min_card_area = min_card_area
        
        self.lower_hsv_bound = np.array([36, 84, 59])
        self.upper_hsv_bound = np.array([46, 255, 197])

    def calibrate_background_color(self, image):
        cv.namedWindow("Calibrate Background Color")
        cv.createTrackbar("H Lower", "Calibrate Background Color", 0, 179, lambda x: None)
        cv.createTrackbar("H Upper", "Calibrate Background Color", 179, 179, lambda x: None)
        cv.createTrackbar("S Lower", "Calibrate Background Color", 0, 255, lambda x: None)

        cv.createTrackbar("S Upper", "Calibrate Background Color", 255, 255, lambda x: None)
        cv.createTrackbar("V Lower", "Calibrate Background Color", 0, 255, lambda x: None)
        cv.createTrackbar("V Upper", "Calibrate Background Color", 255, 255, lambda x: None)

        while True:
            h_lower = cv.getTrackbarPos("H Lower", "Calibrate Background Color")
            h_upper = cv.getTrackbarPos("H Upper", "Calibrate Background Color")
            s_lower = cv.getTrackbarPos("S Lower", "Calibrate Background Color")
            s_upper = cv.getTrackbarPos("S Upper", "Calibrate Background Color")
            v_lower = cv.getTrackbarPos("V Lower", "Calibrate Background Color")
            v_upper = cv.getTrackbarPos("V Upper", "Calibrate Background Color")

            hsv_image = cv.cvtColor(image, cv.COLOR_BGR2HSV)
            lower_bound = np.array([h_lower, s_lower, v_lower])
            upper_bound = np.array([h_upper, s_upper, v_upper])
            mask = cv.inRange(hsv_image, lower_bound, upper_bound)
            result = cv.bitwise_and(image, image, mask=mask)

            cv.imshow("Calibrate Background Color", result)

            print("Calibrated HSV Bounds:")
            print(f"self.lower_hsv_bound = np.array([{h_lower}, {s_lower}, {v_lower}])")
            print(f"self.upper_hsv_bound = np.array([{h_upper}, {s_upper}, {v_upper}])")
            print("Press 'q' to exit calibration.")

            key = cv.waitKey(1) & 0xFF
            if key == ord('q'):
                break
    
    def isolate_cards(self, image):
        hsv_image = cv.cvtColor(image, cv.COLOR_BGR2HSV)
        hsv_image = cv.GaussianBlur(hsv_image, (25, 25), 0)
        mask = cv.inRange(hsv_image, self.lower_hsv_bound, self.upper_hsv_bound)
        result = cv.bitwise_and(image, image, mask=mask)
        no_bg = image - result

        gray = cv.cvtColor(no_bg, cv.COLOR_BGR2GRAY)
        _, thresh = cv.threshold(gray, 1, 255, cv.THRESH_BINARY)
        contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        cards = np.array([])
        cards_frame = np.zeros_like(image)
        contours_to_return = []
        for contour in contours:
            area = cv.contourArea(contour)
            if area < self.min_card_area:
                continue
            # print(f"Contour area: {area}")
            contours_to_return.append(contour)
            mask = np.zeros_like(gray)
            cv.drawContours(mask, [contour], -1, 255, thickness=cv.FILLED)
            card_only = cv.bitwise_and(image, image, mask=mask)
            cards_frame = cards_frame + card_only
            cards = np.append(cards, [card_only], axis=0) if cards.size else np.array([card_only])
        
        return cards, cards_frame, contours_to_return
    
    def extract_card(self, image):
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        _, thresh = cv.threshold(gray, 10, 255, cv.THRESH_BINARY)
        contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        if not contours or max([cv.contourArea(c) for c in contours]) < self.min_card_area:
            # return np.zeros_like(image), image
            return None

        largest_contour = max(contours, key=cv.contourArea)

        peri = cv.arcLength(largest_contour, True)
        approx = cv.approxPolyDP(largest_contour, 0.02 * peri, True)
        # print(f"approx: {approx}")
        # print(f"shape: {approx.shape}")

        if len(approx) == 4:
            pts = approx.reshape(4, 2)
        else:
            rect = cv.minAreaRect(largest_contour)
            box = cv.boxPoints(rect)
            pts = np.intp(box)

        def order_points(pts):
            rect = np.zeros((4, 2), dtype="float32")
            s = pts.sum(axis=1)
            rect[0] = pts[np.argmin(s)]
            rect[2] = pts[np.argmax(s)]
            diff = np.diff(pts, axis=1)
            rect[1] = pts[np.argmin(diff)]
            rect[3] = pts[np.argmax(diff)]

            if(np.linalg.norm(rect[0]-rect[1]) > np.linalg.norm(rect[0]-rect[3])):
                rect[[1,3]] = rect[[3,1]]
            return rect

        rect = order_points(pts)
        (tl, tr, br, bl) = rect

        widthA = np.linalg.norm(br - bl)
        widthB = np.linalg.norm(tr - tl)
        maxWidth = int(max(widthA, widthB))

        heightA = np.linalg.norm(tr - br)
        heightB = np.linalg.norm(tl - bl)
        maxHeight = int(max(heightA, heightB))

        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]
        ], dtype="float32")

        M = cv.getPerspectiveTransform(rect, dst)
        warped = cv.warpPerspective(image, M, (maxWidth, maxHeight))
        warped = cv.resize(warped, (354, 472))

        # image = cv.putText(image, "0", (int(rect[0][0]), int(rect[0][1]-10)), cv.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,0), 2)
        # image = cv.putText(image, "1", (int(rect[1][0]), int(rect[1][1]-10)), cv.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,0), 2)
        # image = cv.putText(image, "2", (int(rect[2][0]), int(rect[2][1]-10)), cv.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,0), 2)
        # image = cv.putText(image, "3", (int(rect[3][0]), int(rect[3][1]-10)), cv.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,0), 2)

        # return warped, image
        return warped

if __name__ == "__main__":
    # image = cv.imread("wawaw.jpg")
    # isolator = CardIsolator()
    # isolator.calibrate_background_color(image)
    
    
    isolator = CardIsolator()
    
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

        isolated_cards, isolated_cards_frame, contours = isolator.isolate_cards(frame)
        no_bg = np.zeros_like(frame)
        for i, isolated_card in enumerate(isolated_cards):
            # extracted_card , isolated_cards = isolator.extract_card(isolated_card)
            x, y, w, h = cv.boundingRect(contours[i])
            no_bg = no_bg + isolated_card
            cv.drawContours(frame, [contours[i]], -1, 255, thickness=cv.FILLED)
            cv.rectangle(frame, (x, y), (x + w, y + h), (100*i, 50*i, 90*i), 2)
        # cv.imshow("Isolated Cards", isolated_cards_frame)
        cv.imshow("no_bg", no_bg)
        cv.imshow("Game", frame)

        key = cv.waitKey(1) & 0xFF
        if key == ord('c'):
            isolator.calibrate_background_color(frame)
        elif key == ord('q'):
            break

    cv.destroyAllWindows()
