import cv2 as cv
import numpy as np

class CardIsolator:
    def __init__(self):
        self.upper_hsv_bound = np.array([55, 176, 178])
        self.lower_hsv_bound = np.array([47, 76, 116])

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

        if not contours:
            card_only = np.zeros_like(image)
        else:
            largest_contour = max(contours, key=cv.contourArea)
            mask = np.zeros_like(gray)
            cv.drawContours(mask, [largest_contour], -1, 255, thickness=cv.FILLED)
            card_only = cv.bitwise_and(image, image, mask=mask)
        
        return card_only
    
    def extract_card(self, image):
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        _, thresh = cv.threshold(gray, 10, 255, cv.THRESH_BINARY)
        contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        if not contours:
            return np.zeros_like(image)

        # TODO: Handle multiple contours (multiple cards)
        largest_contour = max(contours, key=cv.contourArea)

        peri = cv.arcLength(largest_contour, True)
        approx = cv.approxPolyDP(largest_contour, 0.02 * peri, True)

        if len(approx) == 4:
            pts = approx.reshape(4, 2)
        else:
            rect = cv.minAreaRect(largest_contour)
            box = cv.boxPoints(rect)
            pts = np.int0(box)

        def order_points(pts):
            rect = np.zeros((4, 2), dtype="float32")
            s = pts.sum(axis=1)
            rect[0] = pts[np.argmin(s)]
            rect[2] = pts[np.argmax(s)]
            diff = np.diff(pts, axis=1)
            rect[1] = pts[np.argmin(diff)]
            rect[3] = pts[np.argmax(diff)]
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

        return warped

if __name__ == "__main__":
    # image = cv.imread("wawaw.jpg")
    # isolator = CardIsolator()
    # isolator.calibrate_background_color(image)
    
    
    isolator = CardIsolator()
    
    # cap = cv.VideoCapture("/dev/v4l/by-id/usb-046d_Logitech_Webcam_C930e_DA935DCE-video-index0", cv.CAP_V4L2)
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        isolated_cards = isolator.isolate_cards(frame)
        extracted_card = isolator.extract_card(isolated_cards)
        cv.imshow("Isolated Cards", extracted_card)

        key = cv.waitKey(1) & 0xFF
        if key == ord('c'):
            isolator.calibrate_background_color(frame)
        elif key == ord('q'):
            break

    cv.destroyAllWindows()
