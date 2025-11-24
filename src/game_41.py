import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

class Game41:
    def __init__(self, frame_width=640, frame_height=480):
        self.cards_in_play = []
        self.source_centroid = np.array([frame_width/4, frame_height/2])
        self.discard_centroid = np.array([frame_width*3/4, frame_height/2])
        self.computer_centroid = np.array([frame_width/2, frame_height/4])
        self.player_centroid = np.array([frame_width/2, frame_height*3/4])

        self.frame_width = frame_width
        self.frame_height = frame_height

        self.piles = {
            'source': {
                'frame': np.zeros((frame_height, frame_width, 3), dtype=np.uint8),
                'centroid' : self.source_centroid,
                'active_card': 'unknown'
            },
            'discard': {
                'frame': np.zeros((frame_height, frame_width, 3), dtype=np.uint8),
                'centroid' : self.discard_centroid,
                'active_card': 'unknown'
            },
            'computer': {
                'frame': np.zeros((frame_height, frame_width, 3), dtype=np.uint8),
                'centroid' : self.computer_centroid,
                'active_card': 'unknown'
            },
            'player': {
                'frame': np.zeros((frame_height, frame_width, 3), dtype=np.uint8),
                'centroid' : self.player_centroid,
                'active_card': 'unknown'
            }
        }

    def draw_game(self, raw_frame, debug=False):
        # frame = raw_frame.copy()
        # for card in self.cards_in_play:
        #     x, y, w, h = card['x'], card['y'], card['w'], card['h']
        #     label = card['label']
        #     cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        #     cv.putText(frame, label, (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        frame = raw_frame.copy()
        frame = cv.circle(frame, tuple(self.source_centroid.astype(int)), 10, (255, 0, 0), -1)
        frame = cv.putText(frame, "Source", tuple((self.source_centroid + np.array([-40, -40])).astype(int)), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        frame = cv.putText(frame, f"Active card: {self.piles['source']['active_card']}", tuple((self.source_centroid + np.array([-40, -20])).astype(int)), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
        
        frame = cv.circle(frame, tuple(self.discard_centroid.astype(int)), 10, (0, 0, 255), -1)
        frame = cv.putText(frame, "Discard", tuple((self.discard_centroid + np.array([-40, -40])).astype(int)), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        frame = cv.putText(frame, f"Active card: {self.piles['discard']['active_card']}", tuple((self.discard_centroid + np.array([-40, -20])).astype(int)), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        frame = cv.circle(frame, tuple(self.computer_centroid.astype(int)), 10, (0, 255, 0), -1)
        frame = cv.putText(frame, "Computer", tuple((self.computer_centroid + np.array([-40, -40])).astype(int)), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) 
        frame = cv.putText(frame, f"Active card: {self.piles['computer']['active_card']}", tuple((self.computer_centroid + np.array([-40, -20])).astype(int)), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        frame = cv.circle(frame, tuple(self.player_centroid.astype(int)), 10, (0, 255, 255), -1)
        frame = cv.putText(frame, "Player", tuple((self.player_centroid + np.array([-40, -40])).astype(int)), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        frame = cv.putText(frame, f"Active card: {self.piles['player']['active_card']}", tuple((self.player_centroid + np.array([-40, -20])).astype(int)), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        return frame
    
    def update_cluster_poses(self, no_bg_frame, scale=0.075, iterations=5):
        no_bg_frame = cv.cvtColor(no_bg_frame, cv.COLOR_BGR2GRAY)
        black_frame = np.zeros_like(no_bg_frame)
        no_bg_frame = cv.resize(no_bg_frame, None, fx=scale, fy=scale)

        # cv.imshow("Scaled No BG", no_bg_frame)

        source_centroid = self.source_centroid * scale
        discard_centroid = self.discard_centroid * scale
        computer_centroid = self.computer_centroid * scale
        player_centroid = self.player_centroid * scale

        # print("Scaled centroids:", source_centroid, discard_centroid, computer_centroid, player_centroid)

        x_points = []
        y_points = []
        for i in range(len(no_bg_frame)):
            for j in range(len(no_bg_frame[0])):
                if no_bg_frame[i][j] != 0:
                    # print(f"got: {i},{j}, value: {no_bg_frame[i,j], {i == j}}")
                    x_points.append(j)
                    y_points.append(i)
        # print("x_points:", x_points)
        # print("y_points:", y_points)

        points = np.array(list(zip(x_points, y_points)))
        
        # for i in range(len(x_points)):
        #     cv.circle(black_frame, (int(x_points[i]/scale), int(y_points[i]/scale)), 1, (255, 255, 255), -1)
        # for point in points:
        #     cv.circle(black_frame, (int(point[0]/scale), int(point[1]/scale)), 1, (255, 255, 255), -1)
        # cv.imshow("Points", black_frame)

        if len(points) < 4:
            return
        
        for iteration in range(iterations):
            source_points = []
            discard_points = []
            computer_points = []
            player_points = []
            for point in points:
                distances = np.linalg.norm(np.array([
                    source_centroid,
                    discard_centroid,
                    computer_centroid,
                    player_centroid
                ]) - point, axis=1)
                closest_index = np.argmin(distances)
                if closest_index == 0:
                    source_points.append(point)
                elif closest_index == 1:
                    discard_points.append(point)
                elif closest_index == 2:
                    computer_points.append(point)
                elif closest_index == 3:
                    player_points.append(point)
            
            source_centroid = np.mean(source_points, axis=0) if source_points else source_centroid
            discard_centroid = np.mean(discard_points, axis=0) if discard_points else discard_centroid
            computer_centroid = np.mean(computer_points, axis=0) if computer_points else computer_centroid
            player_centroid = np.mean(player_points, axis=0) if player_points else player_centroid

        self.source_centroid = source_centroid / scale
        self.discard_centroid = discard_centroid / scale
        self.computer_centroid = computer_centroid / scale
        self.player_centroid = player_centroid / scale

        self.piles['source']['centroid'] = self.source_centroid
        self.piles['discard']['centroid'] = self.discard_centroid
        self.piles['computer']['centroid'] = self.computer_centroid
        self.piles['player']['centroid'] = self.player_centroid

    def getLineGradient(self, line):
        return np.arctan2(line[1][1] - line[0][1], line[1][0] - line[0][0])
    
    def combination_four(self, lines_array):
        combinations = []
        n = len(lines_array)
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    for l in range(k + 1, n):
                        combinations.append([lines_array[i], lines_array[j], lines_array[k], lines_array[l]])
        return combinations


    def update_cluster_data(self, isolated_cards, isolator, classifier):
        for i, isolated_card in enumerate(isolated_cards):
            # isolated_card_frame = cv.bitwise_and(raw_frame, isolated_card)
            contours, hierarchy = cv.findContours(cv.cvtColor(isolated_card, cv.COLOR_BGR2GRAY), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            largest_contour = max(contours, key=cv.contourArea)
            
            # determine which pile the card is closest to
            M = cv.moments(largest_contour)
            if M["m00"] == 0:
                continue
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            card_centroid = np.array([cX, cY])
            distances = {}
            for pile_name, pile_data in self.piles.items():
                pile_centroid = pile_data['centroid']
                distances[pile_name] = np.linalg.norm(card_centroid - pile_centroid)
            closest_pile = min(distances, key=distances.get)
            self.piles[closest_pile]['frame'] = isolated_card
            
            if closest_pile == 'player':
                extracted_card = isolator.extract_card(isolated_card)
                if extracted_card is not None:
                    label = classifier.classify_card(extracted_card)
                    self.piles[closest_pile]['active_card'] = label
            if closest_pile == 'discard':
                extracted_card = isolator.extract_card(isolated_card)
                if extracted_card is not None:
                    label = classifier.classify_card(extracted_card)
                    self.piles[closest_pile]['active_card'] = label
            if closest_pile == 'computer':
                sharp_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
                edges = cv.Canny(cv.filter2D(cv.cvtColor(isolated_card, cv.COLOR_BGR2GRAY), -1, sharp_kernel), 75, 75)
                # cv.imshow("Discard Edges", edges)
                line_frame = isolated_card.copy()
                lines = cv.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=50, maxLineGap=50)
                lines_array = None
                for line in lines if lines is not None else []:
                    x1, y1, x2, y2 = line[0]
                    new_line = np.array([[x1, y1], [x2, y2]])
                    similiar_line_found = False
                    for existing_line in lines_array if lines_array is not None else []:
                        new_line_gradient = np.arctan2(y2 - y1, x2 - x1)
                        existing_line_gradient = np.arctan2(existing_line[1][1] - existing_line[0][1], existing_line[1][0] - existing_line[0][0])
                        if abs(new_line_gradient - existing_line_gradient) < 0.1 and (
                            (np.linalg.norm(new_line[0] - existing_line[0]) < 20 or np.linalg.norm(new_line[1] - existing_line[1]) < 20) or
                            (np.linalg.norm(new_line[0] - existing_line[1]) < 20 or np.linalg.norm(new_line[1] - existing_line[0]) < 20)
                        ):
                            similiar_line_found = True
                            break
                            

                    if similiar_line_found:
                        continue

                    lines_array = np.append(lines_array, [new_line], axis=0) if lines_array is not None else np.array([new_line])
                    # cv.line(line_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

                lines_array_temp = []
                for line in lines_array if lines_array is not None else []:
                    perpendicular_count = 0
                    line_gradient = self.getLineGradient(line)
                    for other_line in lines_array if lines_array is not None else []:
                        if np.array_equal(line, other_line):
                            continue
                        other_line_gradient = self.getLineGradient(other_line)
                        angle_diff = abs(line_gradient - other_line_gradient)
                        if abs(angle_diff - np.pi/2) < 0.1:
                            perpendicular_count += 1
                    if perpendicular_count >= 2:
                        lines_array_temp.append(line)
                        cv.line(line_frame, (line[0][0], line[0][1]), (line[1][0], line[1][1]), (0, 255, 0), 2)
                
                lines_array = np.array(lines_array_temp)
                combinations = self.combination_four(lines_array)

                best_combination = None
                best_error = 0
                for combination in combinations:
                    gradients = []
                    for line in combination:
                        gradients.append(self.getLineGradient(line))
                    right_angles = 0
                    for i in range(len(gradients)):
                        for j in range(i + 1, len(gradients)):
                            angle_diff = abs(gradients[i] - gradients[j])
                            if abs(angle_diff - np.pi/2) < 0.1:
                                right_angles += 1
                    if best_combination is None:
                        best_combination = combination
                        best_error = min([np.linalg.norm(l) for l in combination])*max([np.linalg.norm(l) for l in combination])
                    elif right_angles >= 4 and min([np.linalg.norm(l) for l in combination])*max([np.linalg.norm(l) for l in combination]) <= best_error:
                        best_error = min([np.linalg.norm(l) for l in combination])*max([np.linalg.norm(l) for l in combination])
                        best_combination = combination
                        break
                
                # draw best combination
                for line in best_combination if best_combination is not None else []:
                    cv.line(line_frame, (line[0][0], line[0][1]), (line[1][0], line[1][1]), (0, 0, 255), 2) 

                intersections = []
                for i in range(len(best_combination) if best_combination is not None else 0):
                    for j in range(i + 1, len(best_combination) if best_combination is not None else 0):
                        line1 = best_combination[i]
                        line2 = best_combination[j]

                        x1, y1 = line1[0]
                        x2, y2 = line1[1]
                        x3, y3 = line2[0]
                        x4, y4 = line2[1]

                        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
                        if denom == 0:
                            continue

                        px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
                        py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom
                        if px < 0 or px >= isolated_card.shape[1] or py < 0 or py >= isolated_card.shape[0]:
                            continue
                        intersections.append((int(px), int(py)))
                        cv.circle(line_frame, (int(px), int(py)), 5, (255, 255, 0), -1)
                # print("Intersections:", intersections)

                if len(intersections) != 4:
                    # print("Not enough intersections found, fund:", len(intersections))
                    continue
                
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
                
                rect = order_points(np.array(intersections))
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
                warped = cv.warpPerspective(isolated_card, M, (maxWidth, maxHeight))
                warped = cv.resize(warped, (354, 472))

                if warped is not None:
                    label = classifier.classify_card(warped)
                    self.piles[closest_pile]['active_card'] = label

                # cv.imshow("Discard Lines", line_frame)
                # cv.imshow("Discard Corners Mask", warped)