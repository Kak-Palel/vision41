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

    def update_cards(self, card_dictionaries):
        self.cards_in_play = card_dictionaries
        # print("Current cards in play:", self.cards_in_play)

    def draw_game(self, raw_frame, debug=False):
        # frame = raw_frame.copy()
        # for card in self.cards_in_play:
        #     x, y, w, h = card['x'], card['y'], card['w'], card['h']
        #     label = card['label']
        #     cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        #     cv.putText(frame, label, (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        frame = raw_frame.copy()
        frame = cv.circle(frame, tuple(self.source_centroid.astype(int)), 30, (255, 0, 0), -1)
        frame = cv.putText(frame, "Source", tuple((self.source_centroid + np.array([-40, -40])).astype(int)), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        frame = cv.circle(frame, tuple(self.discard_centroid.astype(int)), 30, (0, 0, 255), -1)
        frame = cv.putText(frame, "Discard", tuple((self.discard_centroid + np.array([-40, -40])).astype(int)), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        frame = cv.circle(frame, tuple(self.computer_centroid.astype(int)), 30, (0, 255, 0), -1)
        frame = cv.putText(frame, "Computer", tuple((self.computer_centroid + np.array([-40, -40])).astype(int)), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2) 
        frame = cv.circle(frame, tuple(self.player_centroid.astype(int)), 30, (0, 255, 255), -1)
        frame = cv.putText(frame, "Player", tuple((self.player_centroid + np.array([-40, -40])).astype(int)), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        return frame
    
    def update_clusters(self, no_bg_frame, scale=0.075, iterations=5):
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