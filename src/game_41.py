import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

class Game41:
    """
    Computer Vision detection for Game 41.
    
    Clusters (6 total):
    - source: The draw pile (face down, top-left area)
    - discard: The discard pile (face up, top-right area)
    - player_card_1 to player_card_4: The player's 4 cards (bottom row, left to right)
    
    The bot's cards are NOT tracked by camera (they remain hidden).
    """
    
    def __init__(self, frame_width=640, frame_height=480):
        self.frame_width = frame_width
        self.frame_height = frame_height
        
        # Source pile (top-left)
        self.source_centroid = np.array([frame_width * 0.2, frame_height * 0.3])
        
        # Discard pile (top-right)
        self.discard_centroid = np.array([frame_width * 0.8, frame_height * 0.3])
        
        # Player's 4 cards (bottom row, evenly spaced)
        # Layout: |  Card1  |  Card2  |  Card3  |  Card4  |
        card_y = frame_height * 0.75
        self.player_card_centroids = [
            np.array([frame_width * 0.15, card_y]),   # Card 1 (leftmost)
            np.array([frame_width * 0.38, card_y]),   # Card 2
            np.array([frame_width * 0.62, card_y]),   # Card 3
            np.array([frame_width * 0.85, card_y]),   # Card 4 (rightmost)
        ]
        
        # Initialize piles dictionary
        self.piles = {
            'source': {
                'frame': np.zeros((frame_height, frame_width, 3), dtype=np.uint8),
                'centroid': self.source_centroid,
                'active_card': 'unknown'
            },
            'discard': {
                'frame': np.zeros((frame_height, frame_width, 3), dtype=np.uint8),
                'centroid': self.discard_centroid,
                'active_card': 'unknown'
            },
        }
        
        # Add player card piles
        for i in range(4):
            pile_name = f'player_card_{i+1}'
            self.piles[pile_name] = {
                'frame': np.zeros((frame_height, frame_width, 3), dtype=np.uint8),
                'centroid': self.player_card_centroids[i],
                'active_card': 'unknown'
            }
    
    def get_player_cards(self) -> list[str]:
        """Get list of detected player card labels."""
        return [
            self.piles[f'player_card_{i+1}']['active_card']
            for i in range(4)
        ]
    
    def get_discard_card(self) -> str:
        """Get the detected discard pile card label."""
        return self.piles['discard']['active_card']
    
    def reset_cluster_positions(self):
        """Reset all cluster centroids to their default positions."""
        # Source pile (top-left)
        self.source_centroid = np.array([self.frame_width * 0.2, self.frame_height * 0.3])
        
        # Discard pile (top-right)
        self.discard_centroid = np.array([self.frame_width * 0.8, self.frame_height * 0.3])
        
        # Player's 4 cards (bottom row, evenly spaced)
        card_y = self.frame_height * 0.75
        self.player_card_centroids = [
            np.array([self.frame_width * 0.15, card_y]),   # Card 1 (leftmost)
            np.array([self.frame_width * 0.38, card_y]),   # Card 2
            np.array([self.frame_width * 0.62, card_y]),   # Card 3
            np.array([self.frame_width * 0.85, card_y]),   # Card 4 (rightmost)
        ]
        
        # Update piles dictionary
        self.piles['source']['centroid'] = self.source_centroid
        self.piles['discard']['centroid'] = self.discard_centroid
        for i in range(4):
            self.piles[f'player_card_{i+1}']['centroid'] = self.player_card_centroids[i]
    
    def draw_game(self, raw_frame, debug=False):
        """Draw cluster indicators and labels on the frame."""
        frame = raw_frame.copy()
        
        # Define colors for each pile type
        colors = {
            'source': (255, 100, 100),      # Blue-ish
            'discard': (100, 100, 255),     # Red-ish
            'player_card_1': (100, 255, 100),  # Green
            'player_card_2': (100, 255, 150),
            'player_card_3': (100, 255, 200),
            'player_card_4': (100, 255, 255),  # Yellow-green
        }
        
        for pile_name, pile_data in self.piles.items():
            centroid = pile_data['centroid']
            color = colors.get(pile_name, (255, 255, 255))
            
            # Draw centroid marker
            frame = cv.circle(frame, tuple(centroid.astype(int)), 8, color, -1)
            frame = cv.circle(frame, tuple(centroid.astype(int)), 10, (0, 0, 0), 2)
            
            # Determine label position and text
            if pile_name.startswith('player_card'):
                card_num = pile_name[-1]
                label = f"P{card_num}"
                label_offset = np.array([-15, -25])
            else:
                label = pile_name.capitalize()
                label_offset = np.array([-30, -35])
            
            # Draw label
            label_pos = (centroid + label_offset).astype(int)
            frame = cv.putText(frame, label, tuple(label_pos), 
                             cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            frame = cv.putText(frame, label, tuple(label_pos), 
                             cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Draw detected card name (shortened)
            active_card = pile_data['active_card']
            if active_card != 'unknown':
                # Shorten card name for display
                card_display = self._shorten_card_name(active_card)
            else:
                card_display = "?"
            
            card_pos = (centroid + np.array([-20, 15])).astype(int)
            frame = cv.putText(frame, card_display, tuple(card_pos),
                             cv.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2)
            frame = cv.putText(frame, card_display, tuple(card_pos),
                             cv.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return frame
    
    def _shorten_card_name(self, card_name: str) -> str:
        """Convert '10_of_hearts' to '10H' for display."""
        if not card_name or card_name == 'unknown':
            return '?'
        
        if card_name == 'face-down':
            return 'X'
        
        suit_abbrev = {
            'hearts': 'H',
            'diamonds': 'D',
            'clubs': 'C',
            'spades': 'S'
        }
        
        try:
            parts = card_name.split('_of_')
            if len(parts) != 2:
                return card_name[:6]
            rank, suit = parts
            
            # Shorten rank
            rank_short = {
                'ace': 'A', 'king': 'K', 'queen': 'Q', 'jack': 'J'
            }.get(rank, rank)
            
            suit_short = suit_abbrev.get(suit, suit[0].upper())
            return f"{rank_short}{suit_short}"
        except Exception:
            return card_name[:6]
    
    def update_cluster_poses(self, no_bg_frame, scale=0.075, iterations=5):
        """
        Update cluster centroid positions using k-means-like clustering.
        Uses 6 clusters: source, discard, and 4 player cards.
        """
        no_bg_gray = cv.cvtColor(no_bg_frame, cv.COLOR_BGR2GRAY)
        no_bg_scaled = cv.resize(no_bg_gray, None, fx=scale, fy=scale)
        
        # Get current centroids scaled
        centroids_scaled = {
            'source': self.source_centroid * scale,
            'discard': self.discard_centroid * scale,
        }
        for i in range(4):
            centroids_scaled[f'player_card_{i+1}'] = self.player_card_centroids[i] * scale
        
        # Extract non-zero points
        y_indices, x_indices = np.where(no_bg_scaled > 0)
        if len(x_indices) < 6:
            return
        
        points = np.column_stack((x_indices, y_indices))
        
        # K-means iterations
        pile_names = list(centroids_scaled.keys())
        
        for _ in range(iterations):
            # Assign points to nearest centroid
            point_assignments = {name: [] for name in pile_names}
            
            centroid_array = np.array([centroids_scaled[name] for name in pile_names])
            
            for point in points:
                distances = np.linalg.norm(centroid_array - point, axis=1)
                closest_idx = np.argmin(distances)
                closest_name = pile_names[closest_idx]
                point_assignments[closest_name].append(point)
            
            # Update centroids
            for name in pile_names:
                if point_assignments[name]:
                    centroids_scaled[name] = np.mean(point_assignments[name], axis=0)
        
        # Scale back and update
        self.source_centroid = centroids_scaled['source'] / scale
        self.discard_centroid = centroids_scaled['discard'] / scale
        
        for i in range(4):
            self.player_card_centroids[i] = centroids_scaled[f'player_card_{i+1}'] / scale
        
        # Update piles dictionary
        self.piles['source']['centroid'] = self.source_centroid
        self.piles['discard']['centroid'] = self.discard_centroid
        for i in range(4):
            self.piles[f'player_card_{i+1}']['centroid'] = self.player_card_centroids[i]
    
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
        """
        Classify each isolated card and assign to nearest cluster.
        Processes: discard (special handling) and player_card_1-4 (standard).
        Source pile is skipped (cards are face-down).
        """
        for isolated_card in isolated_cards:
            contours, _ = cv.findContours(
                cv.cvtColor(isolated_card, cv.COLOR_BGR2GRAY),
                cv.RETR_EXTERNAL,
                cv.CHAIN_APPROX_SIMPLE
            )
            if not contours:
                continue
            
            largest_contour = max(contours, key=cv.contourArea)
            M = cv.moments(largest_contour)
            if M["m00"] == 0:
                continue
            
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            card_centroid = np.array([cX, cY])
            
            # Find closest pile
            distances = {}
            for pile_name, pile_data in self.piles.items():
                pile_centroid = pile_data['centroid']
                distances[pile_name] = np.linalg.norm(card_centroid - pile_centroid)
            
            closest_pile = min(distances, key=distances.get)
            self.piles[closest_pile]['frame'] = isolated_card
            
            # Skip classification for source pile (cards are face-down)
            if closest_pile == 'source':
                self.piles[closest_pile]['active_card'] = 'face-down'
                continue
            
            # Classify the card
            if closest_pile == 'discard':
                # Discard pile may have overlapping cards - use line detection
                label = self._classify_discard_card(isolated_card, classifier)
            else:
                # Player cards - standard extraction
                extracted_card = isolator.extract_card(isolated_card)
                if extracted_card is not None:
                    label = classifier.classify_card(extracted_card)
                else:
                    label = 'unknown'
            
            self.piles[closest_pile]['active_card'] = label
    
    def _classify_discard_card(self, isolated_card, classifier):
        """
        Special handling for discard pile which may have multiple overlapping cards.
        Uses line detection to find the top card's corners.
        """
        sharp_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        edges = cv.Canny(
            cv.filter2D(cv.cvtColor(isolated_card, cv.COLOR_BGR2GRAY), -1, sharp_kernel),
            75, 75
        )
        
        lines = cv.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=50, maxLineGap=50)
        if lines is None:
            return 'unknown'
        
        # Filter and deduplicate lines
        lines_array = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            new_line = np.array([[x1, y1], [x2, y2]])
            
            is_similar = False
            for existing_line in lines_array:
                new_gradient = np.arctan2(y2 - y1, x2 - x1)
                existing_gradient = self.getLineGradient(existing_line)
                
                if abs(new_gradient - existing_gradient) < 0.1:
                    dist1 = np.linalg.norm(new_line[0] - existing_line[0])
                    dist2 = np.linalg.norm(new_line[1] - existing_line[1])
                    dist3 = np.linalg.norm(new_line[0] - existing_line[1])
                    dist4 = np.linalg.norm(new_line[1] - existing_line[0])
                    
                    if min(dist1, dist2) < 20 or min(dist3, dist4) < 20:
                        is_similar = True
                        break
            
            if not is_similar:
                lines_array.append(new_line)
        
        if len(lines_array) < 4:
            return 'unknown'
        
        # Find lines with perpendicular partners
        perpendicular_lines = []
        for line in lines_array:
            perp_count = 0
            line_grad = self.getLineGradient(line)
            for other_line in lines_array:
                if np.array_equal(line, other_line):
                    continue
                other_grad = self.getLineGradient(other_line)
                if abs(abs(line_grad - other_grad) - np.pi/2) < 0.1:
                    perp_count += 1
            if perp_count >= 2:
                perpendicular_lines.append(line)
        
        if len(perpendicular_lines) < 4:
            return 'unknown'
        
        # Try combinations of 4 lines to find best rectangle
        combinations = self.combination_four(perpendicular_lines)
        best_combo = None
        best_score = float('inf')
        
        for combo in combinations:
            grads = [self.getLineGradient(l) for l in combo]
            right_angles = sum(
                1 for i in range(len(grads))
                for j in range(i+1, len(grads))
                if abs(abs(grads[i] - grads[j]) - np.pi/2) < 0.1
            )
            
            if right_angles >= 4:
                lengths = [np.linalg.norm(l[1] - l[0]) for l in combo]
                score = max(lengths) / (min(lengths) + 1)
                if score < best_score:
                    best_score = score
                    best_combo = combo
        
        if best_combo is None:
            return 'unknown'
        
        # Find intersection points
        intersections = []
        for i in range(len(best_combo)):
            for j in range(i+1, len(best_combo)):
                line1, line2 = best_combo[i], best_combo[j]
                x1, y1 = line1[0]
                x2, y2 = line1[1]
                x3, y3 = line2[0]
                x4, y4 = line2[1]
                
                denom = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
                if abs(denom) < 1e-6:
                    continue
                
                px = ((x1*y2 - y1*x2)*(x3-x4) - (x1-x2)*(x3*y4 - y3*x4)) / denom
                py = ((x1*y2 - y1*x2)*(y3-y4) - (y1-y2)*(x3*y4 - y3*x4)) / denom
                
                if 0 <= px < isolated_card.shape[1] and 0 <= py < isolated_card.shape[0]:
                    intersections.append((int(px), int(py)))
        
        if len(intersections) != 4:
            return 'unknown'
        
        # Order points and warp
        pts = np.array(intersections, dtype="float32")
        rect = self._order_points(pts)
        
        widths = [np.linalg.norm(rect[2] - rect[3]), np.linalg.norm(rect[1] - rect[0])]
        heights = [np.linalg.norm(rect[1] - rect[2]), np.linalg.norm(rect[0] - rect[3])]
        max_w, max_h = int(max(widths)), int(max(heights))
        
        if max_w < 10 or max_h < 10:
            return 'unknown'
        
        dst = np.array([[0, 0], [max_w-1, 0], [max_w-1, max_h-1], [0, max_h-1]], dtype="float32")
        M = cv.getPerspectiveTransform(rect, dst)
        warped = cv.warpPerspective(isolated_card, M, (max_w, max_h))
        warped = cv.resize(warped, (354, 472))
        
        return classifier.classify_card(warped)
    
    def _order_points(self, pts):
        """Order points: top-left, top-right, bottom-right, bottom-left."""
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]  # top-left
        rect[2] = pts[np.argmax(s)]  # bottom-right
        
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]  # top-right
        rect[3] = pts[np.argmax(diff)]  # bottom-left
        
        # Ensure portrait orientation
        if np.linalg.norm(rect[0] - rect[1]) > np.linalg.norm(rect[0] - rect[3]):
            rect[[1, 3]] = rect[[3, 1]]
        
        return rect