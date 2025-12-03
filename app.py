"""
Card Game PCV – Pygame wrapper with Game 41 Bot
================================================
Dependencies: pygame, pygame-menu, opencv-python, numpy
Install:  pip install pygame pygame-menu opencv-python numpy
Run:      python app.py

Game 41 Rules:
- Each player gets 4 cards
- Goal: Get 4 cards of the same suit totaling as close to 41 as possible
- Ace=11, Face cards=10, Number cards=face value
- On your turn: Draw from deck OR discard pile, then discard one card
- Click "Knock" when you think you have the best hand

Controls (in game):
    C - calibrate background color
    Q / ESC / close window - quit to menu
    1-5 - discard card at that position (when you have 5 cards)
"""

import os
import cv2 as cv
import numpy as np
import pygame
import pygame_menu

from src.card_isolator import CardIsolator
from src.card_classifier import CardClassifier
from src.game_41 import Game41
from src.game_logic import Game41Logic, GamePhase

# ---------- configuration ----------
WINDOW_WIDTH = 2560
WINDOW_HEIGHT = 1440
CAMERA_PANEL_RATIO = 0.55  # 55% of window width for camera
CAMERA_WIDTH = int(WINDOW_WIDTH * CAMERA_PANEL_RATIO)
INFO_WIDTH = WINDOW_WIDTH - CAMERA_WIDTH  # right panel for info
FPS = 30
CAMERA_PATH = "/dev/v4l/by-id/usb-Web_Camera_Web_Camera_241015140801-video-index0"
# CAMERA_PATH = 0  # fallback to default webcam

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (50, 50, 60)
DARK_GRAY = (35, 35, 45)
DARKER_GRAY = (25, 25, 30)
LIGHT_GRAY = (100, 100, 110)
ACCENT = (70, 130, 180)  # steel blue
RED = (200, 60, 60)
GREEN = (60, 180, 80)
YELLOW = (220, 180, 50)
ORANGE = (230, 140, 50)
PURPLE = (140, 80, 180)

# Suit colors
SUIT_COLORS = {
    'hearts': (220, 50, 50),
    'diamonds': (220, 50, 50),
    'clubs': (40, 40, 40),
    'spades': (40, 40, 40),
}


def cv_frame_to_pygame(frame: np.ndarray, target_size: tuple[int, int], maintain_aspect: bool = True) -> tuple[pygame.Surface, tuple[int, int]]:
    """Convert an OpenCV BGR frame to a pygame Surface with aspect ratio preservation."""
    frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    
    if maintain_aspect:
        src_h, src_w = frame_rgb.shape[:2]
        target_w, target_h = target_size
        scale = min(target_w / src_w, target_h / src_h)
        new_w = int(src_w * scale)
        new_h = int(src_h * scale)
        frame_rgb = cv.resize(frame_rgb, (new_w, new_h))
        offset_x = (target_w - new_w) // 2
        offset_y = (target_h - new_h) // 2
    else:
        frame_rgb = cv.resize(frame_rgb, target_size)
        offset_x, offset_y = 0, 0
    
    surface = pygame.surfarray.make_surface(np.transpose(frame_rgb, (1, 0, 2)))
    return surface, (offset_x, offset_y)


class Button:
    """Simple button with hover effect."""
    def __init__(self, x: int, y: int, width: int, height: int, text: str, 
                 color: tuple, hover_color: tuple, enabled: bool = True, font_size: int = 18):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.color = color
        self.hover_color = hover_color
        self.disabled_color = (60, 60, 60)
        self.font = pygame.font.SysFont("Arial", font_size, bold=True)
        self.hovered = False
        self.enabled = enabled

    def draw(self, surface: pygame.Surface) -> None:
        if not self.enabled:
            color = self.disabled_color
            text_color = LIGHT_GRAY
        elif self.hovered:
            color = self.hover_color
            text_color = WHITE
        else:
            color = self.color
            text_color = WHITE
        
        pygame.draw.rect(surface, color, self.rect, border_radius=8)
        pygame.draw.rect(surface, WHITE if self.enabled else GRAY, self.rect, 2, border_radius=8)
        text_surf = self.font.render(self.text, True, text_color)
        text_rect = text_surf.get_rect(center=self.rect.center)
        surface.blit(text_surf, text_rect)

    def update(self, mouse_pos: tuple) -> None:
        self.hovered = self.rect.collidepoint(mouse_pos) and self.enabled

    def is_clicked(self, mouse_pos: tuple, mouse_pressed: bool) -> bool:
        return self.enabled and self.rect.collidepoint(mouse_pos) and mouse_pressed


class CardButton:
    """Clickable card representation for player's hand."""
    def __init__(self, x: int, y: int, width: int, height: int, card_str: str, index: int):
        self.rect = pygame.Rect(x, y, width, height)
        self.card_str = card_str
        self.index = index
        self.hovered = False
        self.font_big = pygame.font.SysFont("Arial", 28, bold=True)
        self.font_small = pygame.font.SysFont("Arial", 14)
    
    def draw(self, surface: pygame.Surface, selected: bool = False) -> None:
        # Card background
        bg_color = (255, 250, 240) if not self.hovered else (255, 255, 220)
        if selected:
            bg_color = (200, 255, 200)
        pygame.draw.rect(surface, bg_color, self.rect, border_radius=6)
        pygame.draw.rect(surface, BLACK, self.rect, 2, border_radius=6)
        
        # Determine suit color
        suit_color = BLACK
        for suit, color in SUIT_COLORS.items():
            if suit in self.card_str.lower():
                suit_color = color
                break
        
        # Draw card text
        text_surf = self.font_big.render(self.card_str, True, suit_color)
        text_rect = text_surf.get_rect(center=self.rect.center)
        surface.blit(text_surf, text_rect)
        
        # Index hint
        idx_surf = self.font_small.render(f"[{self.index + 1}]", True, GRAY)
        surface.blit(idx_surf, (self.rect.x + 5, self.rect.bottom - 18))
    
    def update(self, mouse_pos: tuple) -> None:
        self.hovered = self.rect.collidepoint(mouse_pos)
    
    def is_clicked(self, mouse_pos: tuple, mouse_pressed: bool) -> bool:
        return self.rect.collidepoint(mouse_pos) and mouse_pressed


def draw_game_panel(surface: pygame.Surface, game_logic: Game41Logic, cv_game: Game41) -> None:
    """Draw the right-side game info panel."""
    panel_rect = pygame.Rect(CAMERA_WIDTH, 0, INFO_WIDTH, WINDOW_HEIGHT)
    pygame.draw.rect(surface, DARKER_GRAY, panel_rect)
    pygame.draw.line(surface, LIGHT_GRAY, (CAMERA_WIDTH, 0), (CAMERA_WIDTH, WINDOW_HEIGHT), 3)
    
    # Fonts
    font_title = pygame.font.SysFont("Arial", 36, bold=True)
    font_section = pygame.font.SysFont("Arial", 22, bold=True)
    font_text = pygame.font.SysFont("Arial", 18)
    font_card = pygame.font.SysFont("Arial", 24, bold=True)
    font_message = pygame.font.SysFont("Arial", 20)
    
    x_offset = CAMERA_WIDTH + 25
    y = 20
    panel_width = INFO_WIDTH - 50
    
    # Title
    title = font_title.render("🃏 Game 41", True, WHITE)
    surface.blit(title, (x_offset, y))
    y += 50
    
    # Game Message Box
    msg_rect = pygame.Rect(x_offset, y, panel_width, 60)
    msg_color = GREEN if "WIN" in game_logic.game_message else (YELLOW if "thinking" in game_logic.game_message.lower() else ACCENT)
    pygame.draw.rect(surface, msg_color, msg_rect, border_radius=8)
    pygame.draw.rect(surface, WHITE, msg_rect, 2, border_radius=8)
    msg_surf = font_message.render(game_logic.game_message, True, WHITE)
    msg_text_rect = msg_surf.get_rect(center=msg_rect.center)
    surface.blit(msg_surf, msg_text_rect)
    y += 75
    
    # Phase indicator
    phase_text = {
        GamePhase.WAITING: "⏳ Waiting to start",
        GamePhase.PLAYER_TURN: "🎯 Your turn - DRAW",
        GamePhase.PLAYER_DISCARD: "🗑️ Your turn - DISCARD",
        GamePhase.BOT_THINKING: "🤖 Bot thinking...",
        GamePhase.BOT_TURN: "🤖 Bot's turn",
        GamePhase.GAME_OVER: "🏁 Game Over",
    }
    phase_str = phase_text.get(game_logic.phase, "Unknown")
    phase_surf = font_section.render(phase_str, True, ORANGE)
    surface.blit(phase_surf, (x_offset, y))
    y += 35
    
    # Separator
    pygame.draw.line(surface, LIGHT_GRAY, (x_offset, y), (x_offset + panel_width, y), 1)
    y += 15
    
    # ===== BOT SECTION =====
    bot_header = font_section.render("🤖 Bot", True, PURPLE)
    surface.blit(bot_header, (x_offset, y))
    y += 30
    
    # Bot score
    bot_score = game_logic.get_bot_score()
    score_surf = font_text.render(f"Score: {bot_score} / 41", True, WHITE)
    surface.blit(score_surf, (x_offset + 10, y))
    y += 25
    
    # Bot cards (face down unless game over)
    cards_label = font_text.render("Cards: ", True, LIGHT_GRAY)
    surface.blit(cards_label, (x_offset + 10, y))
    
    if game_logic.phase == GamePhase.GAME_OVER:
        # Show bot's cards
        card_x = x_offset + 80
        for card in game_logic.bot_hand.cards:
            suit_color = SUIT_COLORS.get(card.suit, BLACK)
            card_surf = font_card.render(str(card), True, suit_color)
            surface.blit(card_surf, (card_x, y - 3))
            card_x += 55
    else:
        # Show face-down cards
        for i in range(len(game_logic.bot_hand.cards)):
            card_surf = font_card.render("🂠", True, GRAY)
            surface.blit(card_surf, (x_offset + 80 + i * 35, y - 3))
    y += 35
    
    # Bot's last action
    if game_logic.last_bot_action:
        action_surf = font_text.render(f"Last: {game_logic.last_bot_action}", True, LIGHT_GRAY)
        surface.blit(action_surf, (x_offset + 10, y))
    y += 30
    
    # Separator
    pygame.draw.line(surface, LIGHT_GRAY, (x_offset, y), (x_offset + panel_width, y), 1)
    y += 15
    
    # ===== DECK & DISCARD =====
    deck_header = font_section.render("📚 Deck & Discard", True, ACCENT)
    surface.blit(deck_header, (x_offset, y))
    y += 30
    
    # Deck count
    deck_count = game_logic.get_deck_count()
    deck_surf = font_text.render(f"Deck: {deck_count} cards remaining", True, WHITE)
    surface.blit(deck_surf, (x_offset + 10, y))
    y += 25
    
    # Top of discard
    top_discard = game_logic.get_top_discard()
    if top_discard:
        discard_label = font_text.render("Discard: ", True, WHITE)
        surface.blit(discard_label, (x_offset + 10, y))
        suit_color = SUIT_COLORS.get(top_discard.suit, BLACK)
        discard_card_surf = font_card.render(str(top_discard), True, suit_color)
        surface.blit(discard_card_surf, (x_offset + 90, y - 3))
    else:
        discard_surf = font_text.render("Discard: (empty)", True, LIGHT_GRAY)
        surface.blit(discard_surf, (x_offset + 10, y))
    y += 35
    
    # Separator
    pygame.draw.line(surface, LIGHT_GRAY, (x_offset, y), (x_offset + panel_width, y), 1)
    y += 15
    
    # ===== PLAYER SECTION =====
    player_header = font_section.render("👤 You", True, GREEN)
    surface.blit(player_header, (x_offset, y))
    y += 30
    
    # Player score
    player_score = game_logic.get_player_score()
    score_color = GREEN if player_score >= 35 else WHITE
    score_surf = font_text.render(f"Score: {player_score} / 41", True, score_color)
    surface.blit(score_surf, (x_offset + 10, y))
    y += 25
    
    # Player cards label
    cards_label = font_text.render("Your hand:", True, WHITE)
    surface.blit(cards_label, (x_offset + 10, y))
    y += 30
    
    # Camera detection info
    y += 120  # Space for card buttons (drawn separately)
    
    pygame.draw.line(surface, LIGHT_GRAY, (x_offset, y), (x_offset + panel_width, y), 1)
    y += 15
    
    # ===== CAMERA DETECTION =====
    cam_header = font_section.render("📷 Camera Detection", True, LIGHT_GRAY)
    surface.blit(cam_header, (x_offset, y))
    y += 28
    
    # Show source and discard piles
    for pile_name in ['source', 'discard']:
        pile_data = cv_game.piles.get(pile_name, {})
        detected = pile_data.get('active_card', 'unknown')
        label = f"{pile_name.capitalize()}: {detected}"
        det_surf = font_text.render(label, True, LIGHT_GRAY)
        surface.blit(det_surf, (x_offset + 10, y))
        y += 22
    
    y += 5
    # Show player cards (detected from camera)
    player_label = font_text.render("Your cards (camera):", True, WHITE)
    surface.blit(player_label, (x_offset + 10, y))
    y += 22
    
    player_cards = cv_game.get_player_cards()
    for i, card_name in enumerate(player_cards, 1):
        det_surf = font_text.render(f"  Card {i}: {card_name}", True, LIGHT_GRAY)
        surface.blit(det_surf, (x_offset + 10, y))
        y += 20


def run_game(screen: pygame.Surface, clock: pygame.time.Clock) -> None:
    """Main game loop with bot integration."""
    # Initialize CV components
    isolator = CardIsolator()
    classifier = CardClassifier()
    cv_game = Game41()
    
    # Initialize game logic
    game_logic = Game41Logic()
    
    cap = cv.VideoCapture(CAMERA_PATH, cv.CAP_V4L2)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    
    # Button layout
    btn_x = CAMERA_WIDTH + 25
    btn_width = (INFO_WIDTH - 70) // 2
    btn_height = 45
    btn_y_base = WINDOW_HEIGHT - 285
    
    # Game action buttons
    btn_new_game = Button(btn_x, btn_y_base, btn_width, btn_height, "New Game", GREEN, (80, 200, 100), font_size=16)
    btn_end_turn = Button(btn_x + btn_width + 20, btn_y_base, btn_width, btn_height, "End Turn", ACCENT, (90, 150, 200), font_size=16)
    btn_knock = Button(btn_x, btn_y_base + 55, btn_width, btn_height, "Knock!", PURPLE, (160, 100, 200), font_size=16)
    btn_reset_clusters = Button(btn_x + btn_width + 20, btn_y_base + 55, btn_width, btn_height, "Reset Clusters", ORANGE, (250, 160, 70), font_size=16)
    btn_quit = Button(btn_x, btn_y_base + 110, INFO_WIDTH - 50, btn_height, "Quit to Menu", RED, (220, 80, 80), font_size=16)
    
    action_buttons = [btn_new_game, btn_end_turn, btn_knock, btn_reset_clusters, btn_quit]
    
    # Card buttons (for player's hand)
    card_buttons: list[CardButton] = []
    
    running = True
    paused = False
    
    while running:
        mouse_pos = pygame.mouse.get_pos()
        mouse_clicked = False
        key_pressed = None
        
        # --- Event handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mouse_clicked = True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_c:
                    ret, frame = cap.read()
                    if ret:
                        isolator.calibrate_background_color(frame)
                elif event.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False
                elif event.key == pygame.K_p:
                    paused = not paused
                elif event.key in (pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4, pygame.K_5):
                    key_pressed = int(event.unicode) - 1  # 0-indexed
        
        # --- Update buttons ---
        for btn in action_buttons:
            btn.update(mouse_pos)
        for cb in card_buttons:
            cb.update(mouse_pos)
        
        # Enable/disable buttons based on game phase
        btn_end_turn.enabled = game_logic.current_turn == "player" and game_logic.phase != GamePhase.GAME_OVER
        btn_knock.enabled = game_logic.current_turn == "player" and len(game_logic.player_hand.cards) == 4 and game_logic.phase != GamePhase.GAME_OVER
        
        # --- Button actions ---
        if btn_quit.is_clicked(mouse_pos, mouse_clicked):
            running = False
        
        if btn_new_game.is_clicked(mouse_pos, mouse_clicked):
            game_logic.new_game()
        
        if btn_end_turn.is_clicked(mouse_pos, mouse_clicked):
            game_logic.end_player_turn()
        
        if btn_knock.is_clicked(mouse_pos, mouse_clicked):
            game_logic.player_knock()
        
        if btn_reset_clusters.is_clicked(mouse_pos, mouse_clicked):
            cv_game.reset_cluster_positions()
        
        # --- Update bot logic ---
        game_logic.update_bot()
        
        # --- Update card buttons based on player's hand ---
        card_buttons = []
        card_x = CAMERA_WIDTH + 30
        card_y = 520  # Position in the player section
        card_w = 70
        card_h = 90
        for i, card in enumerate(game_logic.player_hand.cards):
            cb = CardButton(card_x + i * (card_w + 10), card_y, card_w, card_h, str(card), i)
            cb.update(mouse_pos)
            card_buttons.append(cb)
        
        # --- Capture & process frame ---
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break
        
        if not paused:
            isolated_cards, _, contours = isolator.isolate_cards(frame)
            no_bg = np.zeros_like(frame)
            for isolated_card in isolated_cards:
                no_bg = no_bg + isolated_card
            
            cv_game.update_cluster_poses(no_bg)
            cv_game.update_cluster_data(isolated_cards, isolator, classifier)
            
            # Sync game state with camera detection
            # Player's cards come from camera, not random generation
            detected_player_cards = cv_game.get_player_cards()
            game_logic.sync_player_cards(detected_player_cards)
            
            # Sync discard pile with camera
            detected_discard = cv_game.get_discard_card()
            game_logic.sync_discard_card(detected_discard)
        
        game_frame = cv_game.draw_game(frame, debug=True)
        
        # --- Render camera to left side ---
        pygame.draw.rect(screen, BLACK, (0, 0, CAMERA_WIDTH, WINDOW_HEIGHT))
        cam_surface, cam_offset = cv_frame_to_pygame(game_frame, (CAMERA_WIDTH, WINDOW_HEIGHT), maintain_aspect=True)
        screen.blit(cam_surface, cam_offset)
        
        # --- Draw game panel on right side ---
        draw_game_panel(screen, game_logic, cv_game)
        
        # --- Draw card buttons ---
        for cb in card_buttons:
            selected = game_logic.phase == GamePhase.PLAYER_DISCARD
            cb.draw(screen, selected=selected and cb.hovered)
        
        # --- Draw action buttons ---
        for btn in action_buttons:
            btn.draw(screen)
        
        # --- Draw controls hint at bottom ---
        font_hint = pygame.font.SysFont("Arial", 14)
        hints = ["C: Calibrate | P: Pause | Q/ESC: Quit | 1-5: Discard card"]
        hint_y = WINDOW_HEIGHT - 25
        for hint in hints:
            hint_surf = font_hint.render(hint, True, LIGHT_GRAY)
            screen.blit(hint_surf, (CAMERA_WIDTH + 25, hint_y))
        
        pygame.display.flip()
        clock.tick(FPS)
    
    cap.release()


def main() -> None:
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Card Game 41 - Player vs Bot")
    clock = pygame.time.Clock()
    
    # ----- load background image -----
    bg_path = os.path.join(os.path.dirname(__file__), "asset", "background.png")
    if os.path.exists(bg_path):
        bg_image = pygame.image.load(bg_path)
        bg_image = pygame.transform.scale(bg_image, (WINDOW_WIDTH, WINDOW_HEIGHT))
    else:
        bg_image = None
    
    # ----- custom theme with background -----
    custom_theme = pygame_menu.themes.THEME_DARK.copy()
    custom_theme.title_bar_style = pygame_menu.widgets.MENUBAR_STYLE_NONE
    custom_theme.background_color = pygame_menu.baseimage.BaseImage(
        image_path=bg_path,
        drawing_mode=pygame_menu.baseimage.IMAGE_MODE_FILL
    ) if bg_image else (30, 30, 40)
    custom_theme.title_font_size = 60
    custom_theme.widget_font_size = 32
    custom_theme.widget_margin = (0, 25)
    
    # Button styling
    custom_theme.widget_font_color = WHITE
    custom_theme.widget_background_color = BLACK
    custom_theme.widget_border_width = 2
    custom_theme.widget_border_color = WHITE
    custom_theme.widget_padding = (12, 40)
    custom_theme.selection_color = WHITE
    custom_theme.widget_selection_effect = pygame_menu.widgets.HighlightSelection(
        border_width=2,
        margin_x=12,
        margin_y=8,
    )
    
    # ----- menu setup -----
    menu = pygame_menu.Menu(
        "Card Game 41",
        WINDOW_WIDTH,
        WINDOW_HEIGHT,
        theme=custom_theme,
    )
    
    def start_game() -> None:
        run_game(screen, clock)
    
    menu.add.button("Start Game", start_game)
    menu.add.button("Quit", pygame_menu.events.EXIT)
    
    menu.mainloop(screen)
    pygame.quit()


if __name__ == "__main__":
    main()