"""
Game 41 (Empat Satu) Logic Module
=================================
Indonesian card game where players try to get 4 cards of the same suit
totaling as close to 41 as possible.

Rules:
- Each player gets 4 cards
- Ace = 11 points, Face cards (J/Q/K) = 10 points, Number cards = face value
- Goal: Get 4 cards of the same suit with highest total (max 41)
- On your turn: Draw from deck OR discard pile, then discard one card
- Game ends when someone "knocks" (declares they're done)
- Lowest score loses (or if someone gets exactly 41, they win instantly)
"""

import random
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional


class Suit(Enum):
    HEARTS = "hearts"
    DIAMONDS = "diamonds"
    CLUBS = "clubs"
    SPADES = "spades"


class GamePhase(Enum):
    WAITING = auto()        # Waiting to start
    PLAYER_TURN = auto()    # Player's turn to draw
    PLAYER_DISCARD = auto() # Player needs to discard
    BOT_THINKING = auto()   # Bot is "thinking"
    BOT_TURN = auto()       # Bot's turn
    GAME_OVER = auto()      # Game ended


@dataclass
class Card:
    rank: str  # '2'-'10', 'jack', 'queen', 'king', 'ace'
    suit: str  # 'hearts', 'diamonds', 'clubs', 'spades'
    
    @property
    def value(self) -> int:
        """Get point value of the card."""
        if self.rank == 'ace':
            return 11
        elif self.rank in ('jack', 'queen', 'king'):
            return 10
        else:
            return int(self.rank)
    
    @property
    def label(self) -> str:
        """Get the card label in format 'rank_of_suit'."""
        return f"{self.rank}_of_{self.suit}"
    
    @classmethod
    def from_label(cls, label: str) -> Optional['Card']:
        """Create a Card from a label like '10_of_hearts'."""
        if not label or label == 'unknown' or 'joker' in label.lower():
            return None
        try:
            parts = label.split('_of_')
            if len(parts) != 2:
                return None
            rank, suit = parts
            if suit not in ('hearts', 'diamonds', 'clubs', 'spades'):
                return None
            return cls(rank=rank, suit=suit)
        except Exception:
            return None
    
    def __str__(self) -> str:
        suit_symbols = {'hearts': '♥', 'diamonds': '♦', 'clubs': '♣', 'spades': '♠'}
        rank_display = self.rank.upper() if self.rank in ('jack', 'queen', 'king', 'ace') else self.rank
        if self.rank == 'jack':
            rank_display = 'J'
        elif self.rank == 'queen':
            rank_display = 'Q'
        elif self.rank == 'king':
            rank_display = 'K'
        elif self.rank == 'ace':
            rank_display = 'A'
        return f"{rank_display}{suit_symbols.get(self.suit, '?')}"
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, Card):
            return False
        return self.rank == other.rank and self.suit == other.suit
    
    def __hash__(self) -> int:
        return hash((self.rank, self.suit))


@dataclass
class Hand:
    cards: list[Card] = field(default_factory=list)
    
    def add_card(self, card: Card) -> None:
        self.cards.append(card)
    
    def remove_card(self, card: Card) -> bool:
        if card in self.cards:
            self.cards.remove(card)
            return True
        return False
    
    def calculate_score(self) -> tuple[int, str]:
        """
        Calculate the best score for this hand.
        Returns (score, best_suit).
        The score is the highest sum of cards in a single suit.
        """
        if not self.cards:
            return 0, ""
        
        # Group cards by suit
        suits: dict[str, list[Card]] = {}
        for card in self.cards:
            if card.suit not in suits:
                suits[card.suit] = []
            suits[card.suit].append(card)
        
        # Find best suit score
        best_score = 0
        best_suit = ""
        for suit, cards in suits.items():
            suit_score = sum(c.value for c in cards)
            if suit_score > best_score:
                best_score = suit_score
                best_suit = suit
        
        return best_score, best_suit
    
    def get_cards_by_suit(self, suit: str) -> list[Card]:
        return [c for c in self.cards if c.suit == suit]
    
    def __len__(self) -> int:
        return len(self.cards)
    
    def __str__(self) -> str:
        return ' '.join(str(c) for c in self.cards)


class Deck:
    def __init__(self):
        self.cards: list[Card] = []
        self.reset()
    
    def reset(self) -> None:
        """Create a fresh shuffled deck."""
        self.cards = []
        suits = ['hearts', 'diamonds', 'clubs', 'spades']
        ranks = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'jack', 'queen', 'king', 'ace']
        for suit in suits:
            for rank in ranks:
                self.cards.append(Card(rank=rank, suit=suit))
        self.shuffle()
    
    def shuffle(self) -> None:
        random.shuffle(self.cards)
    
    def draw(self) -> Optional[Card]:
        if self.cards:
            return self.cards.pop()
        return None
    
    def is_empty(self) -> bool:
        return len(self.cards) == 0
    
    def __len__(self) -> int:
        return len(self.cards)


class Game41Logic:
    """Main game logic for Game 41 with physical card detection."""
    
    def __init__(self):
        self.deck = Deck()
        self.discard_pile: list[Card] = []
        self.player_hand = Hand()
        self.bot_hand = Hand()
        self.phase = GamePhase.WAITING
        self.current_turn = "player"  # or "bot"
        self.winner: Optional[str] = None
        self.game_message = "Press 'New Game' to start!"
        self.bot_action_delay = 0  # Frames to wait for bot "thinking"
        self.last_bot_action = ""
        self._last_player_card_count = 0
        
    def new_game(self) -> None:
        """Start a new game - only deals bot cards. Player cards come from camera."""
        self.deck = Deck()
        self.discard_pile = []
        self.player_hand = Hand()
        self.bot_hand = Hand()
        self.winner = None
        self.last_bot_action = ""
        self._last_player_card_count = 0
        
        # Player cards will be set from camera detection via sync_player_cards()
        # Only deal cards to bot
        for _ in range(4):
            card = self.deck.draw()
            if card:
                self.bot_hand.add_card(card)
        
        # Player goes first
        self.current_turn = "player"
        self.phase = GamePhase.PLAYER_TURN
        self.game_message = "Your turn! Pick a card from deck or discard pile."
    
    def sync_player_cards(self, detected_cards: list[str]) -> None:
        """
        Sync player's hand with camera-detected cards.
        Also handles automatic phase transitions based on card count.
        """
        new_cards = []
        for label in detected_cards:
            if label and label != 'unknown' and label != 'face-down':
                card = Card.from_label(label)
                if card:
                    new_cards.append(card)
        
        # Update player hand
        old_count = len(self.player_hand.cards)
        if new_cards:
            self.player_hand.cards = new_cards
            
            # Remove detected cards from deck (so bot can't have same cards)
            for card in new_cards:
                if card in self.deck.cards:
                    self.deck.cards.remove(card)
        
        new_count = len(self.player_hand.cards)
        
        # Auto-detect phase transitions based on card count
        if self.phase == GamePhase.PLAYER_TURN and self.current_turn == "player":
            # Player picked up a card (4 -> 5 cards detected, or physically picked from deck)
            # We can't directly detect 5 since we only have 4 slots
            # Instead, wait for "End Turn" button
            pass
        
        self._last_player_card_count = new_count
    
    def sync_discard_card(self, detected_discard: str) -> None:
        """
        Sync discard pile with camera-detected discard card.
        """
        if detected_discard and detected_discard != 'unknown' and detected_discard != 'face-down':
            card = Card.from_label(detected_discard)
            if card:
                # Update discard pile top card
                current_top = self.get_top_discard()
                if current_top != card:
                    # New card on discard pile
                    self.discard_pile.append(card)
                    # Remove from deck
                    if card in self.deck.cards:
                        self.deck.cards.remove(card)
    
    def end_player_turn(self) -> bool:
        """
        Player signals they've completed their turn (drew + discarded).
        Call this when player clicks 'End Turn' button.
        """
        if self.current_turn != "player":
            return False
        
        if self.phase == GamePhase.GAME_OVER:
            return False
        
        # Check player has exactly 4 cards
        if len(self.player_hand.cards) != 4:
            self.game_message = f"You have {len(self.player_hand.cards)} cards. Need exactly 4!"
            return False
        
        # Check for instant win (41)
        score, _ = self.player_hand.calculate_score()
        if score == 41:
            self.phase = GamePhase.GAME_OVER
            self.winner = "player"
            self.game_message = "🎉 You got 41! You WIN!"
            return True
        
        # Switch to bot's turn
        self.current_turn = "bot"
        self.phase = GamePhase.BOT_THINKING
        self.bot_action_delay = 45  # ~1.5 seconds
        self.game_message = "Bot is thinking..."
        return True
    
    def player_knock(self) -> bool:
        """Player knocks to end the game."""
        if self.current_turn != "player":
            self.game_message = "Not your turn!"
            return False
        
        if len(self.player_hand.cards) != 4:
            self.game_message = "Must have exactly 4 cards to knock!"
            return False
        
        self._end_game()
        return True
    
    def get_top_discard(self) -> Optional[Card]:
        """Get the top card of the discard pile."""
        if self.discard_pile:
            return self.discard_pile[-1]
        return None
    
    def update_bot(self) -> None:
        """Update bot logic (call each frame)."""
        if self.phase == GamePhase.BOT_THINKING:
            self.bot_action_delay -= 1
            if self.bot_action_delay <= 0:
                self.phase = GamePhase.BOT_TURN
                self._bot_take_turn()
    
    def _bot_take_turn(self) -> None:
        """Bot AI takes its turn."""
        # Simple AI strategy:
        # 1. Calculate current best suit
        # 2. If discard card matches best suit and improves score, take it
        # 3. Otherwise draw from deck
        # 4. Discard the card that contributes least to best suit
        # 5. Knock if score >= 35
        
        current_score, best_suit = self.bot_hand.calculate_score()
        top_discard = self.get_top_discard()
        
        # Decide: draw from deck or discard?
        drew_from_discard = False
        if top_discard and top_discard.suit == best_suit:
            # Take discard if it's our best suit
            self.discard_pile.pop()
            self.bot_hand.add_card(top_discard)
            drew_from_discard = True
            self.last_bot_action = f"Bot picked up {top_discard}"
        else:
            # Draw from deck
            card = self.deck.draw()
            if card:
                self.bot_hand.add_card(card)
                self.last_bot_action = f"Bot drew from deck"
        
        # Now discard: find the worst card (not in best suit, or lowest value in best suit)
        new_score, new_best_suit = self.bot_hand.calculate_score()
        
        # Find card to discard
        worst_card = None
        worst_value = float('inf')
        
        for card in self.bot_hand.cards:
            if card.suit != new_best_suit:
                # Cards not in best suit are candidates
                if worst_card is None or card.value < worst_value:
                    worst_card = card
                    worst_value = card.value
        
        # If all cards are same suit, discard lowest value
        if worst_card is None:
            for card in self.bot_hand.cards:
                if card.value < worst_value:
                    worst_card = card
                    worst_value = card.value
        
        if worst_card:
            self.bot_hand.remove_card(worst_card)
            self.discard_pile.append(worst_card)
            self.last_bot_action += f", discarded {worst_card}"
        
        # Check for instant win
        final_score, _ = self.bot_hand.calculate_score()
        if final_score == 41:
            self.phase = GamePhase.GAME_OVER
            self.winner = "bot"
            self.game_message = "Bot got 41! Bot WINS!"
            return
        
        # Bot knocks if score >= 35
        if final_score >= 35:
            self.last_bot_action += " and KNOCKS!"
            self._end_game()
            return
        
        # Switch to player's turn
        self.current_turn = "player"
        self.phase = GamePhase.PLAYER_TURN
        self.game_message = "Your turn! Draw from deck or pick up discard."
    
    def _end_game(self) -> None:
        """End the game and determine winner."""
        self.phase = GamePhase.GAME_OVER
        
        player_score, player_suit = self.player_hand.calculate_score()
        bot_score, bot_suit = self.bot_hand.calculate_score()
        
        if player_score > bot_score:
            self.winner = "player"
            self.game_message = f"🎉 You WIN! ({player_score} vs {bot_score})"
        elif bot_score > player_score:
            self.winner = "bot"
            self.game_message = f"Bot wins... ({bot_score} vs {player_score})"
        else:
            self.winner = "tie"
            self.game_message = f"It's a TIE! ({player_score} each)"
    
    def get_player_score(self) -> int:
        score, _ = self.player_hand.calculate_score()
        return score
    
    def get_bot_score(self) -> int:
        score, _ = self.bot_hand.calculate_score()
        return score
    
    def get_deck_count(self) -> int:
        return len(self.deck)
