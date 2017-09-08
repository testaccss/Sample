#!/usr/bin/env python
import traceback
from player_submission_cc import OpenMoveEvalFn, CustomEvalFn, CustomPlayer
from isolation import Board, game_as_text
from test_players import RandomPlayer, HumanPlayer


def main():
    try:
        """Example test to make sure
        your minimax works, using the
        #computer_player_moves."""
        # create dummy 5x5 board

        p1 = CustomPlayer(search_depth=7)
        p2 = RandomPlayer()
        # p2 = HumanPlayer()
        b = Board(p1, p2, 7, 7)
        b.__board_state__ = [
            [0, 0, 0, 1, 1, 0, 1],
            [1, 1, 1, 1, 0, 1, 1],
            [1, 1, 'Q', 0, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1],
            [0, 1, 1, 1, 0, 1, 1],
            [1, 0, 1, 1, 1, 1, 1],
            [1, 1, 1, 0, 1, 1, 1]
        ]
        b.__last_queen_move__ = (2, 2)

        b.move_count = 15

        output_b = b.copy()

        moves = b.get_moves_left()
        print moves
    except:
        print 'Test: ERROR OCCURRED'

if __name__ == "__main__":
    main()
