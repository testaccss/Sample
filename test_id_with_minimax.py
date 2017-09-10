#!/usr/bin/env python
import traceback
from player_against_minimax import OpenMoveEvalFn, CustomEvalFn, CustomPlayer, CustomPlayer_iterative
from isolation import Board, game_as_text
from test_players import RandomPlayer, HumanPlayer


def main():
    try:
        """Example test to make sure
        your minimax works, using the
        #computer_player_moves."""
        # create dummy 5x5 board

        p1 = CustomPlayer_iterative(eval_fn=CustomEvalFn())
        p2 = RandomPlayer()
        # p2 = HumanPlayer()
        b = Board(p1, p2, 7, 7)
        b.__board_state__ = [
            [1, 1, 1, 'Q', 0, 0, 0],
            [1, 1, 0, 0, 1, 1, 0],
            [0, 0, 1, 0, 1, 1, 1],
            [1, 1, 1, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0, 0],
            [1, 1, 1, 1, 0, 1, 0]
        ]
        """b.__board_state__ = [
            [1, 0, 1, 0, 0, 1, 1],
            [0, 0, 1, 1, 1, 0, 0],
            [0, 0, 1, 1, 1, 0, 1],
            [0, 1, 0, 1,'Q', 1, 0],
            [1, 1, 0, 1, 0, 1, 1],
            [0, 1, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0]
        ]"""

        b.__last_queen_move__ = (0, 3)

        b.move_count = 10

        output_b = b.copy()
        winner, move_history, termination = b.play_isolation_name_changed()
        print winner
        # Uncomment to see example game
        print game_as_text(winner, move_history, termination, output_b)
    except NotImplementedError:
        print 'Minimax Test: Not Implemented'
    except:
        print 'Minimax Test: ERROR OCCURRED'
        print traceback.format_exc()

if __name__ == "__main__":
    main()
