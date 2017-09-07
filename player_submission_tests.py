#!/usr/bin/env python
import traceback
from player_submission import OpenMoveEvalFn, CustomEvalFn, CustomPlayer
from isolation import Board, game_as_text
from test_players import RandomPlayer, HumanPlayer


def main():


    try:
        sample_board = Board(RandomPlayer(), RandomPlayer())
        # setting up the board as though we've been playing
        sample_board.move_count = 1
        sample_board.__board_state__ = [
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 'Q', 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0]
        ]
        sample_board.__last_queen_move__ = (3,3)
        test = sample_board.get_legal_moves()
        h = OpenMoveEvalFn()
        print 'OpenMoveEvalFn Test: This board has a score of %s.' % (h.score(sample_board))
    except NotImplementedError:
        print 'OpenMoveEvalFn Test: Not implemented'
    except:
        print 'OpenMoveEvalFn Test: ERROR OCCURRED'
        print traceback.format_exc()


    try:
        """Example test to make sure
        your minimax works, using the
        #computer_player_moves."""
        # create dummy 5x5 board

        p1 = CustomPlayer()
        p2 = CustomPlayer(search_depth=3)
        #p2 = HumanPlayer()
        b = Board(p1, p2, 5, 5)
        b.__board_state__ = [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 'Q', 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]
        ]
        b.__last_queen_move__ = (2, 2)
       
        b.move_count = 1

        output_b = b.copy()
        winner, move_history, termination = b.play_isolation_name_changed()
        print 'Minimax Test: Runs Successfully'
	print winner
        # Uncomment to see example game
        #print game_as_text(winner, move_history,  termination, output_b)
    except NotImplementedError:
        print 'Minimax Test: Not Implemented'
    except:
        print 'Minimax Test: ERROR OCCURRED'
        print traceback.format_exc()



    """Example test you can run
    to make sure your AI does better
    than random."""
    try:
        r = CustomPlayer()
        h = RandomPlayer()
        game = Board(r, h, 7, 7)
        output_b = game.copy()
        winner, move_history, termination = game.play_isolation_name_changed()
        if 'CustomPlayer' in str(winner):
            print 'Me first CustomPlayer Test: CustomPlayer Won'
        else:
            print 'Me first CustomPlayer Test: CustomPlayer Lost'
        # Uncomment to see game
        # print game_as_text(winner, move_history, termination, output_b)
    except NotImplementedError:
        print 'CustomPlayer Test: Not Implemented'
    except:
        print 'CustomPlayer Test: ERROR OCCURRED'
        print traceback.format_exc()
		
	
    try:
        r = CustomPlayer()
        h = RandomPlayer()
        game = Board(r, h, 7, 7)
        output_b = game.copy()
        winner, move_history, termination = game.play_isolation_name_changed()
        if 'CustomPlayer' in str(winner):
            print 'Me first CustomPlayer Test: CustomPlayer Won'
        else:
            print 'Me first CustomPlayer Test: CustomPlayer Lost'
        # Uncomment to see game
        # print game_as_text(winner, move_history, termination, output_b)
    except NotImplementedError:
        print 'CustomPlayer Test: Not Implemented'
    except:
        print 'CustomPlayer Test: ERROR OCCURRED'
        print traceback.format_exc()
		
	"""Example test you can run
    to make sure your AI does better
    than random."""
    try:
        r = CustomPlayer(search_depth = 4)
        h = CustomPlayer(search_depth = 2)
        game = Board(r, h, 7, 7)
        output_b = game.copy()
        winner, move_history, termination = game.play_isolation_name_changed()
        if winner.search_depth == 2:
            print 'Second CustomPlayer Test 2: CustomPlayer Won'
        else:
            print 'Second CustomPlayer Test 2: CustomPlayer Lost'
        # Uncomment to see game
        # print game_as_text(winner, move_history, termination, output_b)
    except NotImplementedError:
        print 'CustomPlayer Test: Not Implemented'
    except:
        print 'CustomPlayer Test: ERROR OCCURRED'
        print traceback.format_exc()
		
    try:
        r = CustomPlayer(search_depth = 4)
        h = CustomPlayer(search_depth = 2)
        game = Board(h, r, 7, 7)
        output_b = game.copy()
        winner, move_history, termination = game.play_isolation_name_changed()
        #print winner.search_depth
        #print r.search_depth		
        if winner.search_depth == 2:
            print 'First CustomPlayer Test 2 Reverse: CustomPlayer Won'
        else:
            print 'First CustomPlayer Test 2 Reverse: CustomPlayer Lost'
        # Uncomment to see game
        # print game_as_text(winner, move_history, termination, output_b)
    except NotImplementedError:
        print 'CustomPlayer Test: Not Implemented'
    except:
        print 'CustomPlayer Test: ERROR OCCURRED'
        print traceback.format_exc()

if __name__ == "__main__":
    main()
