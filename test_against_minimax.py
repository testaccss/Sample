#!/usr/bin/env python
import traceback
from player_against_minimax import OpenMoveEvalFn, CustomEvalFn, CustomPlayer, CustomPlayer_iterative
from isolation import Board, game_as_text
from test_players import RandomPlayer, HumanPlayer


def main():
    depth = 5
    test_depth = 3
    try:
        r = CustomPlayer_iterative()
        h = CustomPlayer(search_depth=depth)
        game = Board(h, r, 7, 7)
        output_b = game.copy()
        winner, move_history, termination = game.play_isolation_name_changed()
        #print winner.search_depth
        #print r.search_depth		
        if winner.search_depth == depth:
            print 'Open Eval: CustomPlayer Iterative Lost with custom second play'
            print game_as_text(winner, move_history, termination, output_b)
        else:
            print 'Open Eval: CustomPlayer Iterative Won with custom second play'
        # Uncomment to see game
    except NotImplementedError:
        print 'CustomPlayer Test: Not Implemented'
    except:
        print 'CustomPlayer Test: ERROR OCCURRED'
        print traceback.format_exc()
		
    try:
        r = CustomPlayer_iterative()
        h = CustomPlayer(search_depth=depth)
        game = Board(r, h, 7, 7)
        output_b = game.copy()
        winner, move_history, termination = game.play_isolation_name_changed()
        #print winner.search_depth
        #print r.search_depth		
        if winner.search_depth == depth:
            print 'Open Eval: CustomPlayer Iterative Lost with custom first play'
            print game_as_text(winner, move_history, termination, output_b)   
        else:
            print 'Open Eval: CustomPlayer Iterative Won with custom first play'
        # Uncomment to see game
    except NotImplementedError:
        print 'CustomPlayer Test: Not Implemented'
    except:
        print 'CustomPlayer Test: ERROR OCCURRED'
        print traceback.format_exc()

    try:
        r = CustomPlayer_iterative(eval_fn=CustomEvalFn())
        h = CustomPlayer(search_depth=depth)
        game = Board(h, r, 7, 7)
        output_b = game.copy()
        winner, move_history, termination = game.play_isolation_name_changed()
        #print winner.search_depth
        #print r.search_depth		
        if winner.search_depth == depth:
            print 'Custom Eval: CustomPlayer Iterative Lost with custom second play'
            print game_as_text(winner, move_history, termination, output_b)
        else:
            print 'Custom Eval: CustomPlayer Iterative Won with custom second play'
        # Uncomment to see game
    except NotImplementedError:
        print 'CustomPlayer Test: Not Implemented'
    except:
        print 'CustomPlayer Test: ERROR OCCURRED'
        print traceback.format_exc()
		
    try:
        r = CustomPlayer(eval_fn=CustomEvalFn())
        h = CustomPlayer(search_depth=depth)
        game = Board(r, h, 7, 7)
        output_b = game.copy()
        winner, move_history, termination = game.play_isolation_name_changed()
        #print winner.search_depth
        #print r.search_depth		
        if winner.search_depth == depth:
            print 'Custom Eval: CustomPlayer Iterative Lost with custom first play'
            print game_as_text(winner, move_history, termination, output_b)
        else:
            print 'Custom Eval: CustomPlayer Iterative Won with custom first play'
        # Uncomment to see game
    except NotImplementedError:
        print 'CustomPlayer Test: Not Implemented'
    except:
        print 'CustomPlayer Test: ERROR OCCURRED'
        print traceback.format_exc()
		
if __name__ == "__main__":
    main()
