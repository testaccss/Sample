#!/usr/bin/env python
import traceback
from player_submission import OpenMoveEvalFn, CustomEvalFn, CustomPlayer
from isolation import Board, game_as_text
from test_players import RandomPlayer, HumanPlayer


def main():
    depth = 7	
    try:
        r = CustomPlayer(search_depth=depth)
        h = RandomPlayer()
        game = Board(h, r, 7, 7)
        output_b = game.copy()
        winner, move_history, termination = game.play_isolation_name_changed()
        #print winner.search_depth
        #print r.search_depth		
        if 'CustomPlayer' in str(winner):
            print 'Random test Open Eval: CustomPlayer Won with custom second play'
        else:
            print 'Random test Open Eval: CustomPlayer Lost with custom second play'
        # Uncomment to see game
        # print game_as_text(winner, move_history, termination, output_b)
    except NotImplementedError:
        print 'CustomPlayer Test: Not Implemented'
    except:
        print 'CustomPlayer Test: ERROR OCCURRED'
        print traceback.format_exc()
		
    try:
        r = CustomPlayer(search_depth=depth)
        h = RandomPlayer()
        game = Board(r, h, 7, 7)
        output_b = game.copy()
        winner, move_history, termination = game.play_isolation_name_changed()
        #print winner.search_depth
        #print r.search_depth		
        if 'CustomPlayer' in str(winner):
            print 'Random test Open Eval: CustomPlayer Won with custom first play'
        else:
            print 'Random test Open Eval: CustomPlayer Lost with custom first play'
        # Uncomment to see game
        # print game_as_text(winner, move_history, termination, output_b)
    except NotImplementedError:
        print 'CustomPlayer Test: Not Implemented'
    except:
        print 'CustomPlayer Test: ERROR OCCURRED'
        print traceback.format_exc()

    try:
        r = CustomPlayer(eval_fn=CustomEvalFn(), search_depth=depth)
        h = RandomPlayer()
        game = Board(h, r, 7, 7)
        output_b = game.copy()
        winner, move_history, termination = game.play_isolation_name_changed()
        #print winner.search_depth
        #print r.search_depth		
        if 'CustomPlayer' in str(winner):
            print 'Random test Custom Eval: CustomPlayer Won with custom second play'
        else:
            print 'Random test Custom Eval: CustomPlayer Lost with custom second play'
        # Uncomment to see game
        # print game_as_text(winner, move_history, termination, output_b)
    except NotImplementedError:
        print 'CustomPlayer Test: Not Implemented'
    except:
        print 'CustomPlayer Test: ERROR OCCURRED'
        print traceback.format_exc()
		
    try:
        r = CustomPlayer(eval_fn=CustomEvalFn(), search_depth=depth)
        h = RandomPlayer()
        game = Board(r, h, 7, 7)
        output_b = game.copy()
        winner, move_history, termination = game.play_isolation_name_changed()
        #print winner.search_depth
        #print r.search_depth		
        if 'CustomPlayer' in str(winner):
            print 'Random test Custom Eval: CustomPlayer Won with custom first play'
        else:
            print 'Random test Custom Eval: CustomPlayer Lost with custom first play'
        # Uncomment to see game
        # print game_as_text(winner, move_history, termination, output_b)
    except NotImplementedError:
        print 'CustomPlayer Test: Not Implemented'
    except:
        print 'CustomPlayer Test: ERROR OCCURRED'
        print traceback.format_exc()
		
if __name__ == "__main__":
    main()
