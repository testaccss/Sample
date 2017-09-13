#!/usr/bin/env python
from isolation import Board, game_as_text
import math
from random import shuffle
from random import randint


# This file is your main submission that will be graded against. Do not
# add any classes or functions to this file that are not part of the classes
# that we want.

# Submission Class 1
class OpenMoveEvalFn:
    def score(self, game, maximizing_player_turn=True):
        """Score the current game state

        Evaluation function that outputs a score equal to how many
        moves are open for AI player on the board.

        Args
            param1 (Board): The board and game state.
            param2 (bool): True if maximizing player is active.

        Returns:
            float: The current state's score. Number of your agent's moves.

        """

        # TODO: finish this function!
        # raise NotImplementedError
        return len(game.get_legal_moves())


# Submission Class 2
class CustomEvalFn:
    def __init__(self):
        self.count = 0
        pass

    def get_blank_spaces(self, game):
        """
        Return a list of the locations that are still available on the board.
        """
        return len([(i, j) for j in range(game.width) for i in range(game.height)
                    if game.__board_state__[i][j] == Board.BLANK])

    def dfs(self, x, y, game, visited):

        dx = [-1, 0, 1, 1, 1, 0, -1, -1]
        dy = [1, 1, 1, 0, -1, -1, -1, 0]
        visited[x][y] = 1
        self.count = self.count + 1
        for i in range(8):
            x_ = x + dx[i]
            y_ = y + dy[i]
            if game.move_is_legal(x_, y_) and not visited[x_][y_]:
                self.dfs(x_, y_, game, visited)

    def get_moves_left(self, game):
        visited = [[0 for col in range(game.width)] for row in range(game.height)]
        move = game.__last_queen_move__
        out = game.print_board()
        x, y = move
        self.dfs(x, y, game, visited)
        out = self.count - 1
        self.count = 0
        return out


    def score(self, game, maximizing_player_turn=True):
        """Score the current game state

        Custom evaluation function that acts however you think it should. This
        is not required but highly encouraged if you want to build the best
        AI possible.

        Args
            game (Board): The board and game state.
            maximizing_player_turn (bool): True if maximizing player is active.

        Returns:
            float: The current state's score, based on your own heuristic.

        """
        # TODO: finish this function!
        # raise NotImplementedError
        score = len(game.get_legal_moves())
        if maximizing_player_turn:
            if game.move_count > 5:
                moves_left = self.get_moves_left(game)
                if moves_left + game.move_count != game.width * game.height:
                    if moves_left % 2 == 0:
                        return -100
                    else:
                        #print 'In Max:', game.__last_queen_move__
                        return 100
                else:
                    return score
            else:
                return score
        else:
            if game.move_count > 5:
                moves_left = self.get_moves_left(game)
                if moves_left + game.move_count != game.width * game.height:
                    if moves_left % 2 == 0:
                        return 100
                    else:
                        #print 'In Min:', game.__last_queen_move__
                        return -100
                else:
                    return score
            else:
                return score

class Timeout(Exception):
    pass

class CustomPlayer:
    # TODO: finish this class!
    """Player that chooses a move using
    your evaluation function and
    a minimax algorithm
    with alpha-beta pruning.
    You must finish and test this player
    to make sure it properly uses minimax
    and alpha-beta to return a good move."""

    def __init__(self, search_depth=10, eval_fn=CustomEvalFn()):
        """Initializes your player.

        if you find yourself with a superior eval function, update the default
        value of `eval_fn` to `CustomEvalFn()`

        Args:
            search_depth (int): The depth to which your agent will search
            eval_fn (function): Utility function used by your agent
        """
        self.eval_fn = eval_fn
        self.search_depth = search_depth
        self.second_player_queen_center = False
        self.use_iterative_deepening = False

    def get_blank_spaces(self, game):
        """
        Return a list of the locations that are still available on the board.
        """
        return [(i, j) for j in range(game.width) for i in range(game.height)
                if game.__board_state__[i][j] == Board.BLANK]

    def get_game_move(self, game):
        """
        Return the number of moves that has happened in the game
        """
        return game.width * game.height - len(self.get_blank_spaces(game)) + 1

    def move_is_legal_custom(self, game, legal_moves, next_move):
        is_legal = False
        if game.move_is_legal(next_move[0], next_move[1]):
            for move in legal_moves:
                if move == next_move:
                    is_legal = True
                    break
                else:
                    is_legal = False
            return is_legal
        else:
            return is_legal

    def mirrored_move(self, game, legal_moves):

        last_move = game.__last_queen_move__
        mirror_move = (last_move[1], last_move[0])
        if mirror_move in legal_moves:
            return mirror_move
        else:
            return None

    def move(self, game, legal_moves, time_left):
        """Called to determine one move by your agent

        Args:
            game (Board): The board and game state.
            legal_moves (dict): Dictionary of legal moves and their outcomes
            time_left (function): Used to determine time left before timeout

        Returns:
            (tuple): best_move
        """
        func = self.minimax
        if game.move_count == 0:
            best_move = (int(math.ceil(game.width / 2)), int(math.ceil(game.height / 2)))
        elif game.move_count == 1:  # else part implies that my agent is playing second.
            center_move = (int(math.ceil(game.width / 2)), int(math.ceil(game.height / 2)))
            if center_move in legal_moves:
                best_move = center_move
            else:
                best_move = self.get_predefined_moves(game, legal_moves)
        elif game.move_count < 4:
            if len(legal_moves) > 0:
                if self.second_player_queen_center:
                    nr_move = self.get_non_reflective_moves(game, legal_moves)
                    if nr_move:
                        return nr_move
                    else:
                        r_move = self.mirrored_move(game, legal_moves)
                        if r_move:
                            return r_move
                        else:
                            rand_move = legal_moves[randint(0, len(legal_moves) - 1)]
                            return rand_move
                else:
                    r_move = self.mirrored_move(game, legal_moves)
                    if r_move:
                        return r_move
                    else:
                        nr_move = self.get_non_reflective_moves(game, legal_moves)
                        if nr_move:
                            return nr_move
                        else:
                            rand_move = legal_moves[randint(0, len(legal_moves) - 1)]
                            return rand_move
            else:
                return None
        else:
            # print 'Move:', game.move_count, 'Depth:', depth, 'len:', len(legal_moves)
            #print 'Time Left:', time_left()
            if self.use_iterative_deepening:
                best_move = (-1,-1)
                end_depth = game.width * game.height - game.move_count
                try:
                    for depth in range(1, end_depth):
                        best_move, utility = func(game, time_left, depth=depth)
                        #print depth
                except Timeout:
                    pass
            else:
                best_move, utility = func(game, time_left, depth=self.search_depth)
            # change minimax to alphabeta after completing alphabeta part of assignment
        return best_move

    def utility(self, game, maximizing_player=True):
        """Can be updated if desired"""
        return self.eval_fn.score(game)

    def utility_custom(self, game, maximizing_player, depth, len_moves):

        #print 'Utility:', maximizing_player
        #print game.get_legal_moves()
        if maximizing_player:
            if len_moves == 0:
                #print 'Max: game.get_legal_moves()'
                return -200
        else:
            if len_moves == 0:
                #print 'Min: game.get_legal_moves()'
                return 200

        return self.eval_fn.score(game, maximizing_player)

    def eqdist(self, game_1, game_2):

        return ((game_1[0] - game_2[0]) ** 2 + (game_1[1] - game_2[1]) ** 2)

    def minimax(self, game, time_left, depth=3, maximizing_player=True):
        """Implementation of the minimax algorithm

        Args:
            game (Board): A board and game state.
            time_left (function): Used to determine time left before timeout
            depth: Used to track how deep you are in the search tree
            maximizing_player (bool): True if maximizing player is active.

        Returns:
            (tuple, int): best_move, best_val
        """
        # TODO: finish this function!
        # raise NotImplementedError
        legal_moves = game.get_legal_moves()
        len_moves = len(legal_moves)
        best_move = (-1, -1)
        best_score = self.utility_custom(game, maximizing_player, depth, len_moves)

        if depth == 0 or not legal_moves or time_left() < 40:
            #print 'Inside return'
            #print 'last queen:',game.__last_queen_move__, 'legal_moves:', legal_moves, 'depth:', depth
            return None, best_score
        if maximizing_player:
            best_score = float('-inf')
            for move in legal_moves:
                next_move = game.forecast_move(move)
                p_next_move, score = self.minimax(next_move, time_left, depth - 1, False)
                if p_next_move:
                    dist = self.eqdist(p_next_move, move)
                    score = score + dist
                if score > best_score:
                    best_score = score
                    best_move = move
        else:
            best_score = float('inf')
            for move in legal_moves:
                next_move = game.forecast_move(move)
                p_next_move, score = self.minimax(next_move, time_left, depth - 1, True)
                if p_next_move:
                    dist = self.eqdist(p_next_move, move)
                    score = score - dist
                if score < best_score:
                    best_score = score
                    best_move = move

        return best_move, best_score

    def minimax_id(self, game, time_left, depth=3, maximizing_player=True):
        """Implementation of the minimax algorithm

        Args:
            game (Board): A board and game state.
            time_left (function): Used to determine time left before timeout
            depth: Used to track how deep you are in the search tree
            maximizing_player (bool): True if maximizing player is active.

        Returns:
            (tuple, int): best_move, best_val
        """
        # TODO: finish this function!
        # raise NotImplementedError
        legal_moves = game.get_legal_moves()
        len_moves = len(legal_moves)
        best_move = (-1, -1)
        best_score = self.utility_custom(game, maximizing_player, depth, len_moves)

        if time_left() < 30:
            raise Timeout()

        if depth == 0 or not legal_moves:
            #print 'Inside return'
            #print 'last queen:',game.__last_queen_move__, 'legal_moves:', legal_moves, 'depth:', depth
            return None, best_score
        if maximizing_player:
            best_score = float('-inf')
            for move in legal_moves:
                next_move = game.forecast_move(move)
                p_next_move, score = self.minimax_id(next_move, time_left, depth - 1, False)
                if p_next_move:
                    dist = self.eqdist(p_next_move, move)
                    score = score + dist
                if score > best_score:
                    best_score = score
                    best_move = move
        else:
            best_score = float('inf')
            for move in legal_moves:
                next_move = game.forecast_move(move)
                p_next_move, score = self.minimax_id(next_move, time_left, depth - 1, True)
                if p_next_move:
                    dist = self.eqdist(p_next_move, move)
                    score = score - dist
                if score < best_score:
                    best_score = score
                    best_move = move

        return best_move, best_score

    def get_non_reflective_moves(self, game, legal_moves):
        if game.width == 7 and game.height == 7:
            moves = [(0, 1), (0, 2), (0, 4), (0, 5), (1, 0), (1, 6), (2, 0), (2, 6), (4, 0), (4, 6), (5, 0), (5, 6),
                     (6, 1), (6, 2), (6, 4), (6, 5), (1, 2), (1, 4), (2, 1), (2, 5), (4, 1), (4, 5), (5, 2), (5, 4)]
        elif game.width == 5 and game.height == 5:
            moves = [(0, 1), (0, 3), (1, 0), (1, 4), (3, 0), (3, 4), (4, 1), (4, 3)]
        else:
            return None
        shuffle(moves)
        for move in moves:
            if move in legal_moves:
                return move
        return None

    def get_predefined_moves(self, game, legal_moves):
        #moves = [(2, 3), (3, 2), (3, 4), (4,3)]
        #moves = [(1, 1), (1, game.width - 2), (game.height - 2, 1), (game.height - 2, game.height - 2)]
        shuffle(legal_moves)
        return legal_moves[0]

    def alphabeta(self, game, time_left, depth=3, alpha=float("-inf"), beta=float("inf"),
                  maximizing_player=True):
        """Implementation of the alphabeta algorithm

        Args:
            game (Board): A board and game state.
            time_left (function): Used to determine time left before timeout
            depth: Used to track how deep you are in the search tree
            alpha (float): Alpha value for pruning
            beta (float): Beta value for pruning
            maximizing_player (bool): True if maximizing player is active.

        Returns:
            (tuple, int): best_move, best_val
        """
        # TODO: finish this function!
        # raise NotImplementedError
        legal_moves = game.get_legal_moves()
        len_moves = len(legal_moves)
        best_move = (-1, -1)
        best_score = self.utility_custom(game, maximizing_player, depth, len_moves)

        if depth == 0 or not legal_moves or time_left() < 40:
            #print 'Inside return'
            #print 'last queen:',game.__last_queen_move__, 'legal_moves:', legal_moves, 'depth:', depth
            return best_move, best_score
        #print 'Depth:', depth
        if maximizing_player:
            #print 'Max:'
            best_score = float("-inf")
            for move in legal_moves:
                next_move = game.forecast_move(move)
                #print move, '->', next_move.__last_queen_move__
                _, score = self.alphabeta(next_move, time_left, depth - 1, alpha, beta, False)
                #print move, '->', next_move.get_legal_moves(), 'score:', score
                if score > best_score:
                    best_score = score
                    best_move = move
                if best_score >= beta:
                    return best_move, best_score
                alpha = max(alpha, best_score)
        else:
            #print 'Min:'
            best_score = float("inf")
            for move in legal_moves:
                next_move = game.forecast_move(move)
                #print move, '->', next_move.__last_queen_move__
                _, score = self.alphabeta(next_move, time_left, depth - 1, alpha, beta, True)
                #print move, '->', next_move.get_legal_moves(), 'score:', score
                if score < best_score:
                    best_score = score
                    best_move = move
                if alpha >= best_score:
                    return best_move, best_score
                beta = min(beta, best_score)

        return best_move, best_score

    def alphabeta_id(self, game, time_left, depth=3, alpha=float("-inf"), beta=float("inf"),
                  maximizing_player=True):
        """Implementation of the alphabeta algorithm

        Args:
            game (Board): A board and game state.
            time_left (function): Used to determine time left before timeout
            depth: Used to track how deep you are in the search tree
            alpha (float): Alpha value for pruning
            beta (float): Beta value for pruning
            maximizing_player (bool): True if maximizing player is active.

        Returns:
            (tuple, int): best_move, best_val
        """
        # TODO: finish this function!
        # raise NotImplementedError
        legal_moves = game.get_legal_moves()
        len_moves = len(legal_moves)
        best_move = (-1, -1)
        best_score = self.utility_custom(game, maximizing_player, depth, len_moves)

        if time_left() < 30:
            raise Timeout()

        if depth == 0 or not legal_moves:
            #print 'Inside return'
            #print 'last queen:',game.__last_queen_move__, 'legal_moves:', legal_moves, 'depth:', depth
            return best_move, best_score
        #print 'Depth:', depth
        if maximizing_player:
            #print 'Max:'
            best_score = float("-inf")
            for move in legal_moves:
                next_move = game.forecast_move(move)
                #print move, '->', next_move.__last_queen_move__
                _, score = self.alphabeta_id(next_move, time_left, depth - 1, alpha, beta, False)
                #print move, '->', next_move.get_legal_moves(), 'score:', score
                if score > best_score:
                    best_score = score
                    best_move = move
                if best_score >= beta:
                    return best_move, best_score
                alpha = max(alpha, best_score)
        else:
            #print 'Min:'
            best_score = float("inf")
            for move in legal_moves:
                next_move = game.forecast_move(move)
                #print move, '->', next_move.__last_queen_move__
                _, score = self.alphabeta_id(next_move, time_left, depth - 1, alpha, beta, True)
                #print move, '->', next_move.get_legal_moves(), 'score:', score
                if score < best_score:
                    best_score = score
                    best_move = move
                if alpha >= best_score:
                    return best_move, best_score
                beta = min(beta, best_score)

        return best_move, best_score