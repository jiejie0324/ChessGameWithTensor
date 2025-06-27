#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tensor-Chess CLI — now with player IDs, score tracking, and replay option.
Supports pawns and rooks only; kings can be captured to win the game.
"""

import tensorflow as tf

# ===== 1. Piece encoding ======================================================

piece2idx = {
    'wP': 1, 'wR': 2, 'wN': 3, 'wB': 4, 'wQ': 5, 'wK': 6,
    'bP': 7, 'bR': 8, 'bN': 9, 'bB':10, 'bQ':11, 'bK':12
}
idx2str = {v: k for k, v in piece2idx.items()}
idx2str[0] = '..'

# ===== Castling rights tracking ==============================================
white_king_moved = False
black_king_moved = False
white_rook_moved_k = False  # kingside
white_rook_moved_q = False  # queenside
black_rook_moved_k = False
black_rook_moved_q = False

# ===== 2. Board construction ==================================================

def create_initial_board_tf():
    board = tf.Variable(tf.zeros([8, 8], dtype=tf.int32))

    def fill_rank(rank_idx, pieces):
        for col, p in enumerate(pieces):
            board[rank_idx, col].assign(piece2idx[p])

    fill_rank(0, ['bR', 'bN', 'bB', 'bQ', 'bK', 'bB', 'bN', 'bR'])
    fill_rank(1, ['bP'] * 8)
    fill_rank(6, ['wP'] * 8)
    fill_rank(7, ['wR', 'wN', 'wB', 'wQ', 'wK', 'wB', 'wN', 'wR'])
    return board

# ===== 3. Utility helpers =====================================================

def get_piece(board, r, c):
    return int(board[r, c].numpy())

def set_piece(board, r, c, code):
    board[r, c].assign(code)

def pos_to_coord(pos):
    return 8 - int(pos[1]), ord(pos[0]) - ord('a')

def coord_to_pos(r, c):
    return chr(c + ord('a')) + str(8 - r)

def in_bounds(r, c):
    return 0 <= r < 8 and 0 <= c < 8

def king_exists(board, white: bool) -> bool:
    target = 6 if white else 12
    for r in range(8):
        for c in range(8):
            if get_piece(board, r, c) == target:
                return True
    return False

# ===== 4. Move generation  ======================================

def get_possible_moves(board, row, col):
    code = get_piece(board, row, col)
    if code == 0:
        return []

    white = code <= 6
    ptype = code if white else code - 6
    moves = []

    def is_enemy(r, c):
        t = get_piece(board, r, c)
        return t != 0 and (t <= 6) != white

    if ptype == 1:  # Pawn
        d = -1 if white else 1
        start_row = 6 if white else 1
        if in_bounds(row + d, col) and get_piece(board, row + d, col) == 0:
            moves.append((row + d, col))
            if row == start_row and get_piece(board, row + 2 * d, col) == 0:
                moves.append((row + 2 * d, col))
        for dc in (-1, 1):
            r, c = row + d, col + dc
            if in_bounds(r, c) and is_enemy(r, c):
                moves.append((r, c))
                # En Passant check
        if en_passant_target:
            er, ec = en_passant_target
            if row == (3 if white else 4) and abs(ec - col) == 1 and er == row + d:
                moves.append((er, ec))

    elif ptype == 2:  # Rook
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            r, c = row + dr, col + dc
            while in_bounds(r, c):
                t = get_piece(board, r, c)
                if t == 0:
                    moves.append((r, c))
                else:
                    if (t <= 6) != white:
                        moves.append((r, c))
                    break
                r += dr
                c += dc

    elif ptype == 4:  # Bishop
        for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
            r, c = row + dr, col + dc
            while in_bounds(r, c):
                t = get_piece(board, r, c)
                if t == 0:
                    moves.append((r, c))
                else:
                    if (t <= 6) != white:
                        moves.append((r, c))
                    break
                r += dr
                c += dc

    elif ptype == 5:  # Queen
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
            r, c = row + dr, col + dc
            while in_bounds(r, c):
                t = get_piece(board, r, c)
                if t == 0:
                    moves.append((r, c))
                else:
                    if (t <= 6) != white:
                        moves.append((r, c))
                    break
                r += dr
                c += dc

    elif ptype == 6:  # King
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                r, c = row + dr, col + dc
                if in_bounds(r, c):
                    t = get_piece(board, r, c)
                    if t == 0 or (t <= 6) != white:
                        moves.append((r, c))

        # Debugging castling state
        sr, sc = row, col
        print("Castling check for", "white" if white else "black")
        print("King moved:", white_king_moved if white else black_king_moved)
        print("Rook moved (k/q):", white_rook_moved_k if white else black_rook_moved_k, "|", white_rook_moved_q if white else black_rook_moved_q)
        print("Rook positions:", get_piece(board, sr, 7), get_piece(board, sr, 0))

        if white and not white_king_moved:
            if not white_rook_moved_k and get_piece(board, sr, 7) == 2 and get_piece(board, sr, 5) == 0 and get_piece(board, sr, 6) == 0:
                moves.append((sr, 6))
            if not white_rook_moved_q and get_piece(board, sr, 0) == 2 and get_piece(board, sr, 1) == 0 and get_piece(board, sr, 2) == 0 and get_piece(board, sr, 3) == 0:
                moves.append((sr, 2))
        elif not white and not black_king_moved:
            if not black_rook_moved_k and get_piece(board, sr, 7) == 8 and get_piece(board, sr, 5) == 0 and get_piece(board, sr, 6) == 0:
                moves.append((sr, 6))
            if not black_rook_moved_q and get_piece(board, sr, 0) == 8 and get_piece(board, sr, 1) == 0 and get_piece(board, sr, 2) == 0 and get_piece(board, sr, 3) == 0:
                moves.append((sr, 2))

    elif ptype == 3:  # Knight
        for dr, dc in [(-2, -1), (-2, 1), (-1, -2), (-1, 2),
                       (1, -2), (1, 2), (2, -1), (2, 1)]:
            r, c = row + dr, col + dc
            if in_bounds(r, c):
                t = get_piece(board, r, c)
                if t == 0 or (t <= 6) != white:
                    moves.append((r, c))

    return moves


# ===== 5. Special move functions ==============================================

en_passant_target = None

def is_castling_move(board, sr, sc, er, ec, white_to_move):
    if sr != (7 if white_to_move else 0) or sc != 4:
        return False
    if er != sr or abs(ec - sc) != 2:
        return False

    if white_to_move:
        if white_king_moved:
            return False
        if ec == 6:
            if white_rook_moved_k or get_piece(board, sr, 5) != 0 or get_piece(board, sr, 6) != 0:
                return False
            return get_piece(board, sr, 7) == 2
        elif ec == 2:
            if white_rook_moved_q or get_piece(board, sr, 1) != 0 or get_piece(board, sr, 2) != 0 or get_piece(board, sr, 3) != 0:
                return False
            return get_piece(board, sr, 0) == 2
    else:
        if black_king_moved:
            return False
        if ec == 6:
            if black_rook_moved_k or get_piece(board, sr, 5) != 0 or get_piece(board, sr, 6) != 0:
                return False
            return get_piece(board, sr, 7) == 8
        elif ec == 2:
            if black_rook_moved_q or get_piece(board, sr, 1) != 0 or get_piece(board, sr, 2) != 0 or get_piece(board, sr, 3) != 0:
                return False
            return get_piece(board, sr, 0) == 8

    return False

def handle_castling(board, sr, sc, er, ec):
    global white_king_moved, black_king_moved
    global white_rook_moved_k, white_rook_moved_q
    global black_rook_moved_k, black_rook_moved_q

    white = get_piece(board, sr, sc) <= 6
    set_piece(board, er, ec, get_piece(board, sr, sc))
    set_piece(board, sr, sc, 0)

    if ec > sc:
        rook = get_piece(board, sr, 7)
        set_piece(board, sr, 5, rook)
        set_piece(board, sr, 7, 0)
        if white:
            white_rook_moved_k = True
        else:
            black_rook_moved_k = True
    else:
        rook = get_piece(board, sr, 0)
        set_piece(board, sr, 3, rook)
        set_piece(board, sr, 0, 0)
        if white:
            white_rook_moved_q = True
        else:
            black_rook_moved_q = True

    if white:
        white_king_moved = True
    else:
        black_king_moved = True

def is_en_passant(board, sr, sc, er, ec, white_to_move):
    global en_passant_target
    piece = get_piece(board, sr, sc)
    if piece != (1 if white_to_move else 7):
        return False
    return (er, ec) == en_passant_target

def handle_en_passant(board, sr, sc, er, ec, white_to_move):
    set_piece(board, er, ec, get_piece(board, sr, sc))
    set_piece(board, sr, sc, 0)
    set_piece(board, sr, ec, 0)

def handle_promotion(board, r, c):
    piece = get_piece(board, r, c)
    if piece == 1 and r == 0:
        set_piece(board, r, c, 5)
    elif piece == 7 and r == 7:
        set_piece(board, r, c, 11)
# ===== 6. Display + Game Loop =================================================

def print_board_tf(board):
    header = "  " + "  ".join("abcdefgh")
    print(header)
    for i in range(8):
        rank = 8 - i
        row = f"{rank} "
        for j in range(8):
            row += f"{idx2str[get_piece(board,i,j)]:<2} "
        print(row + f"{rank}")
    print(header)

# ==== main game logic =========================================================

def play_game(white_id, black_id, scores):
    global en_passant_target
    global white_king_moved, black_king_moved
    global white_rook_moved_k, white_rook_moved_q, black_rook_moved_k, black_rook_moved_q

    board = create_initial_board_tf()
    white_to_move = True
    en_passant_target = None
    white_king_moved = black_king_moved = False
    white_rook_moved_k = white_rook_moved_q = False
    black_rook_moved_k = black_rook_moved_q = False

    while True:
        print_board_tf(board)
        side_id = white_id if white_to_move else black_id
        prompt = f"{side_id} ({'White' if white_to_move else 'Black'}) to move: "
        start = input(prompt).strip()

        try:
            sr, sc = pos_to_coord(start)
            scode = get_piece(board, sr, sc)
            if scode == 0 or (scode <= 6) != white_to_move:
                print("That square doesn't contain your piece.")
                continue

            moves = get_possible_moves(board, sr, sc)
            if not moves:
                print("No legal moves for this piece.")
                continue

            print("Legal moves:")
            for i, (r, c) in enumerate(moves):
                print(f"{i+1}. {coord_to_pos(r,c)}")
            idx = int(input("Choose move # ")) - 1
            er, ec = moves[idx]

            actor = white_id if white_to_move else black_id
            opponent = black_id if white_to_move else white_id
            target_piece = get_piece(board, er, ec)

            if target_piece != 0:
                piece_score = {1:1, 2:3, 3:3, 4:5, 5:9, 6:0,
                               7:1, 8:3, 9:3,10:5,11:9,12:0}
                scores[actor]["points"] += piece_score.get(target_piece, 0)

            if is_castling_move(board, sr, sc, er, ec, white_to_move):
                handle_castling(board, sr, sc, er, ec)
                scores[actor]["points"] += 3
            elif is_en_passant(board, sr, sc, er, ec, white_to_move):
                handle_en_passant(board, sr, sc, er, ec, white_to_move)
                scores[actor]["points"] += 2
            else:
                set_piece(board, er, ec, scode)
                set_piece(board, sr, sc, 0)
                
                if scode == 2:  # white rook
                    if sr == 7 and sc == 0:
                        white_rook_moved_q = True
                    elif sr == 7 and sc == 7:
                        white_rook_moved_k = True
                elif scode == 8:  # black rook
                    if sr == 0 and sc == 0:
                        black_rook_moved_q = True
                    elif sr == 0 and sc == 7:
                        black_rook_moved_k = True

                if scode == 6:
                    white_king_moved = True
                elif scode == 12:
                    black_king_moved = True
                elif scode == 2 and sr == 7 and sc == 0:
                    white_rook_moved_q = True
                elif scode == 2 and sr == 7 and sc == 7:
                    white_rook_moved_k = True
                elif scode == 8 and sr == 0 and sc == 0:
                    black_rook_moved_q = True
                elif scode == 8 and sr == 0 and sc == 7:
                    black_rook_moved_k = True

            before = get_piece(board, er, ec)
            handle_promotion(board, er, ec)
            after = get_piece(board, er, ec)
            if before != after:
                scores[actor]["points"] += 5

            if scode in [1, 7] and abs(er - sr) == 2:
                en_passant_target = ((sr + er) // 2, sc)
            else:
                en_passant_target = None

            if not king_exists(board, not white_to_move):
                winner = white_id if white_to_move else black_id
                scores[winner]["wins"] += 1
                scores[winner]["points"] += 10
                print_board_tf(board)
                print(f"\nCheckmate! {winner} wins.")
                print("Current scoreboard:")
                for pid, sc in scores.items():
                    print(f"  {pid}: {sc['wins']} win(s), {sc['points']} point(s)")
                return

            white_to_move = not white_to_move

        except (ValueError, IndexError):
            print("Invalid input, try again.\n")

def main():
    print("=== Tensor-Chess with score tracking ===")
    p1 = input("Player 1, enter your ID (will start as White): ").strip()
    p2 = input("Player 2, enter your ID (will start as Black): ").strip()
    scores = {p1: {"wins": 0, "points": 0}, p2: {"wins": 0, "points": 0}}
    white_id, black_id = p1, p2

    while True:
        play_game(white_id, black_id, scores)
        again = input("\nPlay again? (y/n): ").strip().lower()
        if again != 'y':
            print("Thanks for playing!")
            break
        white_id, black_id = black_id, white_id
        print("\n=== Colors swapped! New game starts. ===\n")

if __name__ == "__main__":
    main()