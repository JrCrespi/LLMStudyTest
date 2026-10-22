/*
 * Exact Connect 4 solver (negamax + alpha-beta, null-window search,
 * transposition table, threat-based move ordering), following the method
 * described by Pascal Pons in "Solving Connect 4: how to build a perfect AI".
 *
 * Bitboard layout matches connect4_zero/game.py: bit (col * 7 + row), row 0
 * at the bottom, one sentinel bit on top of each column.
 *
 * Score convention (from the side to move):
 *   0   draw
 *   > 0 win; (43 - n) / 2 where n is the number of stones after the winning move
 *   < 0 loss, symmetric.
 *
 * Build: gcc -O3 -march=native -shared -fPIC -o libc4solver.so c4solver.c
 */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define W 7
#define H 6
#define CELLS (W * H)
#define MIN_SCORE (-(W * H) / 2 + 3)
#define MAX_SCORE ((W * H + 1) / 2 - 3)
#define TT_SIZE 8388617u /* prime */

typedef uint64_t u64;

static u64 tt_keys[TT_SIZE];
static int8_t tt_vals[TT_SIZE];
static u64 node_count;

static inline u64 bottom_row(void) {
    u64 m = 0;
    for (int c = 0; c < W; c++) m |= 1ULL << (c * (H + 1));
    return m;
}
static inline u64 board_mask(void) { return bottom_row() * ((1ULL << H) - 1); }
static inline u64 column_mask(int c) { return ((1ULL << H) - 1) << (c * (H + 1)); }

static inline int popcount(u64 m) { return __builtin_popcountll(m); }

static inline u64 winning_cells(u64 pos, u64 mask) {
    u64 r = (pos << 1) & (pos << 2) & (pos << 3);
    u64 p;
    /* horizontal */
    p = (pos << (H + 1)) & (pos << 2 * (H + 1));
    r |= p & (pos << 3 * (H + 1));
    r |= p & (pos >> (H + 1));
    p = (pos >> (H + 1)) & (pos >> 2 * (H + 1));
    r |= p & (pos << (H + 1));
    r |= p & (pos >> 3 * (H + 1));
    /* diagonal 1 */
    p = (pos << H) & (pos << 2 * H);
    r |= p & (pos << 3 * H);
    r |= p & (pos >> H);
    p = (pos >> H) & (pos >> 2 * H);
    r |= p & (pos << H);
    r |= p & (pos >> 3 * H);
    /* diagonal 2 */
    p = (pos << (H + 2)) & (pos << 2 * (H + 2));
    r |= p & (pos << 3 * (H + 2));
    r |= p & (pos >> (H + 2));
    p = (pos >> (H + 2)) & (pos >> 2 * (H + 2));
    r |= p & (pos << (H + 2));
    r |= p & (pos >> 3 * (H + 2));
    return r & (board_mask() ^ mask);
}

static inline u64 possible(u64 mask) { return (mask + bottom_row()) & board_mask(); }

static inline u64 non_losing_moves(u64 cur, u64 mask) {
    u64 poss = possible(mask);
    u64 opp_win = winning_cells(cur ^ mask, mask);
    u64 forced = poss & opp_win;
    if (forced) {
        if (forced & (forced - 1)) return 0; /* two threats: lost */
        poss = forced;
    }
    return poss & ~(opp_win >> 1); /* never play directly below an opponent threat */
}

static const int column_order[W] = {3, 2, 4, 1, 5, 0, 6};

static int negamax(u64 cur, u64 mask, int moves, int alpha, int beta) {
    node_count++;
    u64 next = non_losing_moves(cur, mask);
    if (next == 0) return -(CELLS - moves) / 2;
    if (moves >= CELLS - 2) return 0;

    int min = -(CELLS - 2 - moves) / 2;
    if (alpha < min) {
        alpha = min;
        if (alpha >= beta) return alpha;
    }
    int max = (CELLS - 1 - moves) / 2;
    u64 key = cur + mask;
    unsigned idx = (unsigned)(key % TT_SIZE);
    if (tt_keys[idx] == key) max = tt_vals[idx] + MIN_SCORE - 1;
    if (beta > max) {
        beta = max;
        if (alpha >= beta) return beta;
    }

    /* order moves by the number of threats they create, centre first on ties */
    u64 mv[W];
    int sc[W], n = 0;
    for (int i = 0; i < W; i++) {
        u64 m = next & column_mask(column_order[i]);
        if (!m) continue;
        int s = popcount(winning_cells(cur | m, mask));
        int j = n++;
        while (j > 0 && sc[j - 1] < s) {
            mv[j] = mv[j - 1];
            sc[j] = sc[j - 1];
            j--;
        }
        mv[j] = m;
        sc[j] = s;
    }
    for (int i = 0; i < n; i++) {
        u64 ncur = cur ^ mask;
        u64 nmask = mask | mv[i];
        int score = -negamax(ncur, nmask, moves + 1, -beta, -alpha);
        if (score >= beta) return score;
        if (score > alpha) alpha = score;
    }
    tt_keys[idx] = key;
    tt_vals[idx] = (int8_t)(alpha - MIN_SCORE + 1);
    return alpha;
}

/* Exact score of the position for the side to move. */
int c4_solve(u64 cur, u64 mask, int moves) {
    if (winning_cells(cur, mask) & possible(mask)) return (CELLS + 1 - moves) / 2;
    int min = -(CELLS - moves) / 2;
    int max = (CELLS + 1 - moves) / 2;
    while (min < max) {
        int med = min + (max - min) / 2;
        if (med <= 0 && min / 2 < med) med = min / 2;
        else if (med >= 0 && max / 2 > med) med = max / 2;
        int r = negamax(cur, mask, moves, med, med + 1);
        if (r <= med) max = r;
        else min = r;
    }
    return min;
}

void c4_reset(void) {
    memset(tt_keys, 0, sizeof(tt_keys));
    node_count = 0;
}

u64 c4_nodes(void) { return node_count; }
