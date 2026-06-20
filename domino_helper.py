"""
DOMINO HELPER
=============
Tracks a domino game (double-six by default, but configurable) and helps you:

  1. Know exactly how many tiles are left, unseen, for each number (0..max).
  2. Estimate which tiles each opponent is most likely still holding, based on
     what's been played and what they've passed on (a pass when a number N
     was open means that player has ZERO tiles with N left).

HOW THE PREDICTION WORKS
-------------------------
Every tile not yet seen (not in your hand, not played) is "unseen". For each
unseen tile, we figure out which players COULD still have it (anyone who
hasn't proven they lack one of its two numbers via a pass). We then split
the probability of holding that tile among those eligible players, weighted
by how many tiles are still in their hand. Players are ranked tile-by-tile,
so the more passes you log, the sharper the prediction gets.

This is a heuristic (not a full combinatorial simulation) but it's exactly
the kind of reasoning a sharp player does in their head -- except the
computer never miscounts.

RUN IT
------
    python3 domino_helper.py

Then type commands at the prompt. Type 'help' to see all commands at any
time, and 'tutorial' for a guided example.
"""

import re
import sys
from itertools import combinations_with_replacement


# --------------------------------------------------------------------------
# Tile helpers
# --------------------------------------------------------------------------

def parse_tile(s):
    """Parse a tile string like '3-5', '3,5', '6-6', or shorthand '35'."""
    s = s.strip().replace(" ", "")
    m = re.match(r"^(\d+)[-,_:](\d+)$", s)
    if not m:
        m = re.match(r"^(\d)(\d)$", s)  # shorthand like "35" or "66"
    if not m:
        raise ValueError(f"Can't parse tile '{s}'. Use a format like 3-5 or 6-6.")
    a, b = int(m.group(1)), int(m.group(2))
    return (min(a, b), max(a, b))


def tile_str(t):
    return f"{t[0]}-{t[1]}"


# --------------------------------------------------------------------------
# Core game state
# --------------------------------------------------------------------------

class DominoHelper:
    def __init__(self, max_pip=6):
        self.max_pip = max_pip
        self.all_tiles = set(combinations_with_replacement(range(max_pip + 1), 2))
        self.my_hand = set()
        self.played = []          # list of (tile, player_name) in order played
        self.players = []         # ordered list of OPPONENT names (you excluded)
        self.hand_count = {}      # player -> tiles still in hand
        self.boneyard = 0         # tiles in the draw pile, location unknown
        self.constraints = {}     # player -> set of numbers they're known to lack

    # ---- setup -----------------------------------------------------------

    def set_size(self, max_pip):
        self.__init__(max_pip)

    def set_players(self, names):
        self.players = list(names)
        for p in self.players:
            self.hand_count.setdefault(p, 0)
            self.constraints.setdefault(p, set())

    def set_hand(self, tiles):
        for t in tiles:
            if t not in self.all_tiles:
                raise ValueError(f"{tile_str(t)} is not a valid tile for a "
                                  f"double-{self.max_pip} set.")
        self.my_hand = set(tiles)

    def auto_deal(self, boneyard=0):
        """Evenly split the remaining tiles among opponents, after removing
        your hand and the requested boneyard size."""
        if not self.players:
            raise ValueError("Set players first ('players P1 P2 P3').")
        remaining = len(self.all_tiles) - len(self.my_hand) - boneyard
        n = len(self.players)
        if remaining < 0 or remaining % 1 != 0:
            raise ValueError("Not enough tiles to deal with these settings.")
        per, extra = divmod(remaining, n)
        for i, p in enumerate(self.players):
            self.hand_count[p] = per + (1 if i < extra else 0)
        self.boneyard = boneyard

    # ---- live updates ------------------------------------------------------

    def mark_play(self, player, tile):
        if player not in self.players:
            raise ValueError(f"Unknown player '{player}'. Players: {self.players}")
        if tile not in self.all_tiles:
            raise ValueError(f"{tile_str(tile)} is not a valid tile for a "
                              f"double-{self.max_pip} set.")
        if tile in {t for t, _ in self.played}:
            raise ValueError(f"{tile_str(tile)} was already marked as played.")
        if tile in self.my_hand:
            raise ValueError(f"{tile_str(tile)} is in YOUR hand, not theirs.")
        self.played.append((tile, player))
        self.hand_count[player] = max(0, self.hand_count[player] - 1)

    def mark_my_play(self, tile):
        if tile not in self.my_hand:
            raise ValueError(f"{tile_str(tile)} is not in your recorded hand.")
        self.my_hand.discard(tile)
        self.played.append((tile, "ME"))

    def mark_pass(self, player, ends):
        if player not in self.players:
            raise ValueError(f"Unknown player '{player}'. Players: {self.players}")
        self.constraints.setdefault(player, set()).update(ends)

    def mark_draw(self, player):
        """Player draws a tile from the boneyard (draw variants only)."""
        if self.boneyard <= 0:
            raise ValueError("Boneyard is empty.")
        self.boneyard -= 1
        self.hand_count[player] = self.hand_count.get(player, 0) + 1

    # ---- derived info ------------------------------------------------------

    def unseen_tiles(self):
        played_set = {t for t, _ in self.played}
        return self.all_tiles - self.my_hand - played_set

    def category_counts(self):
        """How many unseen tiles still contain each number."""
        unseen = self.unseen_tiles()
        return {n: sum(1 for t in unseen if n in t) for n in range(self.max_pip + 1)}

    def eligible_players_for(self, tile):
        a, b = tile
        elig = [p for p in self.players
                if a not in self.constraints.get(p, set())
                and b not in self.constraints.get(p, set())]
        return elig

    def predict(self, player=None):
        """Return {tile: probability} for one player, or for everyone if
        player is None -> {player: [(tile, prob), ...]} sorted descending."""
        unseen = self.unseen_tiles()
        per_tile_probs = {}  # tile -> {player: prob}

        for t in unseen:
            elig = self.eligible_players_for(t)
            weights = {p: self.hand_count.get(p, 0) for p in elig}
            if self.boneyard > 0:
                weights["(boneyard)"] = self.boneyard
            total = sum(weights.values())
            if total == 0:
                # No info / inconsistent data -> split evenly as a fallback
                pool = elig + (["(boneyard)"] if self.boneyard > 0 else [])
                if not pool:
                    continue
                per_tile_probs[t] = {p: 1 / len(pool) for p in pool}
            else:
                per_tile_probs[t] = {p: w / total for p, w in weights.items()}

        if player is not None:
            result = [(t, probs.get(player, 0.0)) for t, probs in per_tile_probs.items()]
            result = [r for r in result if r[1] > 0]
            result.sort(key=lambda x: -x[1])
            return result

        by_player = {p: [] for p in self.players}
        by_player["(boneyard)"] = []
        for t, probs in per_tile_probs.items():
            for p, pr in probs.items():
                if pr > 0:
                    by_player[p].append((t, pr))
        for p in by_player:
            by_player[p].sort(key=lambda x: -x[1])
        return by_player

    def sanity_check(self):
        """Returns a warning string if the numbers don't add up, else None."""
        unseen = len(self.unseen_tiles())
        accounted = sum(self.hand_count.get(p, 0) for p in self.players) + self.boneyard
        if unseen != accounted:
            return (f"⚠ Mismatch: {unseen} unseen tile(s) but hand counts + "
                     f"boneyard add up to {accounted}. Check your 'deal'/'sethand' "
                     f"values or recent plays.")
        return None


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

HELP = """
COMMANDS
  size <n>                Set max pip number (6 = double-six, default; 9 = double-nine)
  players <p1> <p2> ...   Set opponent names (don't include yourself)
  hand <t1> <t2> ...      Set YOUR hand, e.g.  hand 0-0 1-2 3-3 4-6
  deal [boneyard]         Auto-split remaining tiles evenly among opponents
                          (optionally reserve <boneyard> tiles as a draw pile)
  sethand <player> <n>    Manually set/correct a player's tile count

  play <player> <tile>    Mark an opponent's tile as played, e.g. play Karim 3-5
  myplay <tile>           Mark a tile from YOUR hand as played
  pass <player> <n...>    Mark a player passed when these numbers were open,
                          e.g.  pass Karim 3 5   (means Karim has no 3 and no 5)
  draw <player>           Player draws from the boneyard (draw variants only)

  status                  Show full game status (counts, hands, warnings)
  categories               Show how many unseen tiles remain for each number
  predict [player]        Show most likely tiles for one player, or everyone
  unseen                  List all unseen tiles
  board                   List everything played so far, in order
  hand                    Show your current hand
  tutorial                Walk through a quick example
  reset                   Start over
  help                    Show this message
  quit / exit             Leave
"""

TUTORIAL = """
TUTORIAL (double-six, you + 3 friends)
---------------------------------------
1) size 6
2) players Karim Sami Yasmine
3) hand 0-0 1-2 2-2 3-4 4-5 5-6 6-6
4) deal                      <- splits the remaining 21 tiles 7/7/7
5) play Karim 1-1
6) pass Sami 1                <- Sami had no 1 when a "1" end was open
7) status
8) predict Sami
9) categories
That's it -- keep logging 'play' and 'pass' as the round goes on, and check
'predict' or 'categories' whenever you need to decide your next move.
"""


def fmt_pct(p):
    return f"{p*100:5.1f}%"


def cmd_status(g):
    print(f"Set: double-{g.max_pip}  |  Total tiles: {len(g.all_tiles)}")
    print(f"Your hand ({len(g.my_hand)}): "
          + (", ".join(tile_str(t) for t in sorted(g.my_hand)) or "(empty)"))
    print(f"Boneyard: {g.boneyard}")
    print("Opponents:")
    for p in g.players:
        lacks = sorted(g.constraints.get(p, set()))
        lacks_str = f" | known to lack: {lacks}" if lacks else ""
        print(f"  - {p}: {g.hand_count.get(p, 0)} tile(s) in hand{lacks_str}")
    print(f"Unseen tiles remaining: {len(g.unseen_tiles())}")
    warn = g.sanity_check()
    if warn:
        print(warn)


def cmd_categories(g):
    counts = g.category_counts()
    print("Unseen tiles remaining, by number:")
    for n, c in counts.items():
        bar = "█" * c
        print(f"  {n}: {c:2d}  {bar}")


def cmd_predict(g, player):
    if player:
        if player not in g.players:
            print(f"Unknown player '{player}'. Players: {g.players}")
            return
        rows = g.predict(player)
        if not rows:
            print(f"No tiles can be assigned to {player} with current info "
                  f"(or {player} has 0 tiles left).")
            return
        print(f"Most likely tiles for {player} (top {g.hand_count.get(player,0)} "
              f"highlighted, * marks them):")
        for i, (t, pr) in enumerate(rows):
            mark = "*" if i < g.hand_count.get(player, 0) else " "
            print(f"  {mark} {tile_str(t):>5}  {fmt_pct(pr)}")
    else:
        by_player = g.predict(None)
        for p in g.players:
            rows = by_player.get(p, [])[:g.hand_count.get(p, 0)] or by_player.get(p, [])[:5]
            print(f"\n{p} (has {g.hand_count.get(p,0)} tile(s)) — most likely:")
            if not rows:
                print("  (no candidates / no tiles left)")
            for t, pr in rows:
                print(f"    {tile_str(t):>5}  {fmt_pct(pr)}")


def main():
    g = DominoHelper(max_pip=6)
    print("=== Domino Helper ===")
    print("Type 'tutorial' for a guided example, or 'help' for all commands.\n")

    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        parts = line.split()
        cmd, args = parts[0].lower(), parts[1:]

        try:
            if cmd in ("quit", "exit"):
                break

            elif cmd == "help":
                print(HELP)

            elif cmd == "tutorial":
                print(TUTORIAL)

            elif cmd == "reset":
                g = DominoHelper(max_pip=6)
                print("Reset done.")

            elif cmd == "size":
                g.set_size(int(args[0]))
                print(f"Set to double-{g.max_pip} ({len(g.all_tiles)} tiles total).")

            elif cmd == "players":
                g.set_players(args)
                print(f"Opponents set: {g.players}")

            elif cmd == "hand" and args:
                tiles = [parse_tile(a) for a in args]
                g.set_hand(tiles)
                print(f"Your hand set ({len(tiles)} tiles).")

            elif cmd == "hand":
                print("Your hand: "
                      + (", ".join(tile_str(t) for t in sorted(g.my_hand)) or "(empty)"))

            elif cmd == "deal":
                bone = int(args[0]) if args else 0
                g.auto_deal(boneyard=bone)
                print(f"Dealt: { {p: g.hand_count[p] for p in g.players} }, "
                      f"boneyard={g.boneyard}")

            elif cmd == "sethand":
                player, n = args[0], int(args[1])
                if player not in g.players:
                    print(f"Unknown player '{player}'.")
                else:
                    g.hand_count[player] = n
                    print(f"{player} now set to {n} tile(s).")

            elif cmd == "play":
                player = args[0]
                tile = parse_tile(args[1])
                g.mark_play(player, tile)
                print(f"Marked {tile_str(tile)} as played by {player}. "
                      f"{player} now has {g.hand_count[player]} tile(s) left.")

            elif cmd == "myplay":
                tile = parse_tile(args[0])
                g.mark_my_play(tile)
                print(f"Marked {tile_str(tile)} as played by you.")

            elif cmd == "pass":
                player = args[0]
                ends = [int(x) for x in args[1:]]
                g.mark_pass(player, ends)
                print(f"Noted: {player} has none of {ends}.")

            elif cmd == "draw":
                player = args[0]
                g.mark_draw(player)
                print(f"{player} drew from boneyard -> now {g.hand_count[player]} "
                      f"tile(s), boneyard={g.boneyard}.")

            elif cmd == "status":
                cmd_status(g)

            elif cmd == "categories":
                cmd_categories(g)

            elif cmd == "predict":
                cmd_predict(g, args[0] if args else None)

            elif cmd == "unseen":
                u = sorted(g.unseen_tiles())
                print(f"{len(u)} unseen tile(s): " + ", ".join(tile_str(t) for t in u))

            elif cmd == "board":
                if not g.played:
                    print("Nothing played yet.")
                for t, p in g.played:
                    print(f"  {tile_str(t):>5}  played by {p}")

            else:
                print(f"Unknown command '{cmd}'. Type 'help' for the list.")

        except (ValueError, IndexError) as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
