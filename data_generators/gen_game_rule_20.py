"""Generate single-step GameRule tasks (easy / medium / hard)."""
import json, re, random
from pathlib import Path

BASE = Path(__file__).resolve().parent
OUT = BASE.parent / "data" / "task_sets"
OUT.mkdir(parents=True, exist_ok=True)
N_TOTAL = 70; N_SPLIT = 20; SEED = 20260304

def _rng(offset=0): return random.Random(SEED + offset)

EASY_RULES = [
    ("How many pieces does each player start with in chess?", 16),
    ("How many squares are on a standard chessboard?", 64),
    ("How many cards are in a standard deck (no jokers)?", 52),
    ("How many tiles are used in a standard Scrabble game?", 100),
    ("How many dots are on a standard six-sided die total?", 21),
    ("How many pieces are in a standard dominoes set (double-six)?", 28),
    ("How many red properties are on a Monopoly board?", 3),
    ("How many spaces are on a standard Monopoly board?", 40),
    ("How many pawns does each player start with in chess?", 8),
    ("How many suits are in a standard deck of cards?", 4),
    ("How many black squares are on a standard chessboard?", 32),
    ("How many points is the letter Q worth in Scrabble?", 10),
    ("How many letters are drawn at the start of a Scrabble game per player?", 7),
    ("How many cells are in a standard 9x9 Sudoku grid?", 81),
    ("How many regions are in a standard 9x9 Sudoku grid?", 9),
    ("How many face cards are in a standard deck of cards?", 12),
    ("How many aces are in a standard deck of cards?", 4),
    ("How many Community Chest cards are in standard Monopoly?", 16),
    ("How many Chance cards are in standard Monopoly?", 16),
    ("How many pins are used in a standard game of ten-pin bowling?", 10),
    ("How many holes are in a standard round of golf?", 18),
    ("How many players are on a standard basketball team on the court?", 5),
    ("How many players are on a standard soccer team on the field?", 11),
    ("How many players are on a standard volleyball team on the court?", 6),
    ("How many points is a touchdown worth in American football?", 6),
    ("How many periods are in a standard ice hockey game?", 3),
    ("How many bases are on a baseball diamond?", 4),
    ("How many players are on a baseball team on the field?", 9),
    ("How many points is the bullseye worth in standard darts?", 50),
    ("How many squares are in a standard Snakes and Ladders board?", 100),
    ("How many knights does each chess player start with?", 2),
    ("How many bishops does each chess player start with?", 2),
    ("How many rooks does each chess player start with?", 2),
    ("How many queens does each chess player start with?", 1),
    ("How many frames are in a standard game of bowling?", 10),
    ("How many points is a field goal worth in American football?", 3),
    ("How many players are on a standard cricket team on the field?", 11),
    ("How many sets are needed to win a standard men's tennis Grand Slam match?", 3),
    ("How many pockets are on a standard pool table?", 6),
    ("How many balls are used in a standard game of pool (8-ball)?", 16),
    ("How many cards does each player get in a standard poker hand?", 5),
    ("How many cards does each player start with in standard Blackjack?", 2),
    # Additional easy rules
    ("How many quarters are in a standard basketball game?", 4),
    ("How many periods are in a standard hockey game?", 3),
    ("How many innings are in a standard baseball game?", 9),
    ("How many sets are played in a standard tennis match (men's Grand Slam)?", 5),
    ("How many players are on a standard cricket team?", 11),
    ("How many players are on a standard rugby union team?", 15),
    ("How many wickets does each side have in a cricket innings?", 10),
    ("How many red balls are on a standard snooker table?", 15),
    ("How many total balls are on a standard snooker table at the start?", 22),
    ("How many pockets does a standard pool table have?", 6),
    ("How many lanes are in a standard Olympic swimming pool?", 8),
    ("How many hurdles are in a standard 110m hurdles race?", 10),
    ("How many points is a try worth in rugby union?", 5),
    ("How many overs are in a standard cricket One Day International match per side?", 50),
    ("How many players are on a standard water polo team in the water?", 7),
    ("How many doubles are needed to finish a game of darts (standard 501)?", 1),
    ("How many points is the bullseye worth in darts?", 50),
    ("How many tiles does each player draw at the start of a standard Dominoes game?", 7),
    ("How many marbles does each player start with in Chinese Checkers?", 10),
    ("How many stones does each player start with in Go (standard game)?", 0),
    ("How many total dominoes are in a standard double-six set?", 28),
    ("How many numbered cards are in a standard Uno deck?", 76),
    ("How many wild cards are in a standard Uno deck?", 8),
    ("How many pieces does each player have in standard Checkers?", 12),
    ("How many squares are on a standard Checkers board?", 64),
    ("How many points is a safety worth in American football?", 2),
    ("How many players are on a standard lacrosse team on the field?", 10),
    ("How many chukkers are in a standard polo match?", 6),
    ("How many ends are in a standard curling match?", 10),
    ("How many players are on a standard field hockey team on the field?", 11),
]

MEDIUM_RULES = [
    ("How many total possible opening moves are there in chess for White?", 20),
    ("How many tiles are in a standard Mahjong set?", 144),
    ("How many playable intersections are on a standard Go board (19x19)?", 361),
    ("How many unique two-card starting hands exist in Texas Hold'em poker?", 1326),
    ("How many total dominoes are in a double-nine set?", 55),
    ("How many letter tiles have zero point value in Scrabble (blanks)?", 2),
    ("How many railroad spaces are on a standard Monopoly board?", 4),
    ("How many property groups (color sets) are in standard Monopoly?", 8),
    ("How many cards does each player receive in a standard game of Bridge?", 13),
    ("How many discs does each player start with in Othello (Reversi)?", 2),
    ("How many pieces does each player have in Backgammon?", 15),
    ("How many points are on a standard Backgammon board?", 24),
    ("How many territory cards are in the board game Risk?", 42),
    ("How many hexagonal tiles are in the base Catan board?", 19),
    ("How many resource types are in the base Settlers of Catan?", 5),
    ("How many letter S tiles are in a standard English Scrabble set?", 4),
    ("How many cards are in a standard Uno deck?", 108),
    ("How many maximum points can be scored in a single turn in Yahtzee?", 50),
    ("How many categories are scored in a standard Yahtzee game?", 13),
    ("How many total squares can be found on a chessboard (all sizes)?", 204),
    ("How many unique ways can the first four moves of chess be played?", 71852),
    ("How many dice are used in a game of Yahtzee?", 5),
    ("How many rooms are in the board game Clue (Cluedo)?", 9),
    ("How many suspect characters are in the board game Clue (Cluedo)?", 6),
    ("How many weapon cards are in the board game Clue (Cluedo)?", 6),
    ("How many letter E tiles are in a standard English Scrabble set?", 12),
    ("How many victory points do you need to win base Catan?", 10),
    ("How many harbors are on the standard Catan board?", 9),
    ("How many development cards are in base Catan?", 25),
    ("How many knight cards are in a base Catan development deck?", 14),
    ("How many stones does Black place first in a game of Go?", 1),
    ("How many life points does each player start with in standard Magic: The Gathering?", 20),
    ("How many cards are in a standard Tarot deck?", 78),
    ("How many Major Arcana cards are in a Tarot deck?", 22),
    ("How many tiles are in a standard set of Rummikub?", 106),
    ("How many letter Z tiles are in a standard English Scrabble set?", 1),
    ("How many bonus squares are on a standard Scrabble board?", 60),
    ("How many double word score squares are on a Scrabble board?", 17),
    ("How many triple word score squares are on a Scrabble board?", 8),
    ("How many total rectangles can be found on a chessboard?", 1296),
    ("How many dominoes are in a double-twelve set?", 91),
    ("How many cards are dealt face-up in the initial Solitaire (Klondike) layout?", 7),
    # Additional medium rules
    ("How many possible first moves exist in Checkers?", 7),
    ("How many legal positions exist on a standard tic-tac-toe board?", 5478),
    ("How many different Tetris piece shapes (tetrominoes) are there?", 7),
    ("How many tiles are in a standard Azul game?", 100),
    ("How many train cards does each player start with in Ticket to Ride?", 4),
    ("How many development cards are in the base Settlers of Catan?", 25),
    ("How many knight cards are in the base Settlers of Catan?", 14),
    ("How many harbor types are in the base Settlers of Catan?", 5),
    ("How many plague cubes of each color start in the box in Pandemic?", 24),
    ("How many action points does each player get per turn in Pandemic?", 4),
    ("How many total letter tiles are in a standard Words With Friends game?", 104),
    ("How many bonus squares are on a Words With Friends board?", 92),
    ("How many spaces are on a standard Trivial Pursuit board?", 73),
    ("How many category colors are in standard Trivial Pursuit?", 6),
    ("How many life tiles are in a standard Game of Life?", 18),
    ("How many weapon cards are in the board game Clue?", 6),
    ("How many total cards are in a standard Clue game?", 21),
    ("How many ships does each player have in standard Battleship?", 5),
    ("How many grid squares are on each Battleship board?", 100),
    ("How many property spaces does Monopoly have (not including railroads and utilities)?", 22),
    ("How many utility spaces are on a Monopoly board?", 2),
    ("How many total spaces does a player pass on one trip around the Monopoly board?", 40),
    ("How many resource cards can a player hold without discarding in Catan when a 7 is rolled?", 7),
    ("How many points does a settlement earn in Settlers of Catan?", 1),
    ("How many points does a city earn in Settlers of Catan?", 2),
    ("How many cards are in a standard Skip-Bo deck?", 162),
    ("How many cards are in a standard Phase 10 deck?", 108),
    ("How many different phases must be completed to win Phase 10?", 10),
    ("How many tiles are in a standard Rummikub set?", 106),
    ("How many numbered tiles of each color are in Rummikub?", 26),
]

HARD_RULES = [
    ("How many cards are in a Zephyr deck?", 72),
    ("How many tiles are in a Stormhaven tile set?", 96),
    ("How many pieces does each player start with in Ironforge?", 22),
    ("How many hexes are on a standard Duskfall board?", 127),
    ("How many resource types exist in Crystalheim?", 8),
    ("How many action cards are in a Voidrunner deck?", 45),
    ("How many sectors are on a standard Nebula Drift board?", 36),
    ("How many character classes are in Shadowpact?", 14),
    ("How many dice are used in a round of Thunderstone?", 7),
    ("How many victory points are needed to win Moonrise?", 25),
    ("How many terrain types are in Frostpeak Conquest?", 11),
    ("How many spell cards are in the base Arcane Duel set?", 60),
    ("How many guild tokens are in a Tradecraft starting set?", 18),
    ("How many islands are on a standard Tidalwave board?", 9),
    ("How many crew members does each player start with in Corsair's Gambit?", 12),
    ("How many rounds are in a standard Dragonscale match?", 5),
    ("How many trap cards are in the Labyrinth of Echoes base set?", 30),
    ("How many upgrade slots does each ship have in Starfall Armada?", 6),
    ("How many event cards are in the Plague Chronicles deck?", 40),
    ("How many provinces are contested in the Warpath campaign map?", 16),
    ("How many minion tokens are in a Darkspire Dungeon starting set?", 48),
    ("How many power gems does each player begin with in Runebind?", 3),
    ("How many sanctuary tiles are on a Pilgrimage of Ash board?", 7),
    ("How many trade routes are in the base Merchant Winds set?", 20),
    ("How many relic cards are in the Tomb of Eternity expansion?", 33),
    ("How many garrison slots are in each fortress in Siege of Ironhold?", 4),
    ("How many artifact tokens are in the base Mythfall collection?", 15),
    ("How many weather cards are in the Storm Cycle deck?", 28),
    ("How many settlement tiles are in the Frontier Dawn box?", 24),
    ("How many quest cards are in the Wyrmwood Adventures starter?", 50),
    ("How many fleet cards does each player start with in Aether Seas?", 8),
    ("How many dungeon levels are in the Crypts of Malachar base set?", 10),
    ("How many hero cards are in the Champions of Eldoria deck?", 36),
    ("How many biome types are in the Terraform Genesis board?", 9),
    ("How many outpost tokens are in the Void Frontier base set?", 21),
    ("How many council seats are contested in the Intrigue of Ravenmoor?", 13),
    ("How many mutation cards are in the Genome Wars deck?", 44),
    ("How many sectors are in the Hyperdrive Conflict galaxy map?", 42),
    ("How many tribe tokens are in the Dawn of Tribes starter?", 30),
    ("How many blueprint cards are in the Factory Floor base set?", 55),
    ("How many crew roles exist in the Starship Enigma game?", 11),
    ("How many treasure cards are in the Pirate's Bounty deck?", 38),
    # Additional fictional hard rules
    ("How many rune stones does each player start with in Runecaster?", 8),
    ("How many elemental cards are in the Primal Surge deck?", 64),
    ("How many chapters are in a standard Mythos Unraveled campaign?", 12),
    ("How many mana crystals does each player begin with in Arcane Duels?", 4),
    ("How many tiles make up the Echoing Caverns dungeon board?", 81),
    ("How many faction cards are in the Warfront Allegiance set?", 50),
    ("How many dice are used in a round of Dragon's Gambit?", 8),
    ("How many supply tokens are in the Frontier Settlement starting kit?", 24),
    ("How many hex tiles form the standard Nebula Conquest board?", 37),
    ("How many character abilities does each hero have in Shadow Legends?", 5),
    ("How many quest cards are in the Wanderer's Path base set?", 45),
    ("How many gold coins does each player begin with in Merchant Prince?", 15),
    ("How many artifact slots does each explorer have in Tomb Raiders?", 4),
    ("How many biome tiles are in the standard Terraform Genesis set?", 28),
    ("How many action tokens does each player receive per round in Chrono Clash?", 6),
    ("How many city tiles are in the Metropolis Builder starting pack?", 32),
    ("How many spell components are in the Alchemy Wars base set?", 72),
    ("How many scouts does each player deploy in Recon Strike?", 10),
    ("How many victory flags are contested in Battlefront Zero?", 7),
    ("How many mission cards are in the Covert Ops deck?", 36),
    ("How many weapon types exist in the Gladiator Arena game?", 9),
    ("How many trade goods does each merchant start with in Silk Road Legends?", 20),
    ("How many monster tokens are in the Nightmare Dungeon starting set?", 56),
    ("How many starships does each fleet start with in Galactic Armada?", 8),
    ("How many terrain cards are in the Wilderness Survival deck?", 44),
    ("How many invention tokens are in the Age of Discovery set?", 18),
    ("How many alliance cards are in the Throne Wars expansion?", 28),
    ("How many minion types exist in the Overlord's Siege game?", 13),
    ("How many portal tiles are on the Riftwalk board?", 6),
    ("How many harvest tokens are in the Farmstead Chronicles base set?", 40),
]

def game_env(corpus):
    return [{"name": "GameRuleEnv", "tools": ["lookup_rule"], "parameters": {"corpus": corpus}}]

def task(tid, diff, instr, ans, corpus):
    return {"id": tid, "difficulty": diff, "multi_step": False, "instruction": instr,
            "environments": game_env(corpus), "expected": {"answer": str(ans)}, "tags": ["game_rule", "single_round"]}

def _load_paragraphs(difficulty):
    try:
        from data.task_sets.paragraphs_game_rule import EASY_PARAGRAPHS, MEDIUM_PARAGRAPHS, HARD_PARAGRAPHS
    except ModuleNotFoundError:
        from paragraphs_game_rule import EASY_PARAGRAPHS, MEDIUM_PARAGRAPHS, HARD_PARAGRAPHS
    return {"easy": EASY_PARAGRAPHS, "medium": MEDIUM_PARAGRAPHS, "hard": HARD_PARAGRAPHS}[difficulty]

def gen_single(start_id, difficulty, pool, rng_offset):
    rng = _rng(rng_offset)
    paragraphs = _load_paragraphs(difficulty)
    # Build corpus: question -> paragraph (model must extract the number from the paragraph)
    corpus = {}
    for q, a in pool:
        if q in paragraphs:
            corpus[q] = paragraphs[q]
        else:
            # Fallback if no paragraph
            corpus[q] = f"The answer is {a}."
    items = list(pool)
    rng.shuffle(items)
    out = []
    for i, (q, a) in enumerate(items[:N_TOTAL]):
        instr = f"{q} Answer with only the number."
        out.append(task(start_id + i, difficulty, instr, a, corpus))
    return out

def validate(tasks, label, expected_n=N_TOTAL):
    if len(tasks) != expected_n:
        raise ValueError(f"{label}: expected {expected_n}, got {len(tasks)}")
    inst = [re.sub(r"\s+", " ", t["instruction"].strip().lower()) for t in tasks]
    if len(set(inst)) != len(inst):
        raise ValueError(f"{label}: duplicate instruction found")


def split_train_test(tasks): return tasks[:N_SPLIT], tasks[N_SPLIT:]

def main():
    gens = {
        "game_rule_single_easy": gen_single(3500, "easy", EASY_RULES, 180),
        "game_rule_single_medium": gen_single(3600, "medium", MEDIUM_RULES, 280),
        "game_rule_single_hard": gen_single(3700, "hard", HARD_RULES, 380),
    }
    for name, tasks in gens.items():
        validate(tasks, name)
        train, test = split_train_test(tasks)
        for sn, st in [("train", train), ("test", test)]:
            fname = f"{name}_{sn}.json"
            (OUT / fname).write_text(json.dumps(st, ensure_ascii=False, indent=2) + "\n")
            print(f"wrote {fname}: {len(st)}")

if __name__ == "__main__": main()
