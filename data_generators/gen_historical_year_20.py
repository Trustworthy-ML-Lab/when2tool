"""Generate single-step HistoricalYear tasks (easy / medium / hard)."""
import json, re, random
from pathlib import Path

BASE = Path(__file__).resolve().parent
OUT = BASE.parent / "data" / "task_sets"
OUT.mkdir(parents=True, exist_ok=True)
N_TOTAL = 70; N_SPLIT = 20; SEED = 20260304

def _rng(offset=0): return random.Random(SEED + offset)

EASY_EVENTS = [
    ("humans first landed on the Moon", 1969),
    ("World War II ended", 1945),
    ("the Berlin Wall fell", 1989),
    ("the Declaration of Independence was signed", 1776),
    ("the Titanic sank", 1912),
    ("World War I began", 1914),
    ("the French Revolution began", 1789),
    ("Christopher Columbus reached the Americas", 1492),
    ("the United States Civil War ended", 1865),
    ("Nelson Mandela was released from prison", 1990),
    ("the Soviet Union dissolved", 1991),
    ("Martin Luther King Jr. gave the 'I Have a Dream' speech", 1963),
    ("Pearl Harbor was attacked", 1941),
    ("the atomic bomb was dropped on Hiroshima", 1945),
    ("the Wright brothers made their first powered flight", 1903),
    ("Queen Victoria died", 1901),
    ("the Great Fire of London occurred", 1666),
    ("Abraham Lincoln was assassinated", 1865),
    ("the Magna Carta was signed", 1215),
    ("the first Olympic Games of the modern era were held", 1896),
    ("the Korean War began", 1950),
    ("India gained independence", 1947),
    ("the Panama Canal was completed", 1914),
    ("the Chernobyl disaster occurred", 1986),
    ("the September 11 attacks occurred", 2001),
    ("the first iPhone was released", 2007),
    ("the Euro currency was introduced", 1999),
    ("the Vietnam War ended", 1975),
    ("the Cuban Missile Crisis occurred", 1962),
    ("the Hindenburg disaster occurred", 1937),
    ("the first man-made satellite Sputnik was launched", 1957),
    ("the Suez Crisis occurred", 1956),
    ("Alaska became a U.S. state", 1959),
    ("the first heart transplant was performed", 1967),
    ("the Watergate scandal began", 1972),
    ("the fall of Saigon occurred", 1975),
    ("the Camp David Accords were signed", 1978),
    ("the Iranian Revolution occurred", 1979),
    ("the Falklands War began", 1982),
    ("the Good Friday Agreement was signed", 1998),
    ("Hong Kong was returned to China", 1997),
    # Additional easy events
    ("the Rwandan genocide began", 1994),
    ("the Space Shuttle Challenger disaster occurred", 1986),
    ("the Berlin Olympics were held", 1936),
    ("the stock market crash known as Black Tuesday occurred", 1929),
    ("Prohibition began in the United States", 1920),
    ("the Lusitania was sunk", 1915),
    ("the Boer War ended", 1902),
    ("the Spanish-American War began", 1898),
    ("the Eiffel Tower was completed", 1889),
    ("the Statue of Liberty was dedicated", 1886),
    ("the telephone was patented by Bell", 1876),
    ("the American Civil War began", 1861),
    ("the California Gold Rush began", 1848),
    ("the Trail of Tears began", 1838),
    ("the Battle of Waterloo took place", 1815),
    ("Napoleon became Emperor of France", 1804),
    ("the Boston Massacre occurred", 1770),
    ("the Seven Years War began", 1756),
    ("the Great Plague of London began", 1665),
    ("the Gunpowder Plot was discovered", 1605),
    ("the Spanish Inquisition was established", 1478),
    ("the Hundred Years War ended", 1453),
    ("the Black Death reached Europe", 1347),
    ("the Crusades began", 1096),
    ("William the Conqueror invaded England", 1066),
    ("the fall of the Western Roman Empire occurred", 476),
    ("Julius Caesar was assassinated", 44),
    ("the Great Wall of China construction began", 221),
    ("Alexander the Great died", 323),
    ("the first ancient Olympic Games were held", 776),
]

MEDIUM_EVENTS = [
    ("the Treaty of Westphalia was signed", 1648),
    ("the Edict of Nantes was issued", 1598),
    ("the Battle of Hastings took place", 1066),
    ("the printing press was invented by Gutenberg", 1440),
    ("Magellan's expedition completed circumnavigation of the globe", 1522),
    ("the Congress of Vienna concluded", 1815),
    ("the Louisiana Purchase was completed", 1803),
    ("the Suez Canal was opened", 1869),
    ("the Treaty of Tordesillas was signed", 1494),
    ("the War of the Roses ended", 1485),
    ("the Bastille was stormed in Paris", 1789),
    ("the Act of Union united England and Scotland", 1707),
    ("the Boston Tea Party occurred", 1773),
    ("the Emancipation Proclamation was issued", 1863),
    ("the Treaty of Versailles was signed", 1919),
    ("the Spanish Armada was defeated", 1588),
    ("Napoleon was exiled to Saint Helena", 1815),
    ("the Meiji Restoration began in Japan", 1868),
    ("the Opium War began", 1839),
    ("the Russian Revolution occurred", 1917),
    ("the Russo-Japanese War ended", 1905),
    ("the Boxer Rebellion occurred in China", 1900),
    ("the Ottoman Empire fell", 1922),
    ("the League of Nations was established", 1920),
    ("the Crimean War began", 1853),
    ("the Berlin Conference on Africa was held", 1884),
    ("the Zimmermann Telegram was intercepted", 1917),
    ("the Munich Agreement was signed", 1938),
    ("the Balfour Declaration was issued", 1917),
    ("the first Sino-Japanese War began", 1894),
    ("the Taiping Rebellion began", 1850),
    ("the Franco-Prussian War began", 1870),
    ("the Congress of Berlin was held", 1878),
    ("the Dreyfus Affair began in France", 1894),
    ("the Boxer Protocol was signed", 1901),
    ("the Young Turk Revolution occurred", 1908),
    ("the Italian unification was completed", 1871),
    ("the Missouri Compromise was passed", 1820),
    ("the Mexican-American War began", 1846),
    ("the Sepoy Mutiny began in India", 1857),
    ("the Zulu War began", 1879),
    # Additional medium events
    ("the Defenestration of Prague occurred", 1618),
    ("the Petition of Right was signed", 1628),
    ("the Glorious Revolution occurred in England", 1688),
    ("the War of Spanish Succession began", 1701),
    ("the Treaty of Utrecht was signed", 1713),
    ("the Seven Years War ended", 1763),
    ("the First Partition of Poland occurred", 1772),
    ("the Haitian Revolution began", 1791),
    ("the Battle of Trafalgar took place", 1805),
    ("the Congress of Troppau was held", 1820),
    ("Greek independence was declared", 1821),
    ("the Revolutions of 1848 swept across Europe", 1848),
    ("the Crimean War ended", 1856),
    ("the Suez Canal construction began", 1859),
    ("the Austro-Prussian War occurred", 1866),
    ("the Paris Commune was established", 1871),
    ("the Treaty of San Stefano was signed", 1878),
    ("the Partition of Africa was formalized at Berlin", 1885),
    ("the Fashoda Incident occurred", 1898),
    ("the Anglo-Japanese Alliance was formed", 1902),
    ("the Algeciras Conference was held", 1906),
    ("the Chinese Revolution overthrew the Qing Dynasty", 1911),
    ("the Treaty of Brest-Litovsk was signed", 1918),
    ("the Kellogg-Briand Pact was signed", 1928),
    ("the Lateran Treaty was signed", 1929),
    ("the Spanish Civil War began", 1936),
    ("the Anschluss of Austria occurred", 1938),
    ("the Molotov-Ribbentrop Pact was signed", 1939),
    ("the Yalta Conference was held", 1945),
    ("the North Atlantic Treaty was signed", 1949),
]

# Synthetic events for hard mode
HARD_EVENTS = [
    ("the Accord of Velmorath was signed", 1723),
    ("the Pact of Cendara was ratified", 1587),
    ("the Treaty of Novalund was enacted", 1642),
    ("the Charter of Brimstone Keep was issued", 1381),
    ("the Edict of Thornwall was proclaimed", 1509),
    ("the Congress of Ashenmere concluded", 1756),
    ("the Compact of Silverdale was sealed", 1834),
    ("the Alliance of Duskport was formed", 1693),
    ("the Decree of Frosthollow was announced", 1412),
    ("the Convention of Windmere took place", 1771),
    ("the Protocol of Ironvale was adopted", 1848),
    ("the Summit of Stormreach concluded", 1529),
    ("the Armistice of Darkwater was signed", 1667),
    ("the Declaration of Moonhaven was issued", 1743),
    ("the Ordinance of Coppergate was passed", 1456),
    ("the Mandate of Greypeak was established", 1602),
    ("the Resolution of Brightmoor was ratified", 1819),
    ("the Proclamation of Redshore was made", 1564),
    ("the Statute of Hawkridge was enacted", 1478),
    ("the Agreement of Sunveil was concluded", 1691),
    ("the Covenant of Ashford was sworn", 1537),
    ("the Concordat of Ravenholm was finalized", 1805),
    ("the Directive of Thornbury was issued", 1623),
    ("the Tribunal of Mistfall was convened", 1714),
    ("the Accord of Embervale was concluded", 1552),
    ("the Writ of Starfall was proclaimed", 1489),
    ("the Pact of Ironthorne was established", 1635),
    ("the Edict of Shadowmere was ratified", 1702),
    ("the Treaty of Goldcrest was signed", 1779),
    ("the Alliance of Frostwyrm was formed", 1498),
    ("the Concordat of Silverpine was sealed", 1613),
    ("the Declaration of Ashvale was issued", 1734),
    ("the Protocol of Blackthorn was adopted", 1567),
    ("the Summit of Ember Falls concluded", 1823),
    ("the Charter of Dawnstone was granted", 1446),
    ("the Mandate of Stormhaven was decreed", 1681),
    ("the Convention of Oakmere took place", 1758),
    ("the Compact of Nighthollow was negotiated", 1594),
    ("the Armistice of Brightvale was declared", 1647),
    ("the Ordinance of Wolfcrest was enacted", 1513),
    ("the Resolution of Deepwater was passed", 1836),
    ("the Decree of Fireheart was announced", 1465),
    # Additional synthetic hard events
    ("the Compact of Thornfield was ratified", 1543),
    ("the Edict of Gloomreach was proclaimed", 1658),
    ("the Truce of Coppervale was declared", 1791),
    ("the Charter of Nighthollow was granted", 1423),
    ("the Pact of Silverfen was sealed", 1576),
    ("the Resolution of Dawnbreak was passed", 1804),
    ("the Congress of Ashwick concluded", 1738),
    ("the Alliance of Rimecroft was formed", 1615),
    ("the Declaration of Goldmere was issued", 1487),
    ("the Ordinance of Whitecliff was enacted", 1552),
    ("the Summit of Ravensgate concluded", 1669),
    ("the Protocol of Ironmere was adopted", 1743),
    ("the Tribunal of Shadowfen was convened", 1518),
    ("the Concordat of Brightstone was sealed", 1631),
    ("the Directive of Moonshadow was issued", 1702),
    ("the Covenant of Embercross was sworn", 1456),
    ("the Writ of Stormvale was proclaimed", 1584),
    ("the Agreement of Frostpeak was concluded", 1767),
    ("the Accord of Darkholme was signed", 1639),
    ("the Treaty of Thorngate was enacted", 1508),
    ("the Edict of Sunridge was ratified", 1674),
    ("the Compact of Ironforge was established", 1793),
    ("the Decree of Ashenmoor was announced", 1441),
    ("the Alliance of Silverbrook was formed", 1562),
    ("the Charter of Duskmeadow was granted", 1688),
    ("the Pact of Goldthorn was sealed", 1523),
    ("the Resolution of Mistwood was ratified", 1756),
    ("the Proclamation of Embervale was made", 1614),
    ("the Summit of Frosthaven concluded", 1835),
    ("the Mandate of Nightstone was decreed", 1479),
]

def year_env(corpus):
    return [{"name": "HistoricalYearEnv", "tools": ["lookup_year"], "parameters": {"corpus": corpus}}]

def task(tid, diff, instr, ans, corpus):
    return {"id": tid, "difficulty": diff, "multi_step": False, "instruction": instr,
            "environments": year_env(corpus), "expected": {"answer": str(ans)}, "tags": ["historical_year", "single_round"]}

def _load_paragraphs(difficulty):
    try:
        from data.task_sets.paragraphs_historical_year import EASY_PARAGRAPHS, MEDIUM_PARAGRAPHS, HARD_PARAGRAPHS
    except ModuleNotFoundError:
        from paragraphs_historical_year import EASY_PARAGRAPHS, MEDIUM_PARAGRAPHS, HARD_PARAGRAPHS
    return {"easy": EASY_PARAGRAPHS, "medium": MEDIUM_PARAGRAPHS, "hard": HARD_PARAGRAPHS}[difficulty]

def gen_single(start_id, difficulty, pool, rng_offset):
    rng = _rng(rng_offset)
    paragraphs = _load_paragraphs(difficulty)
    # Build corpus: event -> paragraph (model must extract the year from the paragraph)
    corpus = {}
    for event, year in pool:
        if event in paragraphs:
            corpus[event] = paragraphs[event]
        else:
            # Fallback: one-liner if no paragraph available
            corpus[event] = f"In {year}, {event}."
    items = list(pool)
    rng.shuffle(items)
    out = []
    for i, (event, year) in enumerate(items[:N_TOTAL]):
        instr = f"In what year was {event}? Answer with only the year as a number."
        out.append(task(start_id + i, difficulty, instr, year, corpus))
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
        "historical_year_single_easy": gen_single(3200, "easy", EASY_EVENTS, 170),
        "historical_year_single_medium": gen_single(3300, "medium", MEDIUM_EVENTS, 270),
        "historical_year_single_hard": gen_single(3400, "hard", HARD_EVENTS, 370),
    }
    for name, tasks in gens.items():
        validate(tasks, name)
        train, test = split_train_test(tasks)
        for sn, st in [("train", train), ("test", test)]:
            fname = f"{name}_{sn}.json"
            (OUT / fname).write_text(json.dumps(st, ensure_ascii=False, indent=2) + "\n")
            print(f"wrote {fname}: {len(st)}")

if __name__ == "__main__": main()
