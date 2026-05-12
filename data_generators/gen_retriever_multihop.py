"""Generate multi-hop (3-step chained) Retriever tasks.

Genuine x→y→z dependency: each answer leads to a NEW entity not in the question.
  - Easy: well-known real-world facts (Shakespeare → England → London)
  - Medium: less common real-world facts
  - Hard: fictional entities — agent MUST use corpus, can't rely on prior knowledge
"""
import json
import random
import re
from pathlib import Path
from paragraphs_retriever import HARD_PARAGRAPHS, HARD_FACTS

BASE = Path(__file__).resolve().parent
OUT = BASE.parent / "data" / "task_sets"
OUT.mkdir(parents=True, exist_ok=True)
N_TOTAL = 70
N_SPLIT = 20
SEED = 20260304

RETRIEVER_TOOLS = ["search_corpus", "read_doc"]


def _rng():
    return random.Random(SEED + 170)


def retriever_env(corpus):
    return [
        {
            "name": "RetrieverEnv",
            "tools": RETRIEVER_TOOLS,
            "parameters": {"corpus": corpus},
        }
    ]


def _normalize_inst(x):
    return re.sub(r"\s+", " ", x.strip().lower())


def _doc_with_id(idx, title, text):
    return {"id": f"doc{idx}", "title": title, "text": text}


def _materialize_corpus(raw_docs, rng=None):
    docs = list(raw_docs)
    if rng is not None:
        rng.shuffle(docs)
    return [_doc_with_id(i, d["title"], d["text"]) for i, d in enumerate(docs, 1)]


# =========================================================================
# Chain definitions: each chain is a list of 3 hops
# Each hop: (question, answer, doc_title, doc_text)
# The question for hop N+1 uses the answer from hop N via "x" or "y"
# =========================================================================

# Distractor doc pool (shared across difficulties)
EASY_DISTRACTORS = [
    {"title": "Amazon River", "text": "The Amazon River in South America is the largest river by volume of water flow. It runs through Brazil, Peru, and Colombia, and its basin covers about 40% of South America."},
    {"title": "Great Wall of China", "text": "The Great Wall of China is a series of fortifications stretching over 13,000 miles. It was built over many centuries to protect Chinese states from invasions."},
    {"title": "Sahara Desert", "text": "The Sahara is the largest hot desert in the world, covering much of North Africa. It spans over 9 million square kilometers across 11 countries."},
    {"title": "Pacific Ocean", "text": "The Pacific Ocean is the largest and deepest ocean on Earth, covering more than 63 million square miles. It was named by explorer Ferdinand Magellan."},
    {"title": "Mount Kilimanjaro", "text": "Mount Kilimanjaro in Tanzania is the highest mountain in Africa at 5,895 meters. It is a dormant volcanic massif with three volcanic cones."},
    {"title": "Nile River", "text": "The Nile is a major river in northeastern Africa. It is generally regarded as the longest river in the world, flowing about 6,650 km through 11 countries."},
    {"title": "Antarctica", "text": "Antarctica is Earth's southernmost continent, containing the South Pole. It is the coldest, driest, and windiest continent on average."},
    {"title": "Periodic Table", "text": "The periodic table organizes chemical elements by atomic number. Dmitri Mendeleev published the first widely recognized periodic table in 1869."},
    {"title": "Solar System", "text": "The Solar System consists of the Sun and the objects that orbit it, including eight planets, dwarf planets, and billions of smaller bodies."},
    {"title": "Renaissance", "text": "The Renaissance was a cultural movement that began in Italy in the 14th century and spread throughout Europe. It marked a revival of interest in classical art, literature, and learning."},
]

# =========================================================================
# EASY CHAINS: well-known real-world facts
# =========================================================================
EASY_CHAINS = [
    # Author chains
    [("Who wrote Romeo and Juliet?", "William Shakespeare",
      "Romeo and Juliet", "Romeo and Juliet is a tragedy written by William Shakespeare around 1595 about two young lovers in Verona."),
     ("In which country was x born?", "England",
      "William Shakespeare biography", "William Shakespeare was born in Stratford-upon-Avon, England, in 1564. He is widely regarded as the greatest writer in the English language."),
     ("What is the capital of y?", "London",
      "England", "England is a country that is part of the United Kingdom. Its capital and largest city is London, situated on the River Thames.")],

    [("Who wrote 1984?", "George Orwell",
      "1984 novel", "1984 is a dystopian novel published in 1949 by George Orwell. It depicts a totalitarian society under the surveillance of Big Brother."),
     ("In which country was x born?", "India",
      "George Orwell biography", "George Orwell was born Eric Arthur Blair in Motihari, India, in 1903. He later moved to England and became one of the most influential writers of the 20th century."),
     ("What is the capital of y?", "New Delhi",
      "India", "India is a country in South Asia. Its capital is New Delhi, and its largest city is Mumbai. India has a population of over 1.4 billion people.")],

    [("Who wrote The Hobbit?", "J. R. R. Tolkien",
      "The Hobbit", "The Hobbit is a fantasy novel by J. R. R. Tolkien, published in 1937. It tells the story of Bilbo Baggins' adventure with dwarves."),
     ("In which country was x born?", "South Africa",
      "J. R. R. Tolkien biography", "J. R. R. Tolkien was born in Bloemfontein, South Africa, in 1892. His family moved to England when he was three years old."),
     ("What is the largest city in y?", "Johannesburg",
      "South Africa", "South Africa is a country at the southern tip of Africa. Its largest city is Johannesburg, while its capitals are Pretoria, Cape Town, and Bloemfontein.")],

    [("Who wrote War and Peace?", "Leo Tolstoy",
      "War and Peace", "War and Peace is a novel by Leo Tolstoy, published between 1865 and 1869. It chronicles Russian society during the Napoleonic era."),
     ("In which country was x born?", "Russia",
      "Leo Tolstoy biography", "Leo Tolstoy was born in Yasnaya Polyana, Russia, in 1828. He is regarded as one of the greatest novelists of all time."),
     ("What is the capital of y?", "Moscow",
      "Russia", "Russia is the largest country in the world by area. Its capital and largest city is Moscow, situated on the Moskva River.")],

    [("Who wrote The Great Gatsby?", "F. Scott Fitzgerald",
      "The Great Gatsby", "The Great Gatsby is a 1925 novel by F. Scott Fitzgerald set during the Jazz Age on Long Island."),
     ("In which country was x born?", "United States",
      "F. Scott Fitzgerald biography", "F. Scott Fitzgerald was born in St. Paul, Minnesota, United States, in 1896. He became a leading figure of the Lost Generation."),
     ("What currency is used in y?", "US dollar",
      "United States", "The United States uses the US dollar as its official currency. The dollar is the world's primary reserve currency.")],

    [("Who wrote Pride and Prejudice?", "Jane Austen",
      "Pride and Prejudice", "Pride and Prejudice is a novel by Jane Austen, first published in 1813. It follows Elizabeth Bennet and Mr. Darcy."),
     ("In which country was x born?", "England",
      "Jane Austen biography", "Jane Austen was born in Steventon, Hampshire, England, in 1775. She is one of the most widely read authors in English literature."),
     ("What is the official language of y?", "English",
      "England", "England is part of the United Kingdom. The official language is English, which originated from the Anglo-Saxon kingdoms.")],

    [("Who wrote Hamlet?", "William Shakespeare",
      "Hamlet", "Hamlet is a tragedy by William Shakespeare, written around 1600. It tells the story of Prince Hamlet avenging his father's murder."),
     ("What is the birthplace town of x?", "Stratford-upon-Avon",
      "William Shakespeare birthplace", "William Shakespeare was born in Stratford-upon-Avon in 1564. The town is located in Warwickshire, England."),
     ("In which country is y located?", "England",
      "Stratford-upon-Avon", "Stratford-upon-Avon is a market town in Warwickshire, England. It is famous as the birthplace of William Shakespeare.")],

    [("Who wrote Moby-Dick?", "Herman Melville",
      "Moby-Dick", "Moby-Dick is an 1851 novel by Herman Melville about Captain Ahab's quest to hunt the white whale."),
     ("In which country was x born?", "United States",
      "Herman Melville biography", "Herman Melville was born in New York City, United States, in 1819. He worked as a sailor before becoming a novelist."),
     ("What is the capital of y?", "Washington, D.C.",
      "United States capital", "The capital of the United States is Washington, D.C., established in 1790 as the seat of the federal government.")],

    # Planet / Space chains
    [("Which planet is known as the Red Planet?", "Mars",
      "Red Planet", "Mars is known as the Red Planet due to iron oxide on its surface giving it a reddish appearance."),
     ("What is the largest moon of x?", "Phobos",
      "Mars moons", "Mars has two small moons: Phobos and Deimos. Phobos is the larger of the two, with a diameter of about 22.4 km."),
     ("What does the name y mean in Greek?", "fear",
      "Phobos", "Phobos is named after the Greek god of fear and panic. It orbits Mars at a distance of only 6,000 km from the surface.")],

    [("Which planet is known as the Ringed Planet?", "Saturn",
      "Ringed Planet", "Saturn is known as the Ringed Planet due to its spectacular system of rings made of ice and rock."),
     ("What is the largest moon of x?", "Titan",
      "Saturn moons", "Saturn has 146 known moons. Titan is the largest, with a diameter of 5,150 km — larger than the planet Mercury."),
     ("What kind of atmosphere does y have?", "nitrogen",
      "Titan atmosphere", "Titan is the only moon in the Solar System with a dense atmosphere. Its atmosphere is primarily composed of nitrogen, similar to Earth's.")],

    [("Which planet is closest to the Sun?", "Mercury",
      "Closest planet to Sun", "Mercury is the closest planet to the Sun, orbiting at an average distance of about 58 million kilometers."),
     ("How many moons does x have?", "zero",
      "Mercury moons", "Mercury has zero moons. It is one of only two planets in the Solar System without any natural satellites, the other being Venus."),
     ("Which other planet also has y moons?", "Venus",
      "Moonless planets", "Venus is the other planet with zero natural satellites. Despite being similar in size to Earth, Venus has no moons.")],

    # Inventor chains
    [("Who invented the telephone?", "Alexander Graham Bell",
      "Telephone invention", "The telephone was invented by Alexander Graham Bell, who received the first patent for it in 1876."),
     ("In which country was x born?", "Scotland",
      "Alexander Graham Bell biography", "Alexander Graham Bell was born in Edinburgh, Scotland, in 1847. He later emigrated to Canada and then the United States."),
     ("What is the capital of y?", "Edinburgh",
      "Scotland", "Scotland is a country in the United Kingdom. Its capital is Edinburgh, known for its historic castle and annual festivals.")],

    [("Who invented dynamite?", "Alfred Nobel",
      "Dynamite invention", "Dynamite was invented by Alfred Nobel in 1867. He patented it as a safer alternative to nitroglycerin for use in construction and mining."),
     ("In which country was x born?", "Sweden",
      "Alfred Nobel biography", "Alfred Nobel was born in Stockholm, Sweden, in 1833. He established the Nobel Prizes in his will."),
     ("What is the capital of y?", "Stockholm",
      "Sweden", "Sweden is a Nordic country in Northern Europe. Its capital and largest city is Stockholm, built on 14 islands.")],

    [("Who invented the printing press?", "Johannes Gutenberg",
      "Printing press", "The printing press with movable type was invented by Johannes Gutenberg around 1440. It revolutionized the production of books."),
     ("In which country was x born?", "Germany",
      "Johannes Gutenberg biography", "Johannes Gutenberg was born in Mainz, Germany, around 1400. His invention is considered one of the most important in human history."),
     ("What is the capital of y?", "Berlin",
      "Germany", "Germany is a country in Central Europe. Its capital and largest city is Berlin, which served as the focal point of the Cold War.")],

    [("Who is credited with inventing the light bulb?", "Thomas Edison",
      "Light bulb invention", "Thomas Edison is credited with inventing the practical incandescent light bulb in 1879 at his Menlo Park laboratory."),
     ("In which country was x born?", "United States",
      "Thomas Edison biography", "Thomas Edison was born in Milan, Ohio, United States, in 1847. He held over 1,000 patents for his inventions."),
     ("What is the largest state in y by area?", "Alaska",
      "United States geography", "The largest state in the United States by area is Alaska, covering 663,267 square miles. It was purchased from Russia in 1867.")],

    # Geography chains
    [("What is the highest mountain in the world?", "Mount Everest",
      "Highest mountain", "Mount Everest is the highest mountain in the world at 8,849 meters above sea level. It is located in the Himalayas."),
     ("In which country is x located?", "Nepal",
      "Mount Everest location", "Mount Everest is located on the border between Nepal and Tibet (China). The south side is in Nepal's Sagarmatha National Park."),
     ("What is the capital of y?", "Kathmandu",
      "Nepal", "Nepal is a landlocked country in South Asia. Its capital and largest city is Kathmandu, located in a valley surrounded by the Himalayas.")],

    [("What is the highest mountain in Japan?", "Mount Fuji",
      "Japan highest mountain", "Mount Fuji is the highest mountain in Japan at 3,776 meters. It is an active stratovolcano on Honshu Island."),
     ("On which island is x located?", "Honshu",
      "Mount Fuji location", "Mount Fuji is located on Honshu, the largest and most populous island of Japan. It is about 100 km southwest of Tokyo."),
     ("What is the largest city on y?", "Tokyo",
      "Honshu", "Honshu is the largest island of Japan. The largest city on Honshu is Tokyo, the capital of Japan, with a metropolitan population exceeding 37 million.")],

    [("What is the longest river in Africa?", "Nile",
      "Longest river in Africa", "The Nile is the longest river in Africa, flowing about 6,650 km from its sources in East Africa to the Mediterranean Sea."),
     ("Into which sea does x empty?", "Mediterranean Sea",
      "Nile River", "The Nile River flows northward through northeastern Africa and empties into the Mediterranean Sea at its delta in Egypt."),
     ("Which European country has the longest coastline on y?", "Greece",
      "Mediterranean Sea coastlines", "Greece has the longest coastline on the Mediterranean Sea among European countries, with over 13,600 km of coastline including its many islands.")],

    [("What is the capital of Australia?", "Canberra",
      "Australia capital", "Canberra is the capital of Australia, chosen as a compromise between Sydney and Melbourne. It was purpose-built as the capital."),
     ("In which state or territory is x located?", "Australian Capital Territory",
      "Canberra location", "Canberra is located in the Australian Capital Territory (ACT), a federal territory established in 1911 specifically to house the national capital."),
     ("Which two cities surround y?", "Sydney and Melbourne",
      "Australian Capital Territory", "The Australian Capital Territory is located between Sydney and Melbourne, the two largest cities in Australia. It was placed in this location as a geographic compromise.")],

    # Element / Science chains
    [("What is the chemical symbol for Gold?", "Au",
      "Gold symbol", "Au is the chemical symbol for gold, derived from the Latin word aurum meaning 'shining dawn'. Gold has atomic number 79."),
     ("What Latin word does x come from?", "aurum",
      "Au etymology", "The chemical symbol Au for gold comes from the Latin word aurum, which means 'shining dawn' or 'glow of sunrise'."),
     ("What does y mean in English?", "shining dawn",
      "Aurum meaning", "The Latin word aurum means 'shining dawn' or 'glow of sunrise' in English. It is the origin of the chemical symbol Au for gold.")],

    [("What is the chemical symbol for Iron?", "Fe",
      "Iron symbol", "Fe is the chemical symbol for iron, derived from the Latin word ferrum. Iron has atomic number 26."),
     ("What Latin word does x come from?", "ferrum",
      "Fe etymology", "The chemical symbol Fe for iron comes from the Latin word ferrum, which was used by the Romans to refer to the metal."),
     ("Which historical age is named after the metal that y refers to?", "Iron Age",
      "Ferrum history", "The Latin word ferrum refers to iron. The Iron Age began around 1200 BC, when iron replaced bronze as the primary metal for tools and weapons.")],

    [("What element has the atomic number 1?", "Hydrogen",
      "Atomic number 1", "Hydrogen is the element with atomic number 1. It is the lightest and most abundant element in the universe."),
     ("What is the chemical symbol for x?", "H",
      "Hydrogen", "Hydrogen has the chemical symbol H. It is a colorless, odorless gas at standard conditions."),
     ("What word does y stand for in the abbreviation H2O?", "hydrogen",
      "H2O", "In H2O, the H stands for hydrogen. Water (H2O) consists of two hydrogen atoms bonded to one oxygen atom.")],

    # Currency / Country chains
    [("What currency is used in Japan?", "yen",
      "Japan currency", "The yen is the official currency of Japan, represented by the symbol ¥ and the ISO code JPY."),
     ("What is the ISO currency code for x?", "JPY",
      "Yen details", "The Japanese yen has the ISO currency code JPY. It is the third most traded currency in the foreign exchange market."),
     ("Which country does the currency code y belong to?", "Japan",
      "JPY code", "The currency code JPY stands for Japanese Yen, the official currency of Japan.")],

    [("What is the official language of Brazil?", "Portuguese",
      "Brazil language", "The official language of Brazil is Portuguese. Brazil is the largest Portuguese-speaking country in the world."),
     ("Which European country did x originate from?", "Portugal",
      "Portuguese language origin", "The Portuguese language originated from Portugal, a country on the Iberian Peninsula in southwestern Europe."),
     ("What is the capital of y?", "Lisbon",
      "Portugal", "Portugal is a country in southwestern Europe. Its capital and largest city is Lisbon, situated on the Tagus River estuary.")],

    [("What is the official language of Mexico?", "Spanish",
      "Mexico language", "The official language of Mexico is Spanish. Mexico has the largest number of Spanish speakers in the world."),
     ("Which European country did x originate from?", "Spain",
      "Spanish language origin", "The Spanish language originated from Spain, specifically from the Castile region. It spread globally during the Age of Exploration."),
     ("What is the capital of y?", "Madrid",
      "Spain", "Spain is a country on the Iberian Peninsula in southwestern Europe. Its capital and largest city is Madrid.")],

    # More author chains with different Step 2/3
    [("Who wrote Romeo and Juliet?", "William Shakespeare",
      "Romeo and Juliet", "Romeo and Juliet is a tragedy written by William Shakespeare around 1595 about two young lovers in Verona."),
     ("What is the birthplace town of x?", "Stratford-upon-Avon",
      "William Shakespeare birthplace", "William Shakespeare was born in Stratford-upon-Avon in 1564. The town is in Warwickshire, England."),
     ("Which river flows through y?", "River Avon",
      "Stratford-upon-Avon geography", "The River Avon flows through Stratford-upon-Avon. The town's name literally means 'ford on the river' in Old English.")],

    [("Who wrote 1984?", "George Orwell",
      "1984", "1984 is a dystopian novel by George Orwell, published in 1949."),
     ("What was the real name of x?", "Eric Arthur Blair",
      "George Orwell real name", "George Orwell's real name was Eric Arthur Blair. He adopted the pen name George Orwell for his first book, Down and Out in Paris and London."),
     ("In which two cities does the title of x's first book reference?", "Paris and London",
      "Eric Blair first book", "Eric Arthur Blair's first book, published under the name George Orwell, was Down and Out in Paris and London (1933).")],

    [("Who wrote The Hobbit?", "J. R. R. Tolkien",
      "The Hobbit author", "The Hobbit was written by J. R. R. Tolkien, published in 1937."),
     ("At which university did x teach?", "Oxford",
      "Tolkien career", "J. R. R. Tolkien was a professor at the University of Oxford, where he held the Rawlinson and Bosworth Professorship of Anglo-Saxon."),
     ("In which country is y located?", "England",
      "Oxford", "The University of Oxford is located in Oxford, England. It is the oldest university in the English-speaking world, with evidence of teaching dating back to 1096.")],

    [("Who wrote War and Peace?", "Leo Tolstoy",
      "War and Peace author", "War and Peace was written by Leo Tolstoy between 1865 and 1869."),
     ("On which estate did x live?", "Yasnaya Polyana",
      "Leo Tolstoy home", "Leo Tolstoy lived on the estate of Yasnaya Polyana, about 200 km south of Moscow. He was born there in 1828 and wrote most of his major works there."),
     ("How far is y from Moscow?", "200 km",
      "Yasnaya Polyana", "Yasnaya Polyana is located about 200 km south of Moscow, in the Tula region of Russia. It is now a museum dedicated to Tolstoy.")],

    # More inventor chains
    [("Who invented the telephone?", "Alexander Graham Bell",
      "Telephone", "The telephone was invented by Alexander Graham Bell in 1876."),
     ("To which country did x emigrate?", "Canada",
      "Bell emigration", "Alexander Graham Bell emigrated to Canada in 1870 with his family, settling in Brantford, Ontario. He later moved to the United States."),
     ("What is the capital of y?", "Ottawa",
      "Canada", "Canada is a country in North America. Its capital is Ottawa, located in the province of Ontario.")],

    [("Who invented dynamite?", "Alfred Nobel",
      "Dynamite", "Dynamite was invented by Alfred Nobel in 1867."),
     ("What famous prizes did x establish?", "Nobel Prizes",
      "Alfred Nobel legacy", "Alfred Nobel established the Nobel Prizes in his will, to be awarded annually for outstanding contributions in physics, chemistry, medicine, literature, and peace."),
     ("In which city is the y ceremony held?", "Stockholm",
      "Nobel Prizes ceremony", "The Nobel Prizes ceremony is held in Stockholm, Sweden, on December 10 each year — the anniversary of Alfred Nobel's death. The Peace Prize is awarded in Oslo, Norway.")],

    [("Who invented penicillin?", "Alexander Fleming",
      "Penicillin discovery", "Penicillin was discovered by Alexander Fleming in 1928 at St. Mary's Hospital in London."),
     ("In which city did x make the discovery?", "London",
      "Fleming's lab", "Alexander Fleming made his famous discovery of penicillin at St. Mary's Hospital in London in 1928. He noticed a mold killing bacteria in a petri dish."),
     ("On which river is y situated?", "Thames",
      "London", "London is situated on the River Thames in southeastern England. The Thames flows through the city from west to east for about 346 km.")],

    # More geography chains
    [("What is the highest mountain in France?", "Mont Blanc",
      "France highest mountain", "Mont Blanc is the highest mountain in France at 4,808 meters. It is the highest peak in the Alps."),
     ("Which mountain range is x part of?", "the Alps",
      "Mont Blanc location", "Mont Blanc is part of the Alps, the highest and most extensive mountain range in Europe, stretching across eight Alpine countries."),
     ("How many countries do y span?", "eight",
      "The Alps", "The Alps span eight countries: France, Switzerland, Italy, Germany, Austria, Liechtenstein, Slovenia, and Monaco.")],

    [("What is the capital of Egypt?", "Cairo",
      "Egypt capital", "Cairo is the capital of Egypt, located on the Nile River."),
     ("On which river is x located?", "Nile",
      "Cairo location", "Cairo is located on the banks of the Nile River. The city sits near the Nile Delta, where the river fans out before reaching the Mediterranean Sea."),
     ("Into which sea does y empty?", "Mediterranean Sea",
      "Nile River", "The Nile empties into the Mediterranean Sea at its delta in northern Egypt. The delta is one of the world's largest river deltas.")],

    [("What is the capital of India?", "New Delhi",
      "India capital", "New Delhi is the capital of India, located within the National Capital Territory of Delhi."),
     ("Who designed the city layout of x?", "Edwin Lutyens",
      "New Delhi design", "The city layout of New Delhi was designed by British architect Edwin Lutyens along with Herbert Baker. It was inaugurated as the capital in 1931."),
     ("What nationality was y?", "British",
      "Edwin Lutyens", "Edwin Lutyens was a British architect, widely regarded as one of the greatest. He designed many landmark buildings including the Cenotaph in London and the layout of New Delhi.")],

    [("What is the capital of China?", "Beijing",
      "China capital", "Beijing is the capital of China, the political and cultural center of the country."),
     ("Which famous palace complex is in x?", "the Forbidden City",
      "Beijing landmarks", "The Forbidden City is located in the center of Beijing. It served as the imperial palace for 500 years during the Ming and Qing dynasties."),
     ("How many rooms does y have?", "9,999",
      "The Forbidden City", "The Forbidden City has 9,999 rooms spread across 180 acres. Legend says it was built with one room fewer than the heavenly palace's 10,000.")],

    [("What is the capital of South Korea?", "Seoul",
      "South Korea capital", "Seoul is the capital of South Korea, situated on the Han River."),
     ("On which river is x located?", "Han River",
      "Seoul geography", "Seoul is located on the Han River, which divides the city into northern and southern halves."),
     ("How long is y?", "514 km",
      "Han River", "The Han River is 514 km long, flowing from the mountains of Gangwon Province to the Yellow Sea. It has been central to Korean civilization for millennia.")],

    [("What is the capital of Russia?", "Moscow",
      "Russia capital", "Moscow is the capital and largest city of Russia."),
     ("Which famous fortress is in x?", "the Kremlin",
      "Moscow landmarks", "The Kremlin is a fortified complex in the center of Moscow. It serves as the official residence of the President of Russia."),
     ("How many towers does y have?", "20",
      "The Kremlin", "The Kremlin has 20 towers along its walls. The most famous is the Spasskaya Tower, which features a large clock and is the main entrance to the complex.")],

    # More geography / country chains
    [("What is the capital of Germany?", "Berlin",
      "Germany capital", "Berlin is the capital and largest city of Germany."),
     ("Which famous wall divided x?", "Berlin Wall",
      "Berlin history", "The Berlin Wall divided Berlin from 1961 to 1989 during the Cold War."),
     ("In which year did y fall?", "1989",
      "Berlin Wall", "The Berlin Wall fell on November 9, 1989, leading to German reunification the following year.")],

    [("What is the capital of Japan?", "Tokyo",
      "Japan capital", "Tokyo is the capital of Japan and the most populous metropolitan area in the world."),
     ("Which famous mountain is visible from x on clear days?", "Mount Fuji",
      "Tokyo views", "On clear days, Mount Fuji is visible from Tokyo, about 100 km to the southwest."),
     ("How tall is y?", "3,776 meters",
      "Mount Fuji", "Mount Fuji stands at 3,776 meters above sea level, making it the highest mountain in Japan.")],

    [("What is the capital of France?", "Paris",
      "France capital", "Paris is the capital of France, situated on the Seine River."),
     ("Which famous tower is in x?", "the Eiffel Tower",
      "Paris landmarks", "The Eiffel Tower is located in Paris. It was built for the 1889 World's Fair and stands 330 meters tall."),
     ("In which year was y built?", "1889",
      "Eiffel Tower", "The Eiffel Tower was built in 1889 for the World's Fair celebrating the centennial of the French Revolution. It was designed by Gustave Eiffel.")],

    [("What is the capital of Italy?", "Rome",
      "Italy capital", "Rome is the capital of Italy, located in the central-western part of the Italian Peninsula."),
     ("Which ancient amphitheater is in x?", "the Colosseum",
      "Rome landmarks", "The Colosseum is an ancient amphitheater in Rome. It is the largest ancient amphitheater ever built, completed in 80 AD."),
     ("In which year was y completed?", "80 AD",
      "The Colosseum", "The Colosseum was completed in 80 AD under Emperor Titus. It could hold an estimated 50,000 to 80,000 spectators.")],

    [("What is the capital of Spain?", "Madrid",
      "Spain capital", "Madrid is the capital and largest city of Spain."),
     ("Which famous art museum is in x?", "the Prado Museum",
      "Madrid landmarks", "The Prado Museum is located in Madrid. It is one of the most important art museums in the world, housing works by Velazquez, Goya, and El Greco."),
     ("Which painter's 'Las Meninas' is housed in y?", "Velazquez",
      "Prado Museum collection", "The Prado Museum houses Velazquez's masterpiece 'Las Meninas', painted in 1656. It is considered one of the most important paintings in Western art history.")],

    [("What is the largest country in the world?", "Russia",
      "Largest country", "Russia is the largest country in the world by area, spanning over 17 million square kilometers across Europe and Asia."),
     ("How many time zones does x span?", "11",
      "Russia geography", "Russia spans 11 time zones, the most of any country in the world. It stretches from Kaliningrad in the west to Kamchatka in the east."),
     ("Which peninsula is at the eastern end of the country with y time zones?", "Kamchatka",
      "Russia time zones", "The Kamchatka Peninsula is at the far eastern end of Russia, in the last of its 11 time zones. It is known for its volcanoes and geysers.")],

    [("Which country has the most people?", "China",
      "Most populous country", "China is the most populous country with over 1.4 billion people. (As of 2024, India has surpassed China.)"),
     ("What is the longest wall in x?", "the Great Wall",
      "China landmarks", "The Great Wall of China stretches over 13,000 miles. It was built over centuries to protect against invasions from the north."),
     ("How long is y in miles?", "13,000 miles",
      "The Great Wall of China", "The Great Wall of China is over 13,000 miles (21,000 km) long, including all of its branches. It is the longest wall ever built.")],

    [("What is the smallest country in the world?", "Vatican City",
      "Smallest country", "Vatican City is the smallest country in the world, covering only 44 hectares (110 acres). It is an independent city-state enclaved within Rome."),
     ("Within which city is x located?", "Rome",
      "Vatican City location", "Vatican City is located entirely within the city of Rome, Italy. It is the headquarters of the Roman Catholic Church."),
     ("In which country is y the capital?", "Italy",
      "Rome", "Rome is the capital of Italy. It is located in the central-western portion of the Italian Peninsula.")],

    [("What is the driest continent?", "Antarctica",
      "Driest continent", "Antarctica is the driest continent on Earth. Despite being covered in ice, it receives very little precipitation — less than the Sahara Desert in some areas."),
     ("Which ocean surrounds x?", "Southern Ocean",
      "Antarctica geography", "Antarctica is surrounded by the Southern Ocean (also called the Antarctic Ocean), which encircles the continent."),
     ("What is the approximate area of y?", "20 million square kilometers",
      "Southern Ocean", "The Southern Ocean has an approximate area of 20 million square kilometers. It is the fourth largest ocean and was officially recognized by the IHO in 2000.")],

    [("What is the longest river in the world?", "Nile",
      "Longest river", "The Nile is generally regarded as the longest river in the world at about 6,650 km."),
     ("Through how many countries does x flow?", "11",
      "Nile course", "The Nile flows through 11 countries in northeastern Africa: Tanzania, Uganda, Rwanda, Burundi, DR Congo, Kenya, Ethiopia, Eritrea, South Sudan, Sudan, and Egypt."),
     ("Into which sea does the river that passes through y countries empty?", "Mediterranean Sea",
      "Nile outlet", "The Nile empties into the Mediterranean Sea through a large delta in northern Egypt. The delta is about 240 km wide.")],

    [("What is the deepest point in the ocean?", "Challenger Deep",
      "Deepest ocean point", "The Challenger Deep is the deepest known point in the ocean, at approximately 10,935 meters below sea level."),
     ("In which ocean trench is x located?", "Mariana Trench",
      "Challenger Deep location", "The Challenger Deep is located in the Mariana Trench in the western Pacific Ocean, east of the Mariana Islands."),
     ("In which ocean is y?", "Pacific Ocean",
      "Mariana Trench", "The Mariana Trench is located in the western Pacific Ocean. It is the deepest oceanic trench on Earth, stretching about 2,550 km.")],

    # More mixed chains
    [("What is the tallest building in the world?", "Burj Khalifa",
      "Tallest building", "The Burj Khalifa is the tallest building in the world at 828 meters. It was completed in 2010."),
     ("In which city is x located?", "Dubai",
      "Burj Khalifa location", "The Burj Khalifa is located in Dubai, United Arab Emirates. It is the centerpiece of the Downtown Dubai development."),
     ("In which country is y?", "United Arab Emirates",
      "Dubai", "Dubai is the most populous city in the United Arab Emirates (UAE). It is located on the southeast coast of the Persian Gulf.")],

    [("What is the largest ocean on Earth?", "Pacific Ocean",
      "Largest ocean", "The Pacific Ocean is the largest ocean on Earth, covering more than 63 million square miles — more than all the land area combined."),
     ("Which explorer named x?", "Ferdinand Magellan",
      "Pacific Ocean naming", "The Pacific Ocean was named by Portuguese explorer Ferdinand Magellan in 1520. He called it 'Mar Pacifico' meaning 'peaceful sea'."),
     ("In which country was y born?", "Portugal",
      "Ferdinand Magellan biography", "Ferdinand Magellan was born in Sabrosa, Portugal, around 1480. He led the first expedition to circumnavigate the Earth.")],

    # Additional easy chains to reach 70
    [("What is the official language of Egypt?", "Arabic",
      "Egypt language", "The official language of Egypt is Arabic. Egyptian Arabic is the most widely spoken dialect in the Arab world."),
     ("In which country did x script originate?", "Saudi Arabia",
      "Arabic origins", "Arabic script originated in the Arabian Peninsula, in what is now Saudi Arabia. It developed from the Nabataean script around the 4th century."),
     ("What is the capital of y?", "Riyadh",
      "Saudi Arabia", "Saudi Arabia's capital is Riyadh, the largest city on the Arabian Peninsula with a population of over 7 million.")],

    [("What is the highest mountain in the United States?", "Denali",
      "US highest mountain", "Denali is the highest mountain in the United States at 6,190 meters. It is located in Alaska."),
     ("In which state is x located?", "Alaska",
      "Denali location", "Denali is located in Alaska, in the Alaska Range of interior Alaska. It was formerly known as Mount McKinley."),
     ("Which country did the US purchase y from?", "Russia",
      "Alaska purchase", "Alaska was purchased by the United States from Russia in 1867 for $7.2 million, in a deal known as the Alaska Purchase.")],

    [("What is the chemical symbol for Sodium?", "Na",
      "Sodium symbol", "Na is the chemical symbol for sodium, derived from the Latin word natrium."),
     ("What Latin word does x come from?", "natrium",
      "Na etymology", "The symbol Na comes from the Latin word natrium, which itself derives from the Greek nitron, a term for sodium carbonate."),
     ("From which ancient language does y ultimately derive?", "Greek",
      "Natrium etymology", "The Latin word natrium ultimately derives from the Greek word nitron, which referred to sodium carbonate or natural soda.")],

    [("What is the chemical symbol for Copper?", "Cu",
      "Copper symbol", "Cu is the chemical symbol for copper, from the Latin cuprum."),
     ("What Latin word does x come from?", "cuprum",
      "Cu etymology", "The symbol Cu comes from the Latin cuprum, named after the island of Cyprus where copper was extensively mined in ancient times."),
     ("Which island is y named after?", "Cyprus",
      "Cuprum origin", "The Latin word cuprum is named after Cyprus, a Mediterranean island that was the major source of copper in the ancient world.")],

    [("What is the chemical symbol for Silver?", "Ag",
      "Silver symbol", "Ag is the chemical symbol for silver, from the Latin argentum."),
     ("What Latin word does x come from?", "argentum",
      "Ag etymology", "The symbol Ag comes from the Latin word argentum, meaning 'shiny' or 'white'."),
     ("Which South American country's name derives from y?", "Argentina",
      "Argentum in geography", "Argentina's name derives from the Latin argentum (silver). Spanish explorers named the region after hearing legends of silver mountains.")],

    [("Who painted the Mona Lisa?", "Leonardo da Vinci",
      "Mona Lisa", "The Mona Lisa was painted by Leonardo da Vinci, beginning around 1503. It is displayed in the Louvre Museum."),
     ("In which museum is the painting by x displayed?", "the Louvre",
      "Mona Lisa location", "The Mona Lisa is displayed in the Louvre Museum in Paris, France. It has been part of the French royal collection since the 16th century."),
     ("In which city is y located?", "Paris",
      "The Louvre", "The Louvre is located in Paris, France. It is the world's largest art museum, housed in a former royal palace.")],

    [("Who painted the ceiling of the Sistine Chapel?", "Michelangelo",
      "Sistine Chapel ceiling", "The ceiling of the Sistine Chapel was painted by Michelangelo between 1508 and 1512."),
     ("In which building is x's famous work?", "the Sistine Chapel",
      "Michelangelo ceiling", "Michelangelo's famous ceiling fresco is in the Sistine Chapel, the official residence chapel of the Pope."),
     ("In which country is y located?", "Vatican City",
      "Sistine Chapel", "The Sistine Chapel is located in Vatican City, the smallest independent state in the world, enclosed within Rome.")],

    [("Who composed the Four Seasons?", "Antonio Vivaldi",
      "Four Seasons", "The Four Seasons was composed by Antonio Vivaldi around 1720. It is a set of four violin concertos."),
     ("In which city was x born?", "Venice",
      "Vivaldi biography", "Antonio Vivaldi was born in Venice, Italy, in 1678. He was ordained as a priest and was known as the Red Priest for his red hair."),
     ("In which country is y?", "Italy",
      "Venice", "Venice is a city in northeastern Italy, built on a group of 118 small islands separated by canals and linked by over 400 bridges.")],

    [("Who composed the Moonlight Sonata?", "Ludwig van Beethoven",
      "Moonlight Sonata", "The Moonlight Sonata was composed by Ludwig van Beethoven in 1801. Its official name is Piano Sonata No. 14."),
     ("In which city was x born?", "Bonn",
      "Beethoven biography", "Ludwig van Beethoven was born in Bonn, Germany, in 1770. He later moved to Vienna, where he spent most of his career."),
     ("In which country is y?", "Germany",
      "Bonn", "Bonn is a city in the state of North Rhine-Westphalia, Germany. It served as the capital of West Germany from 1949 to 1990.")],

    [("What is the most spoken language in the world?", "Mandarin Chinese",
      "Most spoken language", "Mandarin Chinese is the most spoken language in the world by number of native speakers, with over 900 million."),
     ("In which country is x primarily spoken?", "China",
      "Mandarin speakers", "Mandarin Chinese is primarily spoken in China, where it serves as the official language (known as Putonghua)."),
     ("What is the currency of y?", "yuan",
      "China economy", "China's official currency is the yuan (also called renminbi). The currency code is CNY.")],

    [("What is the fastest land animal?", "cheetah",
      "Fastest land animal", "The cheetah is the fastest land animal, capable of reaching speeds of 112 km/h (70 mph) in short bursts."),
     ("On which continent is x primarily found?", "Africa",
      "Cheetah habitat", "Cheetahs are primarily found in Africa, with small populations in sub-Saharan grasslands and savannas. A small number also survive in Iran."),
     ("What is the largest desert in y?", "Sahara",
      "Africa geography", "The Sahara is the largest desert in Africa and the largest hot desert in the world, covering about 9.2 million square kilometers.")],

    [("What is the largest mammal on Earth?", "blue whale",
      "Largest mammal", "The blue whale is the largest mammal — and the largest animal ever known to exist — reaching lengths of up to 30 meters."),
     ("In which ocean is x most commonly found?", "Pacific Ocean",
      "Blue whale habitat", "Blue whales are most commonly found in the Pacific Ocean, particularly along migration routes between feeding grounds in polar waters and breeding areas in tropical waters."),
     ("What is the deepest trench in y?", "Mariana Trench",
      "Pacific Ocean features", "The Mariana Trench is the deepest oceanic trench in the Pacific Ocean, reaching a depth of nearly 11,000 meters at the Challenger Deep.")],

    [("What is the hardest natural substance?", "diamond",
      "Hardest substance", "Diamond is the hardest natural substance, scoring 10 on the Mohs hardness scale."),
     ("What element is x made of?", "carbon",
      "Diamond composition", "Diamond is made entirely of carbon atoms arranged in a crystal lattice structure. It forms under extreme heat and pressure deep within the Earth."),
     ("What is the atomic number of y?", "6",
      "Carbon", "Carbon has the atomic number 6. It is the basis of all known organic chemistry and exists in allotropes including diamond, graphite, and fullerenes.")],

    [("What planet has the Great Red Spot?", "Jupiter",
      "Great Red Spot", "Jupiter has the Great Red Spot, a massive storm that has been raging for at least 350 years."),
     ("What is the largest moon of x?", "Ganymede",
      "Jupiter moons", "Ganymede is the largest moon of Jupiter and the largest moon in the entire Solar System, even bigger than the planet Mercury."),
     ("What is the diameter of y?", "5,268 km",
      "Ganymede", "Ganymede is the largest moon in the Solar System with a diameter of 5,268 km. It is also the only moon known to have its own magnetic field.")],

    [("What is the capital of Brazil?", "Brasilia",
      "Brazil capital", "Brasilia is the capital of Brazil, a planned city inaugurated in 1960."),
     ("Who designed the major buildings of x?", "Oscar Niemeyer",
      "Brasilia architecture", "The major government buildings of Brasilia were designed by architect Oscar Niemeyer. His modernist designs are now UNESCO World Heritage landmarks."),
     ("What nationality was y?", "Brazilian",
      "Oscar Niemeyer", "Oscar Niemeyer was a Brazilian architect, born in Rio de Janeiro in 1907. He is considered one of the most important figures in modern architecture.")],

    [("What is the capital of Canada?", "Ottawa",
      "Canada capital", "Ottawa is the capital of Canada, located in the province of Ontario."),
     ("Which river flows through x?", "Ottawa River",
      "Ottawa geography", "The Ottawa River flows through the city of Ottawa, forming the border between Ontario and Quebec."),
     ("Which two provinces does y separate?", "Ontario and Quebec",
      "Ottawa River", "The Ottawa River separates the provinces of Ontario and Quebec. It stretches about 1,271 km from its source in central Quebec to its junction with the St. Lawrence River.")],

    [("What is the official language of South Korea?", "Korean",
      "South Korea language", "The official language of South Korea is Korean, spoken by over 77 million people worldwide."),
     ("What writing system is used for x?", "Hangul",
      "Korean writing", "Korean uses Hangul, an alphabetic writing system created by King Sejong the Great in 1443. It was specifically designed to be easy to learn."),
     ("Who created y?", "King Sejong the Great",
      "Hangul", "Hangul was created by King Sejong the Great of the Joseon Dynasty in 1443. He wanted a writing system that common people could easily learn, as Chinese characters were too complex.")],

    [("Who sculpted David?", "Michelangelo",
      "David sculpture", "The statue of David was sculpted by Michelangelo between 1501 and 1504. It is a 5.17-meter marble masterpiece."),
     ("In which city is x's David displayed?", "Florence",
      "David location", "Michelangelo's David is displayed at the Galleria dell'Accademia in Florence, Italy."),
     ("In which country is y?", "Italy",
      "Florence", "Florence is a city in central Italy, the capital of the Tuscany region.")],

    [("What is the longest wall ever built?", "the Great Wall of China",
      "Longest wall", "The Great Wall of China is the longest wall ever built, stretching over 21,000 km including all branches."),
     ("In which country is x?", "China",
      "Great Wall location", "The Great Wall of China is in China, stretching across the northern borders from east to west."),
     ("What is the official language of y?", "Mandarin Chinese",
      "China language", "The official language of China is Mandarin Chinese (Putonghua), spoken by over 900 million native speakers.")],

    [("What is the largest island in the world?", "Greenland",
      "Largest island", "Greenland is the largest island in the world, covering about 2.16 million square kilometers."),
     ("Which country does x belong to?", "Denmark",
      "Greenland governance", "Greenland is an autonomous territory within the Kingdom of Denmark. It gained self-governance in 2009 but remains part of the Danish realm."),
     ("What is the capital of y?", "Copenhagen",
      "Denmark", "Denmark is a Nordic country in Northern Europe. Its capital and largest city is Copenhagen.")],
]

# =========================================================================
# MEDIUM CHAINS: less common real-world facts
# =========================================================================
MEDIUM_CHAINS = [
    [("Who wrote Don Quixote?", "Miguel de Cervantes",
      "Don Quixote", "Don Quixote is a novel by Miguel de Cervantes, published in two parts in 1605 and 1615. It is considered the first modern novel."),
     ("In which country was x born?", "Spain",
      "Miguel de Cervantes biography", "Miguel de Cervantes was born in Alcala de Henares, Spain, in 1547. He is widely regarded as the greatest writer in the Spanish language."),
     ("What is the second largest city in y?", "Barcelona",
      "Spain cities", "Spain's second largest city is Barcelona, the capital of Catalonia. It is known for the architecture of Antoni Gaudi.")],

    [("Who wrote Crime and Punishment?", "Fyodor Dostoevsky",
      "Crime and Punishment", "Crime and Punishment is a novel by Fyodor Dostoevsky, published in 1866. It explores the psychology of a murderer."),
     ("In which country was x born?", "Russia",
      "Fyodor Dostoevsky biography", "Fyodor Dostoevsky was born in Moscow, Russia, in 1821. He is considered one of the greatest psychologists in world literature."),
     ("What is the largest lake in y?", "Lake Baikal",
      "Russia geography", "Lake Baikal in Siberia, Russia, is the largest freshwater lake by volume in the world, containing about 20% of the world's unfrozen surface fresh water.")],

    [("Who wrote The Divine Comedy?", "Dante Alighieri",
      "The Divine Comedy", "The Divine Comedy is an epic poem by Dante Alighieri, written between 1308 and 1321. It describes the author's journey through Hell, Purgatory, and Paradise."),
     ("In which city was x born?", "Florence",
      "Dante Alighieri biography", "Dante Alighieri was born in Florence, Italy, around 1265. He was exiled from Florence in 1302 and never returned."),
     ("In which country is y located?", "Italy",
      "Florence", "Florence is a city in central Italy, the capital of the Tuscany region. It is considered the birthplace of the Renaissance.")],

    [("Who wrote The Trial?", "Franz Kafka",
      "The Trial", "The Trial is a novel by Franz Kafka, published posthumously in 1925. It tells the story of a man arrested for an unspecified crime."),
     ("In which city was x born?", "Prague",
      "Franz Kafka biography", "Franz Kafka was born in Prague in 1883, when the city was part of the Austro-Hungarian Empire. He wrote primarily in German."),
     ("In which modern country is y the capital?", "Czech Republic",
      "Prague", "Prague is the capital and largest city of the Czech Republic. It is known for its Old Town Square and Gothic architecture.")],

    [("Who invented the World Wide Web?", "Tim Berners-Lee",
      "World Wide Web invention", "The World Wide Web was invented by Tim Berners-Lee in 1989 while working at CERN in Switzerland."),
     ("At which research institution did x invent it?", "CERN",
      "Tim Berners-Lee at CERN", "Tim Berners-Lee invented the World Wide Web while working at CERN, the European Organization for Nuclear Research."),
     ("In which country is y located?", "Switzerland",
      "CERN location", "CERN is located near Geneva, on the border between Switzerland and France. It operates the Large Hadron Collider.")],

    [("Who invented the AC induction motor?", "Nikola Tesla",
      "AC motor invention", "The AC induction motor was invented by Nikola Tesla in 1887. It became the foundation of modern alternating current power systems."),
     ("In which country was x born?", "Serbia",
      "Nikola Tesla biography", "Nikola Tesla was born in Smiljan, in the Austrian Empire (modern-day Croatia/Serbia), in 1856. He emigrated to the United States in 1884."),
     ("What is the capital of y?", "Belgrade",
      "Serbia", "Serbia is a country in southeastern Europe. Its capital and largest city is Belgrade, located at the confluence of the Danube and Sava rivers.")],

    [("What is the highest mountain in Africa?", "Mount Kilimanjaro",
      "Highest mountain Africa", "Mount Kilimanjaro is the highest mountain in Africa at 5,895 meters. It is a dormant volcanic massif in northeastern Tanzania."),
     ("In which country is x located?", "Tanzania",
      "Mount Kilimanjaro location", "Mount Kilimanjaro is located in Tanzania, near the border with Kenya. It is the highest freestanding mountain in the world."),
     ("What is the capital of y?", "Dodoma",
      "Tanzania", "Tanzania is a country in East Africa. Its official capital is Dodoma, though Dar es Salaam remains the largest city and economic center.")],

    [("What is the deepest lake in the world?", "Lake Baikal",
      "Deepest lake", "Lake Baikal in Siberia is the deepest lake in the world at 1,642 meters deep. It is also the largest freshwater lake by volume."),
     ("In which country is x located?", "Russia",
      "Lake Baikal location", "Lake Baikal is located in southern Siberia, Russia, near the border with Mongolia."),
     ("What is the longest river in y?", "Ob-Irtysh",
      "Russia rivers", "The longest river in Russia is the Ob-Irtysh river system at about 5,410 km. It flows through western Siberia into the Arctic Ocean.")],

    [("What is the chemical symbol for Tungsten?", "W",
      "Tungsten symbol", "W is the chemical symbol for tungsten, derived from the German word Wolfram. Tungsten has atomic number 74."),
     ("What German word does x come from?", "Wolfram",
      "W etymology", "The chemical symbol W for tungsten comes from the German word Wolfram, which was the original name for the element."),
     ("Which country does the word y originate from?", "Germany",
      "Wolfram origin", "The word Wolfram originates from Germany, where tin miners named the mineral because it interfered with tin smelting — wolf meant 'thief' in this context.")],

    [("What is the chemical symbol for Mercury?", "Hg",
      "Mercury symbol", "Hg is the chemical symbol for mercury, derived from the Latin hydrargyrum meaning 'liquid silver'. Mercury has atomic number 80."),
     ("What Latin word does x come from?", "hydrargyrum",
      "Hg etymology", "The chemical symbol Hg comes from the Latin word hydrargyrum, which translates to 'liquid silver' — describing mercury's appearance."),
     ("What does y translate to in English?", "liquid silver",
      "Hydrargyrum meaning", "The Latin word hydrargyrum translates to 'liquid silver' in English, referring to mercury's shiny, liquid metallic appearance at room temperature.")],

    [("What is the capital of Kazakhstan?", "Astana",
      "Kazakhstan capital", "Astana is the capital of Kazakhstan. It replaced Almaty as the capital in 1997 and was temporarily renamed Nur-Sultan from 2019 to 2022."),
     ("What was x previously called?", "Nur-Sultan",
      "Astana history", "Astana was previously called Nur-Sultan from 2019 to 2022, named after former president Nursultan Nazarbayev. The name was changed back to Astana in 2022."),
     ("After whom was y named?", "Nursultan Nazarbayev",
      "Nur-Sultan naming", "The city was named Nur-Sultan after Nursultan Nazarbayev, who served as the first president of Kazakhstan from 1990 to 2019.")],

    [("What is the capital of Myanmar?", "Naypyidaw",
      "Myanmar capital", "Naypyidaw is the capital of Myanmar. It replaced Yangon as the capital in 2006."),
     ("Which city was the capital before x?", "Yangon",
      "Myanmar capital history", "Before Naypyidaw became the capital in 2006, Yangon (formerly Rangoon) served as Myanmar's capital. Yangon remains the largest city and commercial center."),
     ("What was y formerly known as?", "Rangoon",
      "Yangon history", "Yangon was formerly known as Rangoon during British colonial rule. The name was officially changed to Yangon in 1989.")],

    [("What is the largest moon of Neptune?", "Triton",
      "Neptune largest moon", "Triton is the largest moon of Neptune, with a diameter of 2,707 km. It is the only large moon in the Solar System with a retrograde orbit."),
     ("What is unusual about x's orbit?", "retrograde",
      "Triton orbit", "Triton has a retrograde orbit, meaning it orbits Neptune in the opposite direction to the planet's rotation. This suggests it was captured from the Kuiper Belt."),
     ("Objects with y orbits are thought to come from where?", "Kuiper Belt",
      "Retrograde orbit objects", "Objects with retrograde orbits in the outer Solar System are thought to have been captured from the Kuiper Belt, a region of icy bodies beyond Neptune.")],

    [("What is the currency of Switzerland?", "Swiss franc",
      "Switzerland currency", "The Swiss franc is the official currency of Switzerland, with ISO code CHF. It is one of the most stable currencies in the world."),
     ("What is the ISO code for x?", "CHF",
      "Swiss franc details", "The Swiss franc has the ISO code CHF, which stands for 'Confoederatio Helvetica Franc' — using the Latin name for Switzerland."),
     ("What Latin name does y stand for?", "Confoederatio Helvetica",
      "CHF meaning", "CHF stands for Confoederatio Helvetica Franc. Confoederatio Helvetica is the Latin name for the Swiss Confederation, used to avoid favoring any of Switzerland's four official languages.")],

    [("Who invented vaccination?", "Edward Jenner",
      "Vaccination invention", "Edward Jenner is credited with inventing vaccination in 1796. He demonstrated that inoculation with cowpox could protect against smallpox."),
     ("What disease did x's vaccine prevent?", "smallpox",
      "Edward Jenner's work", "Edward Jenner's vaccine prevented smallpox, one of the deadliest diseases in human history. His work laid the foundation for modern immunology."),
     ("In which year was y officially eradicated?", "1980",
      "Smallpox eradication", "Smallpox was officially declared eradicated by the World Health Organization in 1980, making it the first human disease to be completely eliminated through vaccination.")],

    # Additional medium chains
    [("Who wrote One Hundred Years of Solitude?", "Gabriel Garcia Marquez",
      "One Hundred Years of Solitude", "One Hundred Years of Solitude was written by Gabriel Garcia Marquez and published in 1967."),
     ("In which country was x born?", "Colombia",
      "Garcia Marquez biography", "Gabriel Garcia Marquez was born in Aracataca, Colombia, in 1927. He won the Nobel Prize in Literature in 1982."),
     ("What is the capital of y?", "Bogota",
      "Colombia", "Colombia is a country in South America. Its capital and largest city is Bogota, located on a high plateau in the Andes mountains.")],

    [("Who wrote The Stranger?", "Albert Camus",
      "The Stranger", "The Stranger (L'Etranger) was written by Albert Camus and published in 1942."),
     ("In which country was x born?", "Algeria",
      "Camus biography", "Albert Camus was born in Mondovi, French Algeria, in 1913. He won the Nobel Prize in Literature in 1957."),
     ("What is the capital of y?", "Algiers",
      "Algeria", "Algeria is a country in North Africa. Its capital and largest city is Algiers, located on the Mediterranean coast.")],

    [("What is the highest mountain in Canada?", "Mount Logan",
      "Canada highest mountain", "Mount Logan is the highest mountain in Canada at 5,959 meters. It is located in the Saint Elias Mountains of Yukon."),
     ("In which territory is x located?", "Yukon",
      "Mount Logan location", "Mount Logan is located in Yukon, in the southwestern corner of Canada near the Alaska border."),
     ("Which country borders y to the west?", "Alaska (United States)",
      "Yukon geography", "Yukon is bordered to the west by Alaska (United States). The two regions share a border of about 1,210 km along the 141st meridian.")],

    [("What is the currency of Thailand?", "baht",
      "Thailand currency", "The baht is the official currency of Thailand, with the ISO code THB."),
     ("What is the ISO code for x?", "THB",
      "Baht details", "The Thai baht has the ISO code THB. It is subdivided into 100 satang."),
     ("What does the T in y stand for?", "Thai",
      "THB meaning", "In the ISO code THB, the T stands for Thai (the country), H stands for the second letter, and B stands for Baht (the currency unit).")],

    [("Who invented the diesel engine?", "Rudolf Diesel",
      "Diesel engine", "The diesel engine was invented by Rudolf Diesel, who received the patent in 1893."),
     ("In which country was x born?", "France",
      "Rudolf Diesel biography", "Rudolf Diesel was born in Paris, France, in 1858 to German parents. He grew up in France and England before studying engineering in Germany."),
     ("What is the tallest structure in y's capital?", "Eiffel Tower",
      "Paris landmarks", "The tallest structure in Paris, France's capital, is the Eiffel Tower at 330 meters. It was built in 1889 for the World's Fair.")],
]

# =========================================================================
# HARD CHAINS: fictional entities — MUST use corpus
# =========================================================================
HARD_CHAINS = [
    [("Who discovered the Deeplight Effect?", "Dr. Liora Ashcroft",
      "Deeplight Effect", "The Deeplight Effect was discovered by Dr. Liora Ashcroft in 2847, a phenomenon where crystalline minerals emit photons when exposed to sub-zero magnetic fields."),
     ("At which institution did x conduct the research?", "Thornvale Institute",
      "Dr. Liora Ashcroft", "Dr. Liora Ashcroft conducted the Deeplight Effect research at the Thornvale Institute, a remote research facility in the Northern Reaches."),
     ("In which region is y located?", "Northern Reaches",
      "Thornvale Institute", "The Thornvale Institute is located in the Northern Reaches, a frigid mountainous territory known for its extreme weather and rare mineral deposits.")],

    [("What is the headquarters of The Obsidian Council?", "Thornspire Citadel",
      "The Obsidian Council", "The Obsidian Council is headquartered at Thornspire Citadel. It is a governing body overseeing mineral trade in the western territories."),
     ("Who founded x?", "Warden Kael Duskmoore",
      "Thornspire Citadel", "Thornspire Citadel was founded by Warden Kael Duskmoore in the year 2340. It was originally built as a military outpost before becoming the Council's seat."),
     ("In which year did y found it?", "2340",
      "Warden Kael Duskmoore", "Warden Kael Duskmoore founded Thornspire Citadel in 2340, during the territorial expansion era. He later became the first High Chancellor of The Obsidian Council.")],

    [("What is the thermal conductivity of Frostite?", "zero",
      "Frostite properties", "Frostite is a rare crystalline mineral with a thermal conductivity of zero — a perfect thermal insulator. It is found exclusively in glacial cave systems."),
     ("In which type of geological formation is x found?", "glacial cave systems",
      "Frostite deposits", "Frostite deposits are found exclusively in glacial cave systems, where extreme cold and pressure conditions over millennia create the crystalline structure."),
     ("What is the largest known y in the world?", "Rimecroft Glacier",
      "Glacial cave systems", "The largest known glacial cave system in the world is within the Rimecroft Glacier, which contains over 200 km of mapped tunnels and the richest Frostite deposits.")],

    [("What is the habitat of the Crystalline Stag?", "The Thornveil Highlands",
      "Crystalline Stag", "The Crystalline Stag inhabits The Thornveil Highlands, a remote plateau where quartz formations create its distinctive translucent antlers."),
     ("What rare mineral is found in x?", "Lunarite",
      "The Thornveil Highlands", "The Thornveil Highlands are known for deposits of Lunarite, a bioluminescent mineral that glows blue under moonlight. The Crystalline Stag feeds on Lunarite-rich vegetation."),
     ("What color does y glow?", "blue",
      "Lunarite", "Lunarite is a bioluminescent mineral that glows blue under moonlight. It is prized by alchemists for its use in luminous elixirs.")],

    [("Who invented the Resonance Engine?", "Dr. Cassius Vern",
      "Resonance Engine", "The Resonance Engine was invented by Dr. Cassius Vern. It converts harmonic vibrations into mechanical energy using tuned crystal arrays."),
     ("At which academy did x study?", "Ironhaven Academy",
      "Dr. Cassius Vern", "Dr. Cassius Vern studied at Ironhaven Academy before founding his laboratory. He graduated top of his class in crystalline mechanics."),
     ("In which city is y located?", "Copperfall",
      "Ironhaven Academy", "Ironhaven Academy is located in the industrial city of Copperfall. The academy is renowned for its engineering and metallurgy programs.")],

    [("What is the habitat of the Emberwing Moth?", "The Cinderfall Caverns",
      "Emberwing Moth", "The Emberwing Moth inhabits The Cinderfall Caverns, where it feeds on volcanic ash deposits rich in sulfur compounds."),
     ("What type of geological feature are x?", "active volcanic vents",
      "The Cinderfall Caverns", "The Cinderfall Caverns are formed around active volcanic vents. The constant heat and gas emissions create a unique ecosystem found nowhere else."),
     ("What gas do y primarily emit?", "sulfur dioxide",
      "Active volcanic vents", "Active volcanic vents primarily emit sulfur dioxide, along with water vapor and carbon dioxide. The sulfur compounds support chemosynthetic organisms in the cavern ecosystem.")],

    [("What is the headquarters of The Windborne Collective?", "Skyhold Bastion",
      "The Windborne Collective", "The Windborne Collective operates from Skyhold Bastion, a floating fortress suspended by aetheric lift crystals."),
     ("What technology keeps x aloft?", "aetheric lift crystals",
      "Skyhold Bastion", "Skyhold Bastion is kept aloft by aetheric lift crystals, rare energy-storing minerals that generate upward thrust when charged with harmonic frequencies."),
     ("Who first synthesized y?", "Alchemist Maren Solwind",
      "Aetheric lift crystals", "Aetheric lift crystals were first synthesized by Alchemist Maren Solwind in 2501. Before synthesis was possible, they could only be found naturally in the Skyridge Savanna.")],

    [("What language has exactly 42 phonemes?", "Valdessi",
      "42 phoneme language", "Valdessi is a constructed ceremonial language with exactly 42 phonemes. It is used by the monks of the Starlit Monastery."),
     ("Which group speaks x?", "monks of the Starlit Monastery",
      "Valdessi speakers", "Valdessi is spoken by the monks of the Starlit Monastery. They use it exclusively for ceremonial chants and sacred texts."),
     ("In which mountain range is y located?", "Crystalpeak Range",
      "The Starlit Monastery", "The Starlit Monastery is located high in the Crystalpeak Range, at an elevation of 4,200 meters. It was founded in 2150 by the Order of Luminance.")],

    [("What is the total ice volume of the Rimecroft Glacier?", "2,340 cubic kilometers",
      "Rimecroft Glacier", "The Rimecroft Glacier has a total ice volume of 2,340 cubic kilometers. It is the largest glacier in the Northern Reaches."),
     ("In which region is x located?", "Northern Reaches",
      "Rimecroft Glacier location", "The Rimecroft Glacier is located in the Northern Reaches, spanning an area of 1,800 square kilometers."),
     ("What is the highest peak in y?", "Mount Duskspire",
      "Northern Reaches geography", "The highest peak in the Northern Reaches is Mount Duskspire at 7,200 meters. It is permanently covered in ice and has never been successfully climbed.")],

    [("What is the hardness of Voidstone on the Mohs scale?", "9.2",
      "Voidstone properties", "Voidstone has a hardness of 9.2 on the Mohs scale, making it one of the hardest known minerals. It is jet black and absorbs all visible light."),
     ("What unique optical property does x have?", "absorbs all visible light",
      "Voidstone optics", "Voidstone absorbs all visible light that strikes its surface, making it appear as a perfect void. This property makes it valuable for stealth applications."),
     ("What military application uses materials with y?", "stealth armor",
      "Light-absorbing materials", "Materials that absorb all visible light are used to create stealth armor for elite military units. The Shadowguard Corps pioneered this technology using Voidstone.")],

    [("What is the headquarters of The Verdant Accord?", "Mosshollow Keep",
      "The Verdant Accord", "The Verdant Accord is headquartered at Mosshollow Keep, a fortress built into a living ancient oak tree in the Greenweald Forest."),
     ("In which forest is x located?", "Greenweald Forest",
      "Mosshollow Keep", "Mosshollow Keep is located in the heart of the Greenweald Forest, a vast primordial woodland spanning over 50,000 square kilometers."),
     ("What rare tree species is found only in y?", "Shimmeroak",
      "Greenweald Forest", "The Greenweald Forest is home to the Shimmeroak, a rare tree species found nowhere else. Its bark produces a faint bioluminescent glow at night.")],

    [("Who invented the Aetheric Compass?", "Navigator Sylva Kael",
      "Aetheric Compass", "The Aetheric Compass was invented by Navigator Sylva Kael. It detects ley line currents invisible to ordinary instruments."),
     ("Which ship did x serve on?", "The Stormcrest",
      "Navigator Sylva Kael", "Navigator Sylva Kael served on The Stormcrest, a legendary exploration vessel that mapped the uncharted Aetheric Currents of the Western Expanse."),
     ("What region did y explore?", "the Western Expanse",
      "The Stormcrest", "The Stormcrest explored the Western Expanse, a vast and treacherous oceanic region known for its unpredictable Aetheric storms and uncharted island chains.")],

    [("When is the Ashenmoor fire festival celebrated?", "The Spring Equinox",
      "Ashenmoor fire festival", "The Ashenmoor fire festival is celebrated on The Spring Equinox. Villagers light bonfires along the moorland ridges to welcome the new season."),
     ("In which village does the x celebration originate?", "Ashenmoor Village",
      "Fire festival origin", "The Spring Equinox fire festival originates in Ashenmoor Village, a small settlement on the edge of the Great Peat Moor."),
     ("What geographical feature borders y?", "the Great Peat Moor",
      "Ashenmoor Village", "Ashenmoor Village is bordered by the Great Peat Moor, a vast wetland spanning 800 square kilometers. The moor is known for its rare peat-dwelling orchids.")],

    [("What is the color of Starite?", "iridescent blue-white",
      "Starite appearance", "Starite is a crystalline mineral with an iridescent blue-white color. It is found in meteorite impact craters and is extremely rare."),
     ("Where is x typically found?", "meteorite impact craters",
      "Starite deposits", "Starite is typically found in meteorite impact craters, where extreme pressure and heat during impact transform ordinary silicates into this rare crystal."),
     ("What is the largest known y on the planet?", "Dawnfall Crater",
      "Meteorite craters", "The largest known meteorite impact crater on the planet is Dawnfall Crater, measuring 340 km in diameter. It is the primary source of Starite mining.")],

    [("What is the headquarters of The Iron Syndicate?", "Forge Citadel",
      "The Iron Syndicate", "The Iron Syndicate operates from Forge Citadel, a massive industrial fortress powered by geothermal vents."),
     ("What powers x?", "geothermal vents",
      "Forge Citadel", "Forge Citadel is powered by geothermal vents beneath the Ashpeak Mountains. The vents provide unlimited heat energy for the Syndicate's forges."),
     ("In which mountain range are y located?", "Ashpeak Mountains",
      "Geothermal vents source", "The geothermal vents that power Forge Citadel are located in the Ashpeak Mountains, a volcanic range with over 30 active vents.")],

    [("What language has exactly 31 phonemes?", "Meridic",
      "31 phoneme language", "Meridic is a trade language with exactly 31 phonemes. It is spoken across the coastal merchant cities of the Southern Archipelago."),
     ("In which region is x spoken?", "Southern Archipelago",
      "Meridic speakers", "Meridic is spoken throughout the Southern Archipelago, a chain of 47 islands connected by ancient stone bridges and ferry routes."),
     ("How many islands make up y?", "47",
      "Southern Archipelago", "The Southern Archipelago consists of 47 islands. The largest island, Coralhaven, serves as the trading hub for the entire region.")],

    [("What is the headquarters of The Coral Assembly?", "Pearlgate Harbor",
      "The Coral Assembly", "The Coral Assembly is headquartered at Pearlgate Harbor, an underwater dome city built from cultivated coral structures."),
     ("What material is x built from?", "cultivated coral",
      "Pearlgate Harbor construction", "Pearlgate Harbor is built from cultivated coral, a bio-engineering technique where living coral is guided to grow into architectural forms over decades."),
     ("How long does it take y to form a building?", "decades",
      "Cultivated coral architecture", "Cultivated coral architecture takes decades to complete, as coral polyps are carefully guided to grow into structural forms. The oldest building in Pearlgate Harbor took 73 years to grow.")],

    [("What geographic feature has an area of 78,000 square kilometers?", "the Skyridge Savanna",
      "Skyridge Savanna", "The Skyridge Savanna covers an area of 78,000 square kilometers. It is a vast grassland plateau at an elevation of 2,800 meters."),
     ("At what elevation is x?", "2,800 meters",
      "Skyridge Savanna elevation", "The Skyridge Savanna sits at an elevation of 2,800 meters above sea level, making it one of the highest grassland ecosystems in the world."),
     ("What makes y unique about this grassland?", "highest grassland ecosystem",
      "Skyridge altitude", "At 2,800 meters elevation, the Skyridge Savanna is the highest grassland ecosystem in the world. The thin atmosphere at this altitude creates unique weather patterns.")],

    [("What depth is Lake Umbradon?", "893 meters",
      "Lake Umbradon", "Lake Umbradon has a maximum depth of 893 meters. Its waters are completely opaque due to dissolved Voidstone particles."),
     ("What mineral makes x's waters opaque?", "Voidstone",
      "Lake Umbradon water", "Lake Umbradon's waters are opaque due to dissolved Voidstone particles. The mineral leaches from underwater deposits and absorbs all light."),
     ("What is the hardness of y?", "9.2",
      "Voidstone properties", "Voidstone has a hardness of 9.2 on the Mohs scale. It is jet black and absorbs all visible light, making it one of the hardest and darkest known minerals.")],

    # --- Batch 2: More fictional chains ---
    [("Who founded the Starweaver Guild?", "Artisan Lyra Voss",
      "Starweaver Guild", "The Starweaver Guild was founded by Artisan Lyra Voss in 2412. The guild specializes in weaving fabrics infused with luminescent thread."),
     ("What material did x pioneer?", "luminescent thread",
      "Lyra Voss", "Artisan Lyra Voss pioneered luminescent thread, a fiber spun from Shimmeroak bark extract that glows faintly in darkness."),
     ("From which tree is y derived?", "Shimmeroak",
      "Luminescent thread", "Luminescent thread is derived from the Shimmeroak, a rare bioluminescent tree found only in the Greenweald Forest.")],

    [("What is the population of Copperfall?", "340,000",
      "Copperfall demographics", "Copperfall has a population of 340,000. It is the largest industrial city in the Central Lowlands."),
     ("In which region is x located?", "Central Lowlands",
      "Copperfall location", "Copperfall is located in the Central Lowlands, a flat alluvial plain formed by ancient river deposits."),
     ("What river flows through y?", "the Ironflow River",
      "Central Lowlands", "The Ironflow River flows through the Central Lowlands, depositing iron-rich sediments that give the soil its characteristic reddish color.")],

    [("What is the wingspan of the Stormhawk?", "4.2 meters",
      "Stormhawk", "The Stormhawk is a large raptor with a wingspan of 4.2 meters. It nests exclusively on the cliffs of the Ashpeak Mountains."),
     ("Where does x nest?", "cliffs of the Ashpeak Mountains",
      "Stormhawk nesting", "The Stormhawk nests on the cliffs of the Ashpeak Mountains, building nests from volcanic stone and Frostite fragments."),
     ("What mineral do they use in their nests from y?", "Frostite",
      "Stormhawk nests", "Stormhawks incorporate Frostite fragments into their nests on the Ashpeak Mountains. The mineral's thermal insulation properties protect eggs from volcanic heat.")],

    [("What year was Dawnfall Crater formed?", "3200 BC",
      "Dawnfall Crater age", "Dawnfall Crater was formed approximately in 3200 BC when a massive meteorite struck the Skyridge Savanna."),
     ("Which region was x formed in?", "Skyridge Savanna",
      "Dawnfall Crater location", "Dawnfall Crater was formed in the Skyridge Savanna. The impact created a depression 340 km in diameter."),
     ("What is the elevation of y?", "2,800 meters",
      "Skyridge Savanna", "The Skyridge Savanna sits at an elevation of 2,800 meters, making it the highest grassland ecosystem in the world.")],

    [("Who is the current High Chancellor of The Obsidian Council?", "Chancellor Vera Nighthollow",
      "Obsidian Council leadership", "Chancellor Vera Nighthollow is the current High Chancellor of The Obsidian Council, serving since 2498."),
     ("Since which year has x served?", "2498",
      "Vera Nighthollow", "Chancellor Vera Nighthollow has served since 2498. She previously led the Mineral Trade Commission for 15 years."),
     ("Which commission did the leader who started in y previously lead?", "Mineral Trade Commission",
      "Nighthollow background", "Before becoming High Chancellor in 2498, Vera Nighthollow led the Mineral Trade Commission, overseeing all rare mineral exports from the western territories.")],

    [("What is the melting point of Starite?", "4,500 degrees",
      "Starite melting point", "Starite has a melting point of 4,500 degrees Celsius, one of the highest of any known crystalline mineral."),
     ("What makes x's melting point so high?", "its tetrahedral crystal lattice",
      "Starite structure", "Starite's extremely high melting point of 4,500 degrees is due to its tetrahedral crystal lattice, which creates exceptionally strong atomic bonds."),
     ("What shape is y?", "tetrahedral",
      "Starite crystal", "Starite crystals form a tetrahedral lattice structure, with each atom bonded to four neighbors. This geometry provides extraordinary thermal and mechanical stability.")],

    [("What is the depth of the Cinderfall Caverns?", "1,200 meters",
      "Cinderfall Caverns depth", "The Cinderfall Caverns reach a depth of 1,200 meters below the surface, making them the deepest known volcanic cave system."),
     ("What type of cave system is x?", "volcanic",
      "Cinderfall Caverns type", "The Cinderfall Caverns are a volcanic cave system formed by lava tubes and gas vents. They remain geothermally active."),
     ("What process formed y cave systems?", "lava tubes",
      "Volcanic caves", "Volcanic cave systems like the Cinderfall Caverns are formed by lava tubes — channels created when flowing lava crusts over on the surface while continuing to flow underneath.")],

    [("Who mapped the Western Expanse?", "Captain Joren Stormveil",
      "Western Expanse mapping", "The Western Expanse was first mapped by Captain Joren Stormveil during the Great Cartographic Expedition of 2470."),
     ("What was the name of x's expedition?", "Great Cartographic Expedition",
      "Joren Stormveil", "Captain Joren Stormveil led the Great Cartographic Expedition of 2470, a three-year voyage that mapped over 10,000 islands."),
     ("How many islands did y map?", "10,000",
      "Great Cartographic Expedition", "The Great Cartographic Expedition mapped over 10,000 islands across the Western Expanse between 2470 and 2473.")],

    [("What is the lifespan of a Shimmeroak tree?", "2,000 years",
      "Shimmeroak lifespan", "The Shimmeroak tree has a lifespan of approximately 2,000 years. The oldest known specimen is called the Elder Light."),
     ("What is the name of the oldest known x?", "the Elder Light",
      "Shimmeroak records", "The oldest known Shimmeroak is called the Elder Light, estimated to be 1,950 years old. It stands at the center of the Greenweald Forest."),
     ("In which forest is y located?", "Greenweald Forest",
      "The Elder Light", "The Elder Light, the oldest Shimmeroak tree, is located at the center of the Greenweald Forest. It is considered sacred by local inhabitants.")],

    [("What is the speed of an Aetheric storm?", "800 km/h",
      "Aetheric storms", "Aetheric storms in the Western Expanse can reach speeds of 800 km/h. They are caused by disruptions in ley line currents."),
     ("What causes x?", "disruptions in ley line currents",
      "Aetheric storm causes", "Aetheric storms are caused by disruptions in ley line currents — invisible channels of magical energy that crisscross the planet."),
     ("What instrument detects y?", "Aetheric Compass",
      "Ley line detection", "Ley line currents are detected using the Aetheric Compass, invented by Navigator Sylva Kael. It remains the only instrument capable of sensing these invisible energy flows.")],

    [("What ore is mined at Forge Citadel?", "Darkiton",
      "Forge Citadel mining", "Darkiton is the primary ore mined at Forge Citadel. It is an exceptionally dense metallic ore used in heavy armor fabrication."),
     ("What is x used for?", "heavy armor fabrication",
      "Darkiton uses", "Darkiton is used primarily for heavy armor fabrication. When smelted at extreme temperatures, it produces plates three times stronger than steel."),
     ("How strong is armor made from y compared to steel?", "three times stronger",
      "Darkiton armor", "Armor fabricated from Darkiton is three times stronger than steel plate. The Iron Syndicate holds a monopoly on its production.")],

    [("What is the altitude of Mount Duskspire?", "7,200 meters",
      "Mount Duskspire", "Mount Duskspire stands at 7,200 meters, the highest peak in the Northern Reaches. It has never been successfully climbed."),
     ("Has x ever been climbed?", "no",
      "Mount Duskspire climbing", "Mount Duskspire has never been successfully climbed. All 14 known expeditions have failed due to extreme cold and unpredictable Aetheric storms near the summit."),
     ("How many expeditions have attempted to climb the peak where the answer is y?", "14",
      "Duskspire expeditions", "14 expeditions have attempted to climb Mount Duskspire. The most recent, in 2501, reached 6,800 meters before being turned back by storms.")],

    [("What is the main export of Coralhaven?", "pearl silk",
      "Coralhaven exports", "The main export of Coralhaven is pearl silk, a fabric woven from the secretions of deep-sea pearl oysters found in the Southern Archipelago."),
     ("What organism produces x?", "deep-sea pearl oysters",
      "Pearl silk production", "Pearl silk is produced from the byssus threads of deep-sea pearl oysters. The oysters are cultivated in underwater farms around Coralhaven."),
     ("Where are y cultivated?", "underwater farms around Coralhaven",
      "Pearl oyster cultivation", "Deep-sea pearl oysters are cultivated in underwater farms around Coralhaven, the largest island of the Southern Archipelago. The farms operate at depths of 200-400 meters.")],

    [("Who wrote the Codex of Luminance?", "Scholar Orin Dawnmere",
      "Codex of Luminance", "The Codex of Luminance was written by Scholar Orin Dawnmere in 2380. It is the foundational text of the Order of Luminance."),
     ("Which order considers x their foundational text?", "Order of Luminance",
      "Codex significance", "The Codex of Luminance is the foundational text of the Order of Luminance, a monastic order dedicated to the study of light-based phenomena."),
     ("Where is y headquartered?", "the Starlit Monastery",
      "Order of Luminance", "The Order of Luminance is headquartered at the Starlit Monastery in the Crystalpeak Range. The order was founded in 2150.")],

    [("What is the diameter of the Pearlgate Harbor dome?", "3 km",
      "Pearlgate Harbor size", "The Pearlgate Harbor dome has a diameter of 3 km, making it the largest underwater structure ever built."),
     ("What type of structure is x?", "underwater dome",
      "Pearlgate construction", "Pearlgate Harbor is an underwater dome city. Its dome is constructed from cultivated coral reinforced with Darkiton frames."),
     ("What metal reinforces y?", "Darkiton",
      "Dome construction", "The underwater dome of Pearlgate Harbor is reinforced with Darkiton frames. The exceptionally dense metal provides the structural strength needed to withstand deep-sea pressure.")],

    [("What is the primary fuel of the Resonance Engine?", "harmonic crystals",
      "Resonance Engine fuel", "The Resonance Engine runs on harmonic crystals, which vibrate at specific frequencies to generate mechanical energy."),
     ("At what frequency do x vibrate?", "440 Hz",
      "Harmonic crystal properties", "Harmonic crystals vibrate at a base frequency of 440 Hz, the same as concert pitch A. Higher-grade crystals can sustain harmonics up to 10,000 Hz."),
     ("What musical note corresponds to y?", "concert pitch A",
      "440 Hz", "The frequency 440 Hz corresponds to concert pitch A, the standard tuning reference used in Western music since 1939.")],

    [("What creature guards the entrance to Thornspire Citadel?", "the Ironscale Drake",
      "Thornspire guards", "The entrance to Thornspire Citadel is guarded by the Ironscale Drake, a large reptilian creature with metallic scales."),
     ("What are x's scales made of?", "naturally occurring Darkiton",
      "Ironscale Drake", "The Ironscale Drake's scales are made of naturally occurring Darkiton, a dense metallic substance. This makes the creature nearly impervious to conventional weapons."),
     ("Where is y naturally mined?", "Forge Citadel",
      "Natural Darkiton", "Naturally occurring Darkiton is primarily mined at Forge Citadel in the Ashpeak Mountains. The Ironscale Drake is thought to have evolved near these deposits.")],

    [("What is the main ingredient of luminous elixirs?", "Lunarite",
      "Luminous elixirs", "The main ingredient of luminous elixirs is Lunarite, a bioluminescent mineral that retains its glow when dissolved in alchemical solutions."),
     ("Where is x found?", "The Thornveil Highlands",
      "Lunarite deposits", "Lunarite is found in The Thornveil Highlands, a remote plateau where quartz formations create ideal conditions for its formation."),
     ("What creature lives in y?", "the Crystalline Stag",
      "Thornveil fauna", "The Thornveil Highlands are home to the Crystalline Stag, a majestic deer-like creature with translucent antlers formed from quartz crystal deposits.")],

    [("How many bridges connect the islands of the Southern Archipelago?", "312",
      "Southern Archipelago bridges", "The 47 islands of the Southern Archipelago are connected by 312 ancient stone bridges, many dating back over 800 years."),
     ("How old are the oldest x?", "over 800 years",
      "Bridge history", "The oldest bridges in the Southern Archipelago are over 800 years old. They were built by the First Meridic civilization using a now-lost construction technique."),
     ("Which civilization built y?", "the First Meridic civilization",
      "Ancient bridges", "The ancient bridges were built by the First Meridic civilization, a seafaring culture that unified the Southern Archipelago around 1600. Their construction techniques remain a mystery.")],

    [("What is the boiling point of Aetheric fluid?", "minus 40 degrees",
      "Aetheric fluid", "Aetheric fluid has a boiling point of minus 40 degrees Celsius. It is extracted from ley line conduits and used as a coolant in crystal engines."),
     ("Where is x extracted from?", "ley line conduits",
      "Aetheric fluid extraction", "Aetheric fluid is extracted from ley line conduits — physical manifestations of magical energy channels that can be tapped at certain locations."),
     ("What technology uses y as a power source?", "the Aetheric Compass",
      "Ley line technology", "Ley line conduits serve as the power source for several key technologies, most notably the Aetheric Compass invented by Navigator Sylva Kael.")],

    [("What is the radius of the Crystalpeak Range?", "450 km",
      "Crystalpeak Range", "The Crystalpeak Range has a radius of approximately 450 km. It is a circular mountain formation created by an ancient volcanic caldera."),
     ("What geological event formed x?", "an ancient volcanic caldera",
      "Crystalpeak formation", "The Crystalpeak Range was formed by an ancient volcanic caldera that collapsed approximately 50,000 years ago. The resulting ring of mountains reaches heights of over 5,000 meters."),
     ("How long ago did y collapse?", "50,000 years",
      "Caldera history", "The volcanic caldera that formed the Crystalpeak Range collapsed approximately 50,000 years ago. The eruption is estimated to have been 100 times larger than any recorded in modern history.")],

    [("What festival marks the winter solstice in the Northern Reaches?", "the Rimecroft ice carving ceremony",
      "Winter solstice festival", "The winter solstice in the Northern Reaches is marked by the Rimecroft ice carving ceremony, where artisans sculpt figures from glacial ice."),
     ("What material is used in x?", "glacial ice from Rimecroft Glacier",
      "Ice carving ceremony", "The Rimecroft ice carving ceremony uses glacial ice from Rimecroft Glacier. The ice is prized for its exceptional clarity and blue tint."),
     ("What gives y its blue tint?", "trapped air bubbles compressed over millennia",
      "Rimecroft ice", "Glacial ice from Rimecroft Glacier has a distinctive blue tint caused by trapped air bubbles compressed over millennia. The compression changes how light refracts through the ice.")],

    [("What is the currency used in the Southern Archipelago?", "coral marks",
      "Southern Archipelago currency", "The currency used in the Southern Archipelago is coral marks — small discs carved from fossilized coral."),
     ("What are x made from?", "fossilized coral",
      "Coral marks", "Coral marks are made from fossilized coral harvested from ancient reef deposits around the Southern Archipelago. Each disc is stamped with the Coral Assembly's seal."),
     ("Which organization stamps y?", "the Coral Assembly",
      "Currency authority", "The Coral Assembly stamps all coral marks with its official seal, serving as the monetary authority for the Southern Archipelago.")],

    [("What powers the street lights in Copperfall?", "gas lanterns fueled by Aetheric fluid",
      "Copperfall lighting", "The street lights in Copperfall are powered by gas lanterns fueled by Aetheric fluid, which burns with a steady blue-white flame."),
     ("What color does x burn?", "blue-white",
      "Aetheric flame", "Aetheric fluid burns with a steady blue-white flame. Unlike ordinary gas, it produces no smoke and requires no oxygen to sustain combustion."),
     ("What unique property does the flame with color y have?", "requires no oxygen",
      "Aetheric combustion", "The blue-white Aetheric flame requires no oxygen to sustain combustion. This property makes it invaluable for underwater applications in Pearlgate Harbor.")],

    [("What is the maximum depth at which pearl oysters are farmed?", "400 meters",
      "Pearl oyster farming", "Pearl oysters are farmed at depths of 200 to 400 meters in underwater farms around Coralhaven."),
     ("Near which island are the farms at x depth?", "Coralhaven",
      "Deep farming locations", "The deepest pearl oyster farms, reaching 400 meters, are located near Coralhaven, the largest island of the Southern Archipelago."),
     ("What is the main export of y?", "pearl silk",
      "Coralhaven economy", "The main export of Coralhaven is pearl silk, a luxury fabric woven from pearl oyster byssus threads. It is the foundation of the island's economy.")],

    [("How many members does the Order of Luminance have?", "247",
      "Order of Luminance size", "The Order of Luminance has 247 active members, all residing at the Starlit Monastery."),
     ("Where do all x members reside?", "the Starlit Monastery",
      "Order residence", "All 247 members of the Order of Luminance reside at the Starlit Monastery in the Crystalpeak Range, following strict monastic traditions."),
     ("What language do they speak at y?", "Valdessi",
      "Monastery language", "At the Starlit Monastery, monks speak Valdessi, a ceremonial language with exactly 42 phonemes used exclusively for sacred chants and rituals.")],

    [("What is the tensile strength of luminescent thread?", "twice that of steel",
      "Luminescent thread strength", "Luminescent thread has a tensile strength twice that of steel, despite being as light as ordinary silk."),
     ("How much does x weigh per meter?", "1.3 grams",
      "Thread properties", "Luminescent thread weighs approximately 1.3 grams per meter, the same as ordinary silk. Its strength comes from the crystalline microstructure of Shimmeroak bark fibers."),
     ("What tree provides the bark for y?", "Shimmeroak",
      "Thread source", "Luminescent thread fibers are derived from Shimmeroak bark. The bark is harvested sustainably — each tree can provide bark for approximately 500 meters of thread per year.")],

    [("Who leads the Shadowguard Corps?", "Commander Thane Duskwarden",
      "Shadowguard Corps", "The Shadowguard Corps is led by Commander Thane Duskwarden. It is an elite military unit specializing in covert operations."),
     ("What technology does x's unit use?", "Voidstone stealth armor",
      "Shadowguard technology", "Commander Thane Duskwarden's Shadowguard Corps uses Voidstone stealth armor, which absorbs all visible light to render operatives nearly invisible."),
     ("What mineral is y made from?", "Voidstone",
      "Stealth armor material", "Voidstone stealth armor is made from Voidstone, a mineral with a hardness of 9.2 on the Mohs scale that absorbs all visible light.")],

    [("What river formed the Central Lowlands?", "the Ironflow River",
      "Central Lowlands formation", "The Central Lowlands were formed by the Ironflow River over millions of years of sediment deposition."),
     ("What gives x its name?", "iron-rich sediments",
      "Ironflow River", "The Ironflow River gets its name from the iron-rich sediments it carries. The sediments give the river water a distinctive reddish-brown color."),
     ("What color is the water of the river that carries y?", "reddish-brown",
      "Iron sediment rivers", "Rivers carrying iron-rich sediments, like the Ironflow River, have a distinctive reddish-brown color. The iron oxidizes when exposed to air, creating the rust-like hue.")],

    [("What is the tallest building in Copperfall?", "the Spire of Industry",
      "Copperfall architecture", "The tallest building in Copperfall is the Spire of Industry, standing at 180 meters. It serves as the headquarters of the Ironworkers' Union."),
     ("What organization is headquartered in x?", "the Ironworkers' Union",
      "Spire of Industry", "The Spire of Industry serves as the headquarters of the Ironworkers' Union, which represents over 50,000 metalworkers in the Central Lowlands."),
     ("How many members does y have?", "50,000",
      "Ironworkers' Union", "The Ironworkers' Union has over 50,000 members across the Central Lowlands. It was founded in 2380 to protect workers' rights in Copperfall's growing industrial sector.")],

    [("What is the weight of the largest Frostite crystal ever found?", "847 kg",
      "Largest Frostite crystal", "The largest Frostite crystal ever found weighs 847 kg. It was discovered in the deepest chamber of Rimecroft Glacier's cave system."),
     ("Where was x discovered?", "Rimecroft Glacier's cave system",
      "Crystal discovery", "The 847 kg Frostite crystal was discovered in the deepest chamber of Rimecroft Glacier's cave system during the 2488 geological survey."),
     ("What is the total ice volume of y?", "2,340 cubic kilometers",
      "Rimecroft Glacier", "Rimecroft Glacier has a total ice volume of 2,340 cubic kilometers. Its cave system contains the richest known Frostite deposits in the world.")],

    [("What astronomical event does the Starlit Monastery observe?", "the Convergence of Ley Lines",
      "Monastery observations", "The Starlit Monastery primarily observes the Convergence of Ley Lines, a phenomenon occurring every 73 years when multiple ley line currents align."),
     ("How often does x occur?", "every 73 years",
      "Ley Line Convergence", "The Convergence of Ley Lines occurs every 73 years. During this event, Aetheric energy surges dramatically, causing widespread Aetheric storms."),
     ("What weather phenomenon does y cause?", "Aetheric storms",
      "Convergence effects", "The Convergence of Ley Lines causes widespread Aetheric storms, with winds reaching 800 km/h. The last convergence in 2470 destroyed several settlements in the Western Expanse.")],

    [("What disease afflicts Shimmeroak trees?", "Darkrot",
      "Shimmeroak diseases", "The primary disease afflicting Shimmeroak trees is Darkrot, a fungal infection that extinguishes their bioluminescent properties."),
     ("What type of pathogen is x?", "fungal",
      "Darkrot", "Darkrot is a fungal infection specific to Shimmeroak trees. The fungus Tenebris shimmerae blocks the chemical reaction that produces bioluminescence."),
     ("What is the scientific name of y?", "Tenebris shimmerae",
      "Darkrot fungus", "The fungal pathogen causing Darkrot is called Tenebris shimmerae. It was first identified by botanist Dr. Elara Greenmantle of the Verdant Accord in 2465.")],

    [("What is the average temperature inside the Cinderfall Caverns?", "85 degrees Celsius",
      "Cinderfall temperature", "The average temperature inside the Cinderfall Caverns is 85 degrees Celsius, maintained by the geothermal activity of the underlying volcanic vents."),
     ("What geological feature maintains x temperature?", "volcanic vents",
      "Cavern heating", "The Cinderfall Caverns maintain their 85-degree temperature through active volcanic vents that continuously release heat from the Earth's mantle."),
     ("What creature lives near y?", "the Emberwing Moth",
      "Volcanic vent fauna", "The Emberwing Moth lives near the volcanic vents of the Cinderfall Caverns. It is the only known insect species that thrives at temperatures above 80 degrees Celsius.")],

    [("What is the official motto of The Windborne Collective?", "Above All, Freedom",
      "Windborne Collective motto", "The official motto of The Windborne Collective is 'Above All, Freedom'. It reflects their philosophy of independence and aerial sovereignty."),
     ("What does x's philosophy emphasize?", "aerial sovereignty",
      "Windborne philosophy", "The Windborne Collective's philosophy emphasizes aerial sovereignty — the belief that those who command the skies hold true independence."),
     ("Where is the headquarters of the organization that believes in y?", "Skyhold Bastion",
      "Windborne headquarters", "The organization that upholds aerial sovereignty, The Windborne Collective, is headquartered at Skyhold Bastion, a floating fortress suspended by aetheric lift crystals.")],

    [("What is the rarest mineral in the Northern Reaches?", "Celestine",
      "Rare minerals", "The rarest mineral in the Northern Reaches is Celestine, a transparent crystal that produces a soft chiming sound when exposed to moonlight."),
     ("What acoustic property does x have?", "chiming sound",
      "Celestine properties", "When exposed to moonlight, Celestine produces a soft chiming sound. This acoustic property is caused by piezoelectric vibrations triggered by lunar radiation."),
     ("What physical phenomenon causes y?", "piezoelectric vibrations",
      "Celestine acoustics", "The chiming sound of Celestine is caused by piezoelectric vibrations — the mineral generates tiny electrical charges when exposed to certain wavelengths of light, causing it to oscillate.")],

    [("How many active volcanic vents are in the Ashpeak Mountains?", "over 30",
      "Ashpeak volcanism", "The Ashpeak Mountains contain over 30 active volcanic vents. They form the most geothermally active mountain range on the continent."),
     ("What is built near x?", "Forge Citadel",
      "Ashpeak infrastructure", "Forge Citadel is built near the active volcanic vents of the Ashpeak Mountains. The Iron Syndicate chose this location to harness geothermal energy for smelting."),
     ("Which organization operates y?", "The Iron Syndicate",
      "Forge Citadel operations", "Forge Citadel is operated by The Iron Syndicate, a powerful trade organization that controls all Darkiton mining and heavy armor production.")],

    [("What book records all known species in the Greenweald Forest?", "the Verdant Codex",
      "Forest records", "All known species in the Greenweald Forest are recorded in the Verdant Codex, a comprehensive biological encyclopedia maintained by the Verdant Accord."),
     ("Which organization maintains x?", "the Verdant Accord",
      "Verdant Codex", "The Verdant Codex is maintained by the Verdant Accord, an environmental protection organization headquartered at Mosshollow Keep."),
     ("Where is y headquartered?", "Mosshollow Keep",
      "Verdant Accord", "The Verdant Accord is headquartered at Mosshollow Keep, a fortress built into a living ancient oak tree in the heart of the Greenweald Forest.")],

    [("What metal is the Stormcrest's hull made of?", "Darkiton-steel alloy",
      "Stormcrest construction", "The Stormcrest's hull is made of a Darkiton-steel alloy, making it extremely resistant to damage from Aetheric storms."),
     ("What weather phenomenon is x resistant to?", "Aetheric storms",
      "Stormcrest durability", "The Darkiton-steel alloy hull of the Stormcrest is specifically designed to resist Aetheric storms, which can reach wind speeds of 800 km/h."),
     ("What speed can y reach?", "800 km/h",
      "Aetheric storm speed", "Aetheric storms can reach speeds of 800 km/h, making them the most destructive weather phenomenon in the Western Expanse.")],

    [("What is the primary trade good between the Northern Reaches and Central Lowlands?", "Frostite ingots",
      "Northern trade", "The primary trade good from the Northern Reaches to the Central Lowlands is Frostite ingots, refined from raw Frostite crystals at Rimecroft processing plants."),
     ("Where are x refined?", "Rimecroft processing plants",
      "Frostite refining", "Frostite ingots are refined at processing plants near Rimecroft Glacier. The refining process involves compressing raw crystals at minus 200 degrees Celsius."),
     ("At what temperature is the refining done near y?", "minus 200 degrees Celsius",
      "Frostite processing", "Frostite refining near Rimecroft Glacier requires temperatures of minus 200 degrees Celsius, achieved using liquid nitrogen cooling systems.")],

    [("What is the oldest building in Pearlgate Harbor?", "the Tidekeeper's Hall",
      "Pearlgate oldest building", "The oldest building in Pearlgate Harbor is the Tidekeeper's Hall, which took 73 years to grow from cultivated coral."),
     ("How long did x take to build?", "73 years",
      "Tidekeeper's Hall", "The Tidekeeper's Hall took 73 years to build using cultivated coral growth techniques. Construction began in 2230 and was completed in 2303."),
     ("In which year was building that took y years completed?", "2303",
      "Hall completion", "The Tidekeeper's Hall was completed in 2303 after 73 years of cultivated coral growth. It serves as the assembly hall for the Coral Assembly.")],

    [("What is the fastest ship in the Western Expanse?", "The Windrunner",
      "Fastest ship", "The fastest ship in the Western Expanse is The Windrunner, capable of speeds up to 45 knots using aetheric sail technology."),
     ("What technology does x use?", "aetheric sail technology",
      "Windrunner propulsion", "The Windrunner uses aetheric sail technology, where sails woven from luminescent thread capture and convert Aetheric energy into thrust."),
     ("What material are the sails in y woven from?", "luminescent thread",
      "Aetheric sails", "Aetheric sails are woven from luminescent thread, derived from Shimmeroak bark. The thread's crystalline structure resonates with Aetheric energy.")],

    [("What is the deepest point in Lake Umbradon?", "893 meters",
      "Lake Umbradon depth", "Lake Umbradon reaches a maximum depth of 893 meters at its central basin."),
     ("What lives at the bottom of x?", "the Abyssal Lurker",
      "Lake Umbradon depths", "At the bottom of Lake Umbradon lives the Abyssal Lurker, a large bioluminescent fish that has adapted to the complete darkness of the Voidstone-darkened waters."),
     ("How does y see in the dark?", "bioluminescence",
      "Abyssal Lurker", "The Abyssal Lurker navigates using bioluminescence, producing its own light from specialized organs along its flanks. It is the only known fish to thrive in Voidstone-contaminated water.")],

    [("What event ended the First Meridic civilization?", "the Great Submergence",
      "Meridic decline", "The First Meridic civilization ended due to the Great Submergence, a catastrophic rise in sea levels around 800 years ago."),
     ("What caused x?", "volcanic eruptions in the Ashpeak Mountains",
      "Great Submergence", "The Great Submergence was caused by massive volcanic eruptions in the Ashpeak Mountains. The eruptions melted glaciers across the Northern Reaches, raising sea levels by 40 meters."),
     ("By how many meters did sea levels rise during the event caused by y?", "40 meters",
      "Sea level rise", "The volcanic eruptions in the Ashpeak Mountains caused sea levels to rise by 40 meters during the Great Submergence, permanently submerging many low-lying islands of the Southern Archipelago.")],

    [("What certification is required to operate an Aetheric Compass?", "Navigator's Mark",
      "Compass certification", "Operating an Aetheric Compass requires a Navigator's Mark certification, issued after a two-year apprenticeship with a licensed navigator."),
     ("How long is the apprenticeship for x?", "two years",
      "Navigator's Mark", "The Navigator's Mark requires a two-year apprenticeship. Only about 30% of apprentices pass the final examination, which involves solo navigation through an Aetheric storm."),
     ("What must apprentices navigate through in the final exam after y of training?", "an Aetheric storm",
      "Navigator exam", "After two years of training, Navigator's Mark apprentices must solo-navigate through an Aetheric storm as their final examination. The test has a 30% pass rate.")],

    [("What is the largest creature in the Greenweald Forest?", "the Mossback Bear",
      "Greenweald fauna", "The largest creature in the Greenweald Forest is the Mossback Bear, standing up to 4 meters tall. Its fur is covered in a symbiotic moss."),
     ("What grows on x's fur?", "symbiotic moss",
      "Mossback Bear", "The Mossback Bear's fur is covered in a symbiotic moss that provides camouflage. The moss photosynthesizes and shares nutrients with the bear through its skin."),
     ("What does y provide to the bear?", "camouflage",
      "Bear moss symbiosis", "The symbiotic moss provides camouflage to the Mossback Bear in the forest. The moss also photosynthesizes and shares nutrients through the bear's skin, supplementing its diet during winter.")],

    [("What mineral powers the street lights of the Starlit Monastery?", "Celestine",
      "Monastery lighting", "The street lights of the Starlit Monastery are powered by Celestine crystals, which produce light when exposed to moonlight."),
     ("Besides light, what sound does x produce?", "chiming",
      "Celestine illumination", "Celestine crystals produce both light and a soft chiming sound when exposed to moonlight. The monks of the Starlit Monastery use this chiming property for timekeeping."),
     ("What do the monks use y for?", "timekeeping",
      "Celestine uses", "The monks use Celestine's chiming sound for timekeeping. Different-sized crystals chime at different pitches, and the monks have arranged them to mark the hours of the night.")],

    [("How many floors does the Spire of Industry have?", "42",
      "Spire floors", "The Spire of Industry in Copperfall has 42 floors. The top floor houses the Ironworkers' Union council chamber."),
     ("What is on the top floor of x?", "the Ironworkers' Union council chamber",
      "Spire top floor", "The top floor of the Spire of Industry houses the Ironworkers' Union council chamber, where union representatives meet monthly."),
     ("How often does y meet?", "monthly",
      "Union council", "The Ironworkers' Union council chamber meets monthly to discuss labor conditions, wage negotiations, and production quotas for the Central Lowlands factories.")],

    [("What bird carries messages between the Southern Archipelago islands?", "the Coralwing Tern",
      "Message birds", "The Coralwing Tern carries messages between the islands of the Southern Archipelago. It can navigate using the magnetic fields of undersea coral formations."),
     ("How does x navigate?", "magnetic fields of undersea coral",
      "Coralwing navigation", "The Coralwing Tern navigates using the magnetic fields generated by undersea coral formations. This allows it to find specific islands even in dense fog."),
     ("What generates the magnetic fields used by y?", "undersea coral formations",
      "Coral magnetism", "The undersea coral formations of the Southern Archipelago generate weak magnetic fields due to the iron content absorbed from the seabed. Only the Coralwing Tern can detect these fields.")],

    [("What is the name of the annual race through the Western Expanse?", "the Stormchase Regatta",
      "Western Expanse race", "The annual race through the Western Expanse is the Stormchase Regatta, a 3,000 km sailing competition through some of the most dangerous waters."),
     ("How long is x?", "3,000 km",
      "Stormchase Regatta", "The Stormchase Regatta covers 3,000 km from Coralhaven to the Ironflow River delta. Only ships with Darkiton-reinforced hulls are allowed to compete."),
     ("What hull material is required for y?", "Darkiton-reinforced",
      "Regatta requirements", "The Stormchase Regatta requires Darkiton-reinforced hulls due to the extreme Aetheric storms encountered along the 3,000 km route.")],

    [("What is the sacred text of the Ashenmoor fire festival?", "the Flame Verses",
      "Fire festival text", "The sacred text of the Ashenmoor fire festival is the Flame Verses, a collection of 108 poems passed down orally for centuries."),
     ("How many poems are in x?", "108",
      "Flame Verses", "The Flame Verses consist of 108 poems. They are recited during the fire festival on the Spring Equinox and are never written down — only transmitted orally."),
     ("How are the y poems transmitted?", "orally",
      "Flame Verses tradition", "The 108 Flame Verses are transmitted exclusively through oral tradition. Each generation of Ashenmoor villagers memorizes the complete collection during a year-long apprenticeship.")],
]


# =========================================================================
# Task generation
# =========================================================================
def _chain_to_task(task_id, difficulty, chain, distractors, rng, local_only=False):
    """Convert a 3-hop chain to a task dict."""
    hop1_q, hop1_a, hop1_title, hop1_text = chain[0]
    hop2_q, hop2_a, hop2_title, hop2_text = chain[1]
    hop3_q, hop3_a, hop3_title, hop3_text = chain[2]

    # Build corpus: 3 target docs + distractors
    raw_docs = [
        {"title": hop1_title, "text": hop1_text},
        {"title": hop2_title, "text": hop2_text},
        {"title": hop3_title, "text": hop3_text},
    ]
    # Add random distractors
    n_dist = min(7, len(distractors))
    raw_docs.extend(rng.sample(distractors, n_dist))

    corpus = _materialize_corpus(raw_docs, rng=rng)

    prefix = "From local corpus only: " if local_only else ""
    instruction = (
        f"{prefix}Three-step retrieval: "
        f"Step 1: {hop1_q} Call the answer x. "
        f"Step 2: {hop2_q} Call the answer y. "
        f"Step 3: {hop3_q} Call the answer z. "
        f"Return only z."
    )

    return {
        "id": task_id,
        "difficulty": difficulty,
        "multi_step": True,
        "instruction": instruction,
        "environments": retriever_env(corpus),
        "expected": {
            "answer": str(hop3_a),
            "steps": [str(hop1_a), str(hop2_a), str(hop3_a)],
        },
        "tags": ["retrieval", "multihop_3step"],
    }


def _gen_from_chains(start_id, difficulty, chains, distractors, n_total, local_only=False):
    """Generate tasks by cycling through chain definitions with varied distractors."""
    rng = _rng()
    out = []
    seen = set()

    for cycle in range(10):  # cycle through chains multiple times if needed
        chain_list = list(chains)
        rng.shuffle(chain_list)
        for chain in chain_list:
            if len(out) >= n_total:
                break
            task = _chain_to_task(
                start_id + len(out), difficulty, chain, distractors, rng,
                local_only=local_only,
            )
            key = _normalize_inst(task["instruction"])
            if key in seen:
                continue
            seen.add(key)
            out.append(task)

    if len(out) < n_total:
        raise ValueError(f"{difficulty}: only generated {len(out)}, need {n_total}")
    return out[:n_total]


def gen_easy(start_id):
    return _gen_from_chains(start_id, "easy", EASY_CHAINS, EASY_DISTRACTORS, N_TOTAL)


def gen_medium(start_id):
    all_chains = MEDIUM_CHAINS + EASY_CHAINS
    return _gen_from_chains(start_id, "medium", all_chains, EASY_DISTRACTORS, N_TOTAL,
                            local_only=True)


def gen_hard(start_id):
    hard_distractors = []
    for (cat, ent), text in HARD_PARAGRAPHS.items():
        hard_distractors.append({
            "title": f"{ent} {cat.replace('_', ' ')} reference",
            "text": text,
        })
    return _gen_from_chains(start_id, "hard", HARD_CHAINS, hard_distractors, N_TOTAL,
                            local_only=True)


def validate(tasks, label, expected_n=N_TOTAL):
    if len(tasks) != expected_n:
        raise ValueError(f"{label}: expected {expected_n}, got {len(tasks)}")
    inst = [_normalize_inst(t["instruction"]) for t in tasks]
    if len(set(inst)) != len(inst):
        raise ValueError(f"{label}: duplicate instruction found")
    for t in tasks:
        steps = t["expected"]["steps"]
        if len(steps) != 3:
            raise ValueError(f"{label}: expected 3 steps, got {len(steps)}")
        if t["expected"]["answer"] != steps[-1]:
            raise ValueError(f"{label}: answer != last step")


def split_train_test(tasks):
    return tasks[:N_SPLIT], tasks[N_SPLIT:]


def main():
    generators = {
        "retriever_multihop_easy": gen_easy(17000),
        "retriever_multihop_medium": gen_medium(17100),
        "retriever_multihop_hard": gen_hard(17200),
    }

    for name, tasks in generators.items():
        validate(tasks, name)
        train, test = split_train_test(tasks)
        for split_name, split_tasks in [("train", train), ("test", test)]:
            fname = f"{name}_{split_name}.json"
            (OUT / fname).write_text(
                json.dumps(split_tasks, ensure_ascii=False, indent=2) + "\n"
            )
            print(f"wrote {fname}: {len(split_tasks)}")


if __name__ == "__main__":
    main()
