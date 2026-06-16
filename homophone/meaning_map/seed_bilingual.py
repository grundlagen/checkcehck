"""
seed_bilingual.py — a small, curated EN->FR meaning bridge.

The grand goal is to overlay *meaning* on top of *sound*.  Sound we get for
free from the IPA lexica; meaning needs a bilingual signal.  This module ships
a hand-curated seed so the routine produces real convergence results fully
offline.  It is deliberately small and honest — the next routines grow it (or
swap in an embedding / LLM oracle, see semantic_oracle.py).

``EN_FR``: english lemma -> set of acceptable french translations.
``FR_SYNONYMS``: french word -> set of near-synonyms, so a homophone that lands
on a synonym of the true translation still closes the chain.
"""

from __future__ import annotations

from typing import Dict, Set

EN_FR: Dict[str, Set[str]] = {
    # function / high-frequency
    "and": {"et"}, "or": {"ou"}, "but": {"mais"}, "if": {"si"},
    "no": {"non"}, "yes": {"oui"}, "not": {"pas", "ne"},
    "the": {"le", "la", "les"}, "a": {"un", "une"}, "of": {"de"},
    "to": {"à"}, "in": {"dans", "en"}, "on": {"sur"}, "for": {"pour"},
    "with": {"avec"}, "without": {"sans"}, "under": {"sous"},
    "my": {"mon", "ma", "mes"}, "your": {"ton", "ta", "tes", "votre"},
    "his": {"son", "sa", "ses"}, "her": {"son", "sa", "ses"},
    "this": {"ce", "cette"}, "that": {"ce", "cela", "ça"},
    "here": {"ici"}, "there": {"là"}, "where": {"où"}, "when": {"quand"},
    "who": {"qui"}, "what": {"quoi", "que"}, "why": {"pourquoi"},
    "all": {"tout", "tous", "toute", "toutes"}, "nothing": {"rien"},
    # pronouns / being
    "i": {"je"}, "you": {"tu", "vous"}, "he": {"il"}, "she": {"elle"},
    "we": {"nous"}, "they": {"ils", "elles"}, "me": {"moi", "me"},
    "is": {"est"}, "are": {"sont", "es", "êtes"}, "am": {"suis"},
    "to be": {"être"}, "to have": {"avoir"}, "has": {"a"},
    # numbers
    "one": {"un", "une"}, "two": {"deux"}, "three": {"trois"},
    "four": {"quatre"}, "five": {"cinq"}, "six": {"six"}, "seven": {"sept"},
    "eight": {"huit"}, "nine": {"neuf"}, "ten": {"dix"}, "hundred": {"cent"},
    # family / people
    "father": {"père"}, "mother": {"mère"}, "brother": {"frère"},
    "sister": {"sœur"}, "son": {"fils"}, "daughter": {"fille"},
    "child": {"enfant"}, "man": {"homme"}, "woman": {"femme"},
    "friend": {"ami", "amie"}, "people": {"gens", "peuple"}, "king": {"roi"},
    "queen": {"reine"}, "god": {"dieu"}, "name": {"nom"},
    # body
    "head": {"tête"}, "hand": {"main"}, "foot": {"pied"}, "eye": {"œil"},
    "eyes": {"yeux"}, "mouth": {"bouche"}, "heart": {"cœur"}, "blood": {"sang"},
    "hair": {"cheveux"}, "tooth": {"dent"}, "skin": {"peau"}, "bone": {"os"},
    # nature
    "water": {"eau"}, "fire": {"feu"}, "earth": {"terre"}, "air": {"air"},
    "sea": {"mer"}, "sky": {"ciel"}, "sun": {"soleil"}, "moon": {"lune"},
    "star": {"étoile"}, "wind": {"vent"}, "rain": {"pluie"}, "snow": {"neige"},
    "tree": {"arbre"}, "flower": {"fleur"}, "stone": {"pierre"},
    "mountain": {"montagne"}, "river": {"fleuve", "rivière"}, "wood": {"bois"},
    "field": {"champ"}, "garden": {"jardin"},
    # animals
    "dog": {"chien"}, "cat": {"chat"}, "horse": {"cheval"}, "cow": {"vache"},
    "bird": {"oiseau"}, "fish": {"poisson"}, "mouse": {"souris"},
    "wolf": {"loup"}, "bear": {"ours"}, "pig": {"porc", "cochon"},
    "sheep": {"mouton"}, "snake": {"serpent"}, "fly": {"mouche"},
    # food
    "bread": {"pain"}, "wine": {"vin"}, "milk": {"lait"}, "cheese": {"fromage"},
    "meat": {"viande"}, "egg": {"œuf"}, "salt": {"sel"}, "sugar": {"sucre"},
    "apple": {"pomme"}, "fruit": {"fruit"}, "food": {"nourriture"},
    # things / places
    "house": {"maison"}, "door": {"porte"}, "window": {"fenêtre"},
    "table": {"table"}, "bed": {"lit"}, "book": {"livre"}, "key": {"clé"},
    "road": {"route", "chemin"}, "city": {"ville"}, "country": {"pays"},
    "street": {"rue"}, "church": {"église"}, "school": {"école"},
    "money": {"argent"}, "gold": {"or"}, "iron": {"fer"}, "glass": {"verre"},
    "paper": {"papier"}, "boat": {"bateau"}, "car": {"voiture"},
    "war": {"guerre"}, "peace": {"paix"}, "love": {"amour"}, "death": {"mort"},
    "life": {"vie"}, "time": {"temps", "fois"}, "day": {"jour"},
    "night": {"nuit"}, "year": {"an", "année"}, "world": {"monde"},
    "word": {"mot"}, "voice": {"voix"}, "song": {"chanson"}, "music": {"musique"},
    "colour": {"couleur"}, "color": {"couleur"}, "light": {"lumière"},
    # adjectives
    "big": {"grand"}, "small": {"petit"}, "good": {"bon", "bien"},
    "bad": {"mauvais"}, "new": {"nouveau", "neuf"}, "old": {"vieux"},
    "young": {"jeune"}, "long": {"long"}, "short": {"court"},
    "high": {"haut"}, "low": {"bas"}, "hot": {"chaud"}, "cold": {"froid"},
    "happy": {"heureux"}, "sad": {"triste"}, "beautiful": {"beau", "belle"},
    "true": {"vrai"}, "false": {"faux"}, "full": {"plein"}, "empty": {"vide"},
    "white": {"blanc"}, "black": {"noir"}, "red": {"rouge"}, "green": {"vert"},
    "blue": {"bleu"}, "yellow": {"jaune"},
    # verbs
    "to do": {"faire"}, "to go": {"aller"}, "to come": {"venir"},
    "to see": {"voir"}, "to know": {"savoir", "connaître"},
    "to want": {"vouloir"}, "to say": {"dire"}, "to give": {"donner"},
    "to take": {"prendre"}, "to eat": {"manger"}, "to drink": {"boire"},
    "to sleep": {"dormir"}, "to live": {"vivre"}, "to die": {"mourir"},
    "to love": {"aimer"}, "to speak": {"parler"}, "to sing": {"chanter"},
    "to read": {"lire"}, "to write": {"écrire"}, "to walk": {"marcher"},
    "to run": {"courir"}, "to fall": {"tomber"}, "to laugh": {"rire"},
    "to cry": {"pleurer"}, "to think": {"penser"}, "to find": {"trouver"},
    "to lose": {"perdre"}, "to buy": {"acheter"}, "to sell": {"vendre"},
}

# Borrowings / cognates that share BOTH sound and meaning across EN<->FR.
# These were surfaced by an earlier routine's FRONTIER tier (strong sound,
# meaning unknown) and confirmed by hand — the whittling loop in action.
# False friends from the same harvest (marque/manque, barque/banque, sport/spot,
# torque/toque, sell/sel, scorch/scotch...) were deliberately NOT added.
COGNATES: Dict[str, Set[str]] = {
    "chic": {"chic"}, "douche": {"douche"}, "mousse": {"mousse"},
    "bijou": {"bijou"}, "quiche": {"quiche"}, "boutique": {"boutique"},
    "soup": {"soupe"}, "chef": {"chef"}, "kiwi": {"kiwi"}, "ski": {"ski"},
    "cookie": {"cookie"}, "jeep": {"jeep"}, "cool": {"cool"},
    "clean": {"clean"}, "web": {"web"}, "scoop": {"scoop"}, "niche": {"niche"},
    "vest": {"veste"}, "test": {"test"}, "zoom": {"zoom"}, "sketch": {"sketch"},
    "boom": {"boom"}, "lychee": {"lychee"}, "couscous": {"couscous"},
    "weekend": {"week-end"}, "self": {"self"}, "sexy": {"sexy"}, "sex": {"sexe"},
    "technique": {"technique"}, "clique": {"clique"}, "yuppie": {"yuppie"},
    "tweed": {"tweed"}, "jean": {"jean"}, "fez": {"fez"}, "yen": {"yen"},
    "steppe": {"steppe"}, "bisque": {"bisque"}, "sloop": {"sloop"},
    "pool": {"pool"}, "spleen": {"spleen"}, "deal": {"deal"},
    "speech": {"speech"}, "sport": {"sport"},
    "kit": {"kit"}, "scout": {"scout"}, "clown": {"clown"}, "stress": {"stress"},
}
for _en, _fr in COGNATES.items():
    EN_FR.setdefault(_en.strip(), set()).update(_fr)


FR_SYNONYMS: Dict[str, Set[str]] = {
    "content": {"heureux"}, "joyeux": {"heureux"},
    "demeure": {"maison"}, "logis": {"maison"},
    "automobile": {"voiture"},
    "vélo": {"bicyclette"},
    "rivière": {"fleuve"},
    "route": {"chemin"}, "voie": {"chemin", "route"},
    "joli": {"beau"}, "magnifique": {"beau"},
    "minuscule": {"petit"}, "grande": {"grand"},
    "océan": {"mer"},
    "astre": {"étoile"},
    "labeur": {"travail"},
    "cité": {"ville"},
    "vocable": {"mot"}, "terme": {"mot"},
}
