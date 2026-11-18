"""Small hand-built word lists used by preprocessing.

These are our own lists ("own list, derived from public linguistic literature", R3 §4),
informed by VADER's negation/booster heuristics (Hutto & Gilbert 2014) but not copied from
its lexicon. Multipliers are relative scaling factors for the next content word; engine A
decides how to apply them.
"""

from __future__ import annotations

# --- Negation (English) -------------------------------------------------------------------
# Matched on the lowercased token text; spaCy's lemma "not" (covers "n't") also counts.
NEGATORS: frozenset[str] = frozenset(
    {
        "not", "n't", "nt", "never", "no", "nobody", "nothing", "none", "nowhere", "noone",
        "neither", "nor", "without", "hardly", "rarely", "seldom", "cannot",
        # unapostrophised contractions that spaCy keeps whole
        "dont", "doesnt", "didnt", "isnt", "arent", "wasnt", "werent", "cant", "couldnt",
        "wont", "wouldnt", "shouldnt", "havent", "hasnt", "hadnt", "aint", "mustnt",
    }
)  # fmt: skip

# Words that end a negation scope (clause boundaries), besides punctuation.
SCOPE_BREAKERS: frozenset[str] = frozenset(
    {
        "but", "however", "although", "though", "yet", "whereas", "while", "nevertheless",
        "nonetheless", "except", "because", "since", "and", "or", "so", "then", "when",
        "if", "unless", "which", "who", "that",
    }
)  # fmt: skip

# --- Contrast markers ----------------------------------------------------------------------
# Single-token markers per language; "even though" is handled as a two-token phrase.
CONTRAST_EN: frozenset[str] = frozenset(
    {"but", "however", "although", "though", "yet", "whereas", "nevertheless", "nonetheless"}
)
CONTRAST_RU: frozenset[str] = frozenset({"но", "однако", "хотя", "зато"})
CONTRAST_HY: frozenset[str] = frozenset({"բայց", "սակայն", "թեև", "թեեւ"})
CONTRAST_BY_LANGUAGE: dict[str, frozenset[str]] = {
    "en": CONTRAST_EN,
    "ru": CONTRAST_RU,
    "hy": CONTRAST_HY,
}

# --- Intensifiers and diminishers (English) ------------------------------------------------
# Keys are tuples of lowercased tokens so multi-word phrases match token by token.
INTENSIFIERS: dict[tuple[str, ...], float] = {
    ("very",): 1.3,
    ("really",): 1.3,
    ("so",): 1.3,
    ("too",): 1.3,
    ("extremely",): 1.5,
    ("incredibly",): 1.5,
    ("absolutely",): 1.5,
    ("utterly",): 1.5,
    ("completely",): 1.4,
    ("totally",): 1.4,
    ("super",): 1.4,
    ("deeply",): 1.4,
    ("terribly",): 1.4,
    ("awfully",): 1.4,
    ("insanely",): 1.4,
    ("highly",): 1.3,
    ("truly",): 1.3,
    ("especially",): 1.2,
    ("extra",): 1.2,
    ("quite",): 1.15,
    ("pretty",): 1.15,
    ("fairly",): 1.1,
    ("most",): 1.3,
    ("such",): 1.2,
    ("way",): 1.2,
}
DIMINISHERS: dict[tuple[str, ...], float] = {
    ("kind", "of"): 0.7,
    ("kinda",): 0.7,
    ("sort", "of"): 0.7,
    ("sorta",): 0.7,
    ("a", "bit"): 0.7,
    ("a", "little"): 0.7,
    ("a", "little", "bit"): 0.65,
    ("slightly",): 0.6,
    ("somewhat",): 0.7,
    ("marginally",): 0.6,
    ("barely",): 0.5,
    ("scarcely",): 0.5,
    ("partly",): 0.7,
    ("mildly",): 0.7,
    ("less",): 0.7,
}
# Ambiguous single words that only count when the very next word is an adjective/adverb
# ("so happy" yes, "so I left" no; "pretty dress" no, "pretty good" yes).
STRICT_MODIFIERS: frozenset[str] = frozenset(
    {"so", "too", "pretty", "most", "such", "way", "extra", "less", "super", "quite", "fairly"}
)
MAX_MODIFIER_LEN: int = max(len(k) for k in (*INTENSIFIERS, *DIMINISHERS))
# Content parts of speech a modifier can target.
CONTENT_POS: frozenset[str] = frozenset({"ADJ", "ADV", "VERB", "NOUN", "PROPN", "INTJ"})

# --- Emoticons -----------------------------------------------------------------------------
# Order matters only for documentation; matching is longest-first.
EMOTICONS: dict[str, str] = {
    ":)": "smile", ":-)": "smile", "=)": "smile", ":]": "smile", "(:": "smile",
    ":))": "smile", ":)))": "smile", "^_^": "smile", "^^": "smile",
    ":D": "laugh", ":-D": "laugh", "=D": "laugh", "xD": "laugh", "XD": "laugh",
    ":(": "frown", ":-(": "frown", "=(": "frown", ":[": "frown", "):": "frown",
    ":((": "frown", ":(((": "frown",
    ";)": "wink", ";-)": "wink", ";D": "wink",
    ":P": "tongue", ":-P": "tongue", ":p": "tongue", ":-p": "tongue", "xP": "tongue",
    ":O": "surprise", ":-O": "surprise", ":o": "surprise", ":-o": "surprise", "o_O": "surprise",
    "O_o": "surprise", "o_o": "surprise",
    ":'(": "cry", ":'-(": "cry", ";(": "cry", "T_T": "cry", "T.T": "cry", ";_;": "cry",
    ":/": "skeptical", ":-/": "skeptical", ":\\": "skeptical", ":S": "skeptical",
    ":|": "neutral", ":-|": "neutral", "-_-": "annoyed",
    ":*": "kiss", ":-*": "kiss",
    "<3": "heart", "<33": "heart", "</3": "broken_heart", "<\\3": "broken_heart",
    ">:(": "angry", ">:-(": "angry",
}  # fmt: skip

# --- Elongation ----------------------------------------------------------------------------
# Words that keep a double letter when an elongated run is collapsed ("goooood" -> "good").
# Anything else collapses to a single letter ("soooo" -> "so", "pleeease" -> "please").
DOUBLE_LETTER_WORDS: frozenset[str] = frozenset(
    {
        "good", "cool", "too", "soon", "see", "feel", "need", "sleep", "been", "food", "look",
        "all", "will", "well", "still", "sweet", "book", "boo", "moon", "keep", "deep", "free",
        "three", "coffee", "agree", "meet", "feet", "tree", "wood", "mood", "school", "pool",
        "fool", "poor", "door", "floor", "full", "kill", "ill", "fall", "call", "tell", "hell",
        "yell", "smell", "off", "stuff", "miss", "kiss", "less", "mess", "boss", "class", "pass",
        "guess", "shall", "add", "odd", "egg", "aww", "oops", "woo", "yay",
        "week", "seen", "teen", "queen", "green", "speed", "bleed", "seem", "beer", "cheers",
        "cheer", "soo", "loose", "lose", "goodbye", "noob", "yummy", "happy", "silly", "really",
        "sorry", "pretty", "funny", "little", "better", "bitter", "sadder", "madder", "cuddle",
        "hello", "zoom", "room", "doom", "gloom", "baby", "kitty", "puppy", "woohoo",
    }
)  # fmt: skip

# --- Transliterated (Latin-script) Armenian ------------------------------------------------
# Strong markers are rarely English; weak ones collide with English or Romance words.
LATIN_ARMENIAN_STRONG: frozenset[str] = frozenset(
    {
        "barev", "barew", "inchpes", "inchpesa", "inchpeses", "vonc", "voncs", "vonces",
        "voch", "voč", "chem", "ches", "chka", "chunem", "chgitem", "shnorhakalutyun",
        "shnorhakal", "shnorhavor", "hima", "inchqan", "inchu", "inchi", "axper",
        "gnanq", "gnacinq", "uzum", "uzem", "kuzem", "sirum", "dzez", "qez", "hajox", "hajogh",
        "gisher", "katak", "ashxatanq", "ashxatum", "kneres", "nerogh",
        "karogh", "mernem", "jogum", "chgitem", "gitem", "gites", "asum", "anum", "galis",
        "gnum", "tesnvenq", "lavem", "lavenq", "xosum", "hayeren",
    }
)  # fmt: skip
LATIN_ARMENIAN_WEAK: frozenset[str] = frozenset(
    {"es", "em", "e", "en", "enq", "eq", "lav", "inch", "jan", "shat", "ayo", "che", "im", "mer",
     "ari", "mama", "tun", "apres", "merci", "qo", "sirun", "luys", "ha", "du", "na", "aper",
     "bari", "karas", "tox", "togh"}
)  # fmt: skip
