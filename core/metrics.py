import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import cmudict
import string
import numpy as np
from scipy.stats import norm
from pathlib import Path

# Load CMU Pronouncing Dictionary
pronunciation_dict = cmudict.dict()

def syllable_count(word):
    """Determine the number of syllables in a word."""
    word = word.lower()
    if word in pronunciation_dict:
        return max([len([phoneme for phoneme in phonetic if phoneme[-1].isdigit()]) for phonetic in pronunciation_dict[word]])
    return 1  # Assume one syllable if the word isn't found

def is_polysyllabic(word):
    """Identify if a word is polysyllabic (i.e., has 3 or more syllables)."""
    return syllable_count(word) >= 3

def calculate_complexity_index(text):
    """
    Calculate a complexity index (0-100) based on Flesch-Kincaid grade level and percentage of complex words.
    
    Args:
        text (str): The text to analyze
        
    Returns:
        float: Complexity index from 0-100
    """
    # Handle empty text
    if not text or not text.strip():
        return 0
    
    sentences = sent_tokenize(text)
    tokens = word_tokenize(text)
    
    sentence_count = max(1, len(sentences))
    word_count = max(1, len(tokens))
    
    # Calculate Flesch-Kincaid Grade Level using the same formula as the reference code
    total_syllables = sum(syllable_count(token) for token in tokens)
    fk_grade_level = 0.39 * (word_count / sentence_count) + 11.8 * (total_syllables / word_count) - 15.59
    
    # Calculate percentage of complex words
    complex_word_count = sum(1 for token in tokens if is_polysyllabic(token))
    percent_complex_words = (complex_word_count / word_count) * 100
    
    # Cap FK grade at 14 (college level)
    fk_grade_level = min(fk_grade_level, 14)
    
    # Cap percent complex at 20%
    percent_complex_words = min(percent_complex_words, 20)
    
    # Normalize scores to 0-100 range
    fk_normalized = (fk_grade_level / 14) * 100
    complex_normalized = (percent_complex_words / 20) * 100
    
    # Average the two normalized scores for the final complexity index
    complexity_index = (fk_normalized + complex_normalized) / 2
    
    return round(complexity_index, 2)

# Calculates a slop score for a provided text

import json
import re
import numpy as np
from joblib import Parallel, delayed

def load_and_preprocess_slop_words():
    with open('data/slop_phrase_prob_adjustments.json', 'r') as f:
        slop_phrases = json.load(f)
    
    phrase_weighting = [1.0 - prob_adjustment for word, prob_adjustment in slop_phrases]
    max_score = max(phrase_weighting)
    scaled_weightings = [score / max_score for score in phrase_weighting]
    n_slop_words = 600
    return {word.lower(): score for (word, _), score in zip(slop_phrases[:n_slop_words], scaled_weightings[:n_slop_words])}

def extract_text_blocks(file_path, compiled_pattern):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    matches = compiled_pattern.findall(content)
    return '\n'.join(matches)

def calculate_slop_score_chunk(args):
    text, slop_words_chunk = args
    return sum(
        score * len(re.findall(r'\b' + re.escape(word) + r'\b', text))
        for word, score in slop_words_chunk.items()
    )

def split_into_chunks(slop_words, num_chunks):
    slop_words_items = list(slop_words.items())
    chunk_size = len(slop_words_items) // num_chunks
    if chunk_size == 0:
        chunk_size = 1
    return [dict(slop_words_items[i:i + chunk_size]) for i in range(0, len(slop_words_items), chunk_size)]


# Call this to function to calculate a slop score.
# This is the way it's calculated for the eqbench creative writing leaderboard.
def calculate_slop_index(extracted_text):    
    slop_words = load_and_preprocess_slop_words()
    
    num_chunks = 12 #mp.cpu_count()
    slop_words_chunks = split_into_chunks(slop_words, num_chunks)
    
    if not extracted_text:
        slop_index = 0.0
    else:
        # Parallelize the calculation using joblib
        slop_scores = Parallel(n_jobs=num_chunks)(delayed(calculate_slop_score_chunk)((extracted_text, chunk)) for chunk in slop_words_chunks)
        
        slop_score = sum(slop_scores)
        total_words = len(extracted_text.split())
        slop_index = (slop_score / total_words) * 1000 if total_words > 0 else 0
    return slop_index


import re
import json # Keep if you might load texts from JSON elsewhere
from collections import Counter, defaultdict # Added defaultdict
from tqdm import tqdm
import numpy as np
from wordfreq import word_frequency
import os # Keep if needed elsewhere
import string # Keep if needed elsewhere
from typing import List, Tuple, Dict, Set # Added type hints

# --- BEGIN HELPER FUNCTIONS (Verified & Slightly Corrected) ---

# Define the set of forbidden substrings (lowercase)
FORBIDDEN_SUBSTRINGS = {
    "jolyne", "yennefer", "revy", "stoer", "beretta", "sisyphus",
    "retiarius", "alucard", "azra", "jigen", "ludus", "professore",
    "alessandra", "midas", "chewy",
    "makima", "neegan", "immateria", "offworlder", "piguaquan",
    "vengerberg", "lanista", "morska", "scythan", "woolong",
    "cujoh", "underhold",
    "darkroom", "bookstore", "guildmaster", "volkov", "katra", "arthur",
    "lucifer", "lilith", "antares", "chronowatch", "nettle", "nettes",
    "busker", "rewound", "rewind", "laddie",
    "spike", "elliot", "vespa", "alasdair", "sorceress", "mora",
    "lighthouse", "gladiator", "bookshop", "koala", "crow", "boulder",
    "jt", "interstellar", "dreamscape", "xxxx"
}


# Regex pattern for word extraction (compile once)
WORD_PATTERN = re.compile(r"\b[a-zA-Z]+(?:'[a-zA-Z]+)?")

def normalize_apostrophes(text):
    """Replaces common apostrophe variants with the standard ASCII apostrophe."""
    if not isinstance(text, str):
        return ""
    text = text.replace("’", "'")
    text = text.replace("‘", "'")
    text = text.replace("ʼ", "'")
    return text

def _extract_words(normalized_text: str, min_length: int = 4) -> List[str]:
    """Extracts words meeting criteria from normalized text."""
    words = WORD_PATTERN.findall(normalized_text)
    return [
        word for word in words
        if len(word) >= min_length or "'" in word
    ]

def get_word_counts(texts: List[str], min_length: int = 4) -> Counter:
    """
    Count overall word frequencies in a list of texts.
    """
    word_counts = Counter()
    for text in tqdm(texts, desc="Counting words", leave=False, disable=True): # Often nested, disable bar
        if not isinstance(text, str):
            continue
        normalized_text = normalize_apostrophes(text.lower())
        words = _extract_words(normalized_text, min_length)
        word_counts.update(words)
    return word_counts

# --- NEW HELPER FUNCTION ---
def get_word_prompt_map(texts_with_ids: List[Tuple[str, str]], min_length: int = 4) -> Dict[str, Set[str]]:
    """
    Creates a map of words to the set of prompt IDs they appear in.
    """
    word_prompts = defaultdict(set)
    for text, prompt_id in tqdm(texts_with_ids, desc="Mapping words to prompts", leave=False, disable=True): # Often nested
        if not isinstance(text, str):
            continue
        normalized_text = normalize_apostrophes(text.lower())
        words = _extract_words(normalized_text, min_length)
        for word in words:
            word_prompts[word].add(prompt_id)
    return dict(word_prompts) # Convert back to standard dict if preferred

# --- EXISTING HELPER FUNCTIONS (Unchanged, use original code) ---
def filter_mostly_numeric(word_counts):
    """Filters out words containing a high proportion of digits."""
    def is_mostly_numbers(word):
        if not word: return False
        digit_count = sum(c.isdigit() for c in word)
        return (digit_count / len(word) > 0.2) if len(word) > 0 else False

    return Counter({word: count for word, count in word_counts.items() if not is_mostly_numbers(word)})

KNOWN_CONTRACTIONS_S = {
    "it's", "that's", "what's", "who's", "he's", "she's",
    "there's", "here's", "where's", "when's", "why's", "how's",
    "let's"
}

def merge_plural_possessive_s(word_counts):
    """Merges counts of possessive words ending in 's with their base words, excluding known contractions."""
    merged_counts = Counter()
    for word, count in word_counts.items(): # No tqdm needed here, usually fast
        if word.endswith("'s") and word not in KNOWN_CONTRACTIONS_S:
            base_word = word[:-2]
            if base_word:
                merged_counts[base_word] += count
        else:
            merged_counts[word] += count
    return merged_counts

def filter_forbidden_words(word_counts, forbidden_substrings):
    """Filters out words containing any of the forbidden substrings (case-insensitive)."""
    if not forbidden_substrings:
        return word_counts
    return Counter({
        word: count for word, count in word_counts.items()
        if not any(sub in word.lower() for sub in forbidden_substrings)
    })

def filter_by_minimum_count(word_counts, min_count):
    """Filters out words that appear less than or equal to min_count times."""
    if min_count <= 0:
        return word_counts
    return Counter({word: count for word, count in word_counts.items() if count > min_count})

def analyze_word_rarity(word_counts):
    """Analyzes word rarity based on corpus and wordfreq frequencies."""
    if not word_counts: return {}, {}, np.nan, np.nan, np.nan
    total_words = sum(word_counts.values())
    if total_words == 0: return {}, {}, np.nan, np.nan, np.nan

    corpus_frequencies = {word: count / total_words for word, count in word_counts.items()}
    wordfreq_frequencies = {}
    for word in tqdm(list(word_counts.keys()), desc="Fetching wordfreq data", leave=False, disable=True): # Often nested
        wordfreq_frequencies[word] = word_frequency(word, 'en')

    valid_words = [word for word, freq in wordfreq_frequencies.items() if freq > 0]
    if not valid_words: return corpus_frequencies, wordfreq_frequencies, np.nan, np.nan, np.nan

    corpus_freq_list = [corpus_frequencies[word] for word in valid_words]
    wordfreq_freq_list = [wordfreq_frequencies[word] for word in valid_words]

    avg_corpus_rarity = np.mean([-np.log10(freq) for freq in corpus_freq_list]) if corpus_freq_list else np.nan
    avg_wordfreq_rarity = np.mean([-np.log10(freq) for freq in wordfreq_freq_list]) if wordfreq_freq_list else np.nan

    correlation = np.nan
    if len(corpus_freq_list) >= 2:
        with np.errstate(divide='ignore', invalid='ignore'): # Suppress warnings for log10(0) or corrcoef issues
            correlation_matrix = np.corrcoef(corpus_freq_list, wordfreq_freq_list)
            if isinstance(correlation_matrix, np.ndarray) and correlation_matrix.shape == (2, 2):
                 correlation = correlation_matrix[0, 1]

    return corpus_frequencies, wordfreq_frequencies, avg_corpus_rarity, avg_wordfreq_rarity, correlation


def find_over_represented_words(corpus_frequencies, wordfreq_frequencies, top_n=50000):
    """Finds words most over-represented compared to wordfreq."""
    over_representation = {}
    for word, corpus_freq in corpus_frequencies.items():
        wordfreq_freq = wordfreq_frequencies.get(word, 0)
        if wordfreq_freq > 0:
            over_representation[word] = corpus_freq / wordfreq_freq
        elif corpus_freq > 0:
             over_representation[word] = corpus_freq / 1e-12

    return sorted(over_representation.items(), key=lambda item: item[1], reverse=True)[:top_n]


# --- MODIFIED MAIN FUNCTIONS ---

def _get_filtered_word_counts(
    texts_with_ids: List[Tuple[str, str]],
    min_repetition_count: int,
    min_prompt_ids: int = 2 # New parameter: Minimum number of unique prompt IDs a word must appear in
) -> Counter:
    """
    Internal helper to get word counts filtered by numeric, possessive,
    forbidden, minimum prompts, and minimum overall count.
    """
    if not texts_with_ids:
        return Counter()

    # Extract all texts for overall counting
    all_texts = [text for text, _ in texts_with_ids]
    if not all_texts:
        return Counter()

    # 1. Get overall raw counts
    raw_word_counts = get_word_counts(all_texts)
    # 2. Filter numeric words
    filtered_counts_numeric = filter_mostly_numeric(raw_word_counts)
    # 3. Merge possessives
    merged_counts = merge_plural_possessive_s(filtered_counts_numeric)
    # 4. Filter forbidden words
    filtered_counts_forbidden = filter_forbidden_words(merged_counts, FORBIDDEN_SUBSTRINGS)

    # *** NEW: Filter by minimum prompt IDs ***
    if min_prompt_ids > 1:
        # Get the map of word -> set(prompt_ids)
        word_prompt_map = get_word_prompt_map(texts_with_ids)
        # Identify words appearing in enough distinct prompts
        multi_prompt_words = {
            word for word, prompt_ids in word_prompt_map.items()
            if len(prompt_ids) >= min_prompt_ids
        }
        # Filter the counts to keep only multi-prompt words
        filtered_counts_multi_prompt = Counter({
            word: count for word, count in filtered_counts_forbidden.items()
            if word in multi_prompt_words
        })
    else:
        # If min_prompt_ids is 1 or less, skip this filtering step
        filtered_counts_multi_prompt = filtered_counts_forbidden


    # 5. Filter by minimum overall repetition count (applied AFTER multi-prompt filter)
    final_counts = filter_by_minimum_count(filtered_counts_multi_prompt, min_repetition_count)

    return final_counts


def calculate_repetition_metric(
    texts_with_ids: List[Tuple[str, str]],
    top_n: int = 100,
    min_repetition_count: int = 5,
    min_prompt_ids: int = 2 # Minimum number of unique prompt IDs a word must appear in
) -> float:
    """
    Calculate a repetition metric based on over-represented words,
    filtering out forbidden words, words occurring infrequently overall,
    and words not appearing in at least `min_prompt_ids` unique prompts.

    Args:
        texts_with_ids: List of tuples (text_sample, prompt_id) to analyze.
        top_n: Number of top over-represented words to consider for the score.
        min_repetition_count: Minimum number of times a word must appear in the
                              *entire* corpus (across all prompts) to be considered.
                              Words appearing `min_repetition_count` or fewer
                              times are excluded *after* multi-prompt filtering.
        min_prompt_ids: Minimum number of unique prompt IDs a word must have
                        appeared in to be considered for the analysis. Defaults to 2.

    Returns:
        float: Repetition score (sum of corpus frequencies of top_n
               over-represented words, as a percentage), considering only words
               meeting all filtering criteria.
    """
    final_counts = _get_filtered_word_counts(texts_with_ids, min_repetition_count, min_prompt_ids)

    if not final_counts:
        print(f"Warning: No words remaining after filtering (min count > {min_repetition_count}, min prompts >= {min_prompt_ids}).")
        return 0.0

    # Analyze rarity (uses the final filtered counts)
    corpus_frequencies, wordfreq_frequencies, _, _, _ = analyze_word_rarity(final_counts)

    if not corpus_frequencies:
        print("Warning: Corpus frequencies could not be calculated (possibly all remaining words have zero wordfreq).")
        return 0.0

    # Find over-represented words (based on final filtered data)
    over_represented = find_over_represented_words(corpus_frequencies, wordfreq_frequencies, top_n=top_n)

    # Calculate score: Sum the corpus frequencies of these top words
    repetition_score = sum(corpus_frequencies.get(word, 0) for word, score in over_represented)

    # Normalize to percentage
    normalized_score = repetition_score * 100

    return round(normalized_score, 4)


def get_top_repetitive_words(
    texts_with_ids: List[Tuple[str, str]],
    top_n: int = 20,
    min_repetition_count: int = 5,
    min_prompt_ids: int = 2 # Minimum number of unique prompt IDs a word must appear in
) -> List[Tuple[str, float]]:
    """
    Get the top over-represented words with their scores, filtering out
    forbidden words, words occurring infrequently overall, and words not
    appearing in at least `min_prompt_ids` unique prompts.

    Args:
        texts_with_ids: List of tuples (text_sample, prompt_id) to analyze.
        top_n: Number of top over-represented words to return.
        min_repetition_count: Minimum number of times a word must appear in the
                              *entire* corpus (across all prompts) to be considered.
                              Words appearing `min_repetition_count` or fewer
                              times are excluded *after* multi-prompt filtering.
        min_prompt_ids: Minimum number of unique prompt IDs a word must have
                        appeared in to be considered for the analysis. Defaults to 2.

    Returns:
        List of tuples (word, over_representation_score) for words meeting
        all filtering criteria.
    """
    final_counts = _get_filtered_word_counts(texts_with_ids, min_repetition_count, min_prompt_ids)

    if not final_counts:
        print(f"Warning: No words remaining after filtering (min count > {min_repetition_count}, min prompts >= {min_prompt_ids}).")
        return []

    # Analyze rarity
    corpus_frequencies, wordfreq_frequencies, _, _, _ = analyze_word_rarity(final_counts)

    if not corpus_frequencies:
         print("Warning: Corpus frequencies could not be calculated (possibly all remaining words have zero wordfreq).")
         return []

    # Find over-represented words
    over_represented = find_over_represented_words(corpus_frequencies, wordfreq_frequencies, top_n=top_n)

    # Return words and their *over-representation* score, not frequency
    return over_represented



# --- Add these imports ---
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import ngrams as nltk_ngrams # Alias to avoid potential name conflict
import string
from collections import Counter, defaultdict # Counter/defaultdict likely already there

# --- Add NLTK downloads (run once) ---
if False:
    try:
        nltk.data.find('tokenizers/punkt')
    except nltk.downloader.DownloadError:
        nltk.download('punkt', quiet=True)
    try:
        nltk.data.find('corpora/stopwords')
    except nltk.downloader.DownloadError:
        nltk.download('stopwords', quiet=True)

# --- Initialize stopwords and punctuation (global scope is fine here) ---
stop_words = set(stopwords.words('english'))
punctuation_set = set(string.punctuation) # Use a different name than the module


# ── helper ────────────────────────────────────────────────────────────────────
def _load_human_ngram_freqs(human_profile_path: Path, n: int) -> Dict[str, float]:
    """
    Load the pre-computed human baseline n-gram distribution and return a
    normalised frequency map (freq = count / total_counts).
    """
    with open(human_profile_path, "r", encoding="utf-8") as f:
        hp = json.load(f)["human-authored"]

    key_lookup = {2: "top_bigrams", 3: "top_trigrams"}
    key = key_lookup.get(n)
    if key is None or key not in hp:
        return {}

    items = hp[key]
    total = sum(int(it["frequency"]) for it in items if "frequency" in it)
    if total == 0:
        return {}

    freqs: Dict[str, float] = {}
    for it in items:
        try:
            ngram_str = " ".join(tok.lower() for tok in word_tokenize(it["ngram"]) if tok.isalpha())
            freqs[ngram_str] = int(it["frequency"]) / total
        except Exception:
            continue
    return freqs


# ── main ──────────────────────────────────────────────────────────────────────
def get_multi_prompt_ngrams(
    prompts_data: Dict[str, List[str]],
    n: int,
    top_k: int = 20,
    min_prompt_ids: int = 2,
    human_profile_path: Path = Path("data/human_writing_profile.json"),
) -> List[Tuple[Tuple[str, ...], int]]:
    """
    Extract n-grams that (i) appear in at least `min_prompt_ids` unique prompts and
    (ii) are the top-k most over-used relative to the human baseline.

    Returns
    -------
    List[Tuple[Tuple[str, ...], int]]
        Each tuple is (ngram_as_tuple, raw_incidence_count_in_model_texts),
        sorted by descending count.
    """
    global stop_words
    if stop_words is None:
        stop_words = set(stopwords.words("english"))

    # 1. collect counts + prompt coverage
    ngram_counts: Counter = Counter()
    ngram_prompt_map: defaultdict = defaultdict(set)

    for prompt_id, texts in prompts_data.items():
        for text in texts:
            if not isinstance(text, str) or not text.strip():
                continue
            tokens = [
                t.lower() for t in word_tokenize(text)
                if t.isalpha() and t.lower() not in stop_words
            ]
            if len(tokens) < n:
                continue
            for ng in nltk_ngrams(tokens, n):
                ngram_counts[ng] += 1
                ngram_prompt_map[ng].add(prompt_id)

    # 2. filter by prompt coverage
    filtered_counts = {
        ng: cnt for ng, cnt in ngram_counts.items()
        if len(ngram_prompt_map[ng]) >= min_prompt_ids
    }
    if not filtered_counts:
        return []

    # 3. model frequencies (string-keyed for easy comparison)
    counts_str = {" ".join(ng): cnt for ng, cnt in filtered_counts.items()}
    total_model = sum(counts_str.values())
    model_freqs = {k: v / total_model for k, v in counts_str.items()}

    # 4. human baseline frequencies
    human_freqs = _load_human_ngram_freqs(human_profile_path, n)
    if not human_freqs:
        # no baseline: keep most frequent n-grams by raw count
        top_ngs = sorted(filtered_counts.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        return [(ng, cnt) for ng, cnt in top_ngs]

    # 5. over-use ratios
    epsilon = 1e-12
    overuse_ratio = {
        ng_str: model_freqs[ng_str] / (human_freqs.get(ng_str, 0.0) + epsilon)
        for ng_str in model_freqs
    }

    # 6. choose top-k by ratio
    top_by_ratio = sorted(overuse_ratio.items(), key=lambda kv: kv[1], reverse=True)[:top_k]

    # 7. build result list with raw counts
    results: List[Tuple[Tuple[str, ...], int]] = []
    for ng_str, _ in top_by_ratio:
        ng_tuple = tuple(ng_str.split())
        count = filtered_counts.get(ng_tuple, 0)
        results.append((ng_tuple, count))

    # 8. final sort by incidence count
    results.sort(key=lambda x: x[1], reverse=True)
    return results



# ── NEW helper ────────────────────────────────────────────────────────────────
def has_sentence_end_in_the_middle(phrase: str) -> bool:
    """
    Returns True if there is . ? or ! in the middle of 'phrase'
    (not counting the very last character).
    """
    s = phrase.strip()
    if len(s) <= 2:
        return False
    for c in ".?!":
        if c in s[:-1]:  # check everything except the last character
            return True
    return False

def _process_one_text_for_substrings_multi(
    text: str,
    ngram_sets_by_len: Dict[int, Set[Tuple[str, ...]]],
    stop_words_set: Set[str],
) -> Counter:
    """
    Same idea as the old process_one_text_for_substrings() but supports
    *multiple* n-values in a single pass.  Returns Counter{exact_phrase: count}.
    """
    local_counter = Counter()
    if not isinstance(text, str) or not text.strip():
        return local_counter

    # 1. naive tokenisation + offsets
    tokens_with_spans = []
    offset = 0
    for tk in word_tokenize(text):
        idx = text.find(tk, offset)
        if idx == -1:
            continue
        tokens_with_spans.append((tk, idx, idx + len(tk)))
        offset = idx + len(tk)

    # 2. build cleaned tokens & offset map
    cleaned_tokens, char_map = [], []
    for tk, st, en in tokens_with_spans:
        lt = tk.lower()
        if lt.isalpha() and lt not in stop_words_set:
            cleaned_tokens.append(lt)
            char_map.append((st, en))

    # 3. slide over each requested n
    clen = len(cleaned_tokens)
    for n, ngram_set in ngram_sets_by_len.items():
        if clen < n:
            continue
        limit = clen - n + 1
        for i in range(limit):
            cand = tuple(cleaned_tokens[i : i + n])
            if cand in ngram_set:
                st = char_map[i][0]
                en = char_map[i + n - 1][1]
                local_counter[text[st:en]] += 1
    return local_counter

from functools import partial
from multiprocessing import Pool

# ── NEW main ──────────────────────────────────────────────────────────────────
def extract_slop_phrases(
    texts: List[str],
    ngram_list: List[Tuple[Tuple[str, ...], int]],
    top_k_ngrams: int = 1000,          # keep this for parity
    top_phrases_to_save: int = 10_000,
    chunksize: int = 50,
):
    """
    Pull the exact over-used phrases (any n-gram length) from raw texts.

    Parameters
    ----------
    texts : list[str]
        The corpus you want to mine.
    ngram_list : list[(tuple, count)]
        Output of get_multi_prompt_ngrams() (can mix 2-grams, 3-grams, …).
    """


    if not ngram_list:
        return

    # 1. limit to top-k by *relative-freq ranking* already present in ngram_list
    ngram_list = ngram_list[:top_k_ngrams]

    # 2. bucket by n
    ngram_sets_by_len: Dict[int, Set[Tuple[str, ...]]] = defaultdict(set)
    for ng, _cnt in ngram_list:
        ngram_sets_by_len[len(ng)].add(tuple(tok.lower() for tok in ng))

    # 3. MP extraction
    process_func = partial(
        _process_one_text_for_substrings_multi,
        ngram_sets_by_len=ngram_sets_by_len,
        stop_words_set=stop_words,
    )
    num_procs = min(os.cpu_count() or 1, 12)
    with Pool(processes=num_procs) as p:
        counters = list(
            tqdm(p.imap_unordered(process_func, texts, chunksize=chunksize),
                 total=len(texts),
                 desc="substring extraction")
        )

    combined = Counter()
    for c in counters:
        combined.update(c)

    # 4. punctuation filter
    filtered = Counter(
        {ph: c for ph, c in combined.items() if not has_sentence_end_in_the_middle(ph)}
    )

    # 5. save
    top_phrases = filtered.most_common(top_phrases_to_save)
    return top_phrases




import json
import re
import os # For checking file existence
from collections import Counter # Useful for counting n-grams in text

# Assume NLTK is available and imported if needed for tokenization/ngrams
# If not using NLTK, adjust tokenization/ngram generation accordingly.
try:
    import nltk
    from nltk.tokenize import word_tokenize
    from nltk import ngrams
    nltk_available = True
    # Minimal check for resources needed
    if False:
        try:
            nltk.data.find('tokenizers/punkt', quiet=True)
        except:
            print("Warning: NLTK 'punkt' resource might be needed. Downloading...")
            nltk.download('punkt', quiet=True)

except ImportError:
    print("Warning: NLTK not installed. Using basic split() for tokenization.")
    nltk_available = False

# --- Helper Function to Load Slop Lists ---

def load_slop_list_to_set(filename):
    """Loads slop words/phrases from the specific JSON format into a set."""
    if not os.path.exists(filename):
        print(f"Warning: Slop file not found: {filename}. Returning empty set.")
        return set()
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # Extract the first element from each inner list and lowercase it
        # Handles format like [["word1"], ["word2 phrase"], ...]
        slop_items = {item[0].lower() for item in data if item} # Ensure inner list is not empty
        print(f"Loaded {len(slop_items)} items from {filename}")
        return slop_items
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {filename}. Returning empty set.")
        return set()
    except Exception as e:
        print(f"Error loading {filename}: {e}. Returning empty set.")
        return set()

# --- New Slop Index Calculation Function ---

def calculate_slop_index_new(extracted_text, debug=True):
    """
    Calculates a slop index based on hits in word, bigram, and trigram slop lists.

    Args:
        extracted_text (str): The text to analyze.
        debug (bool): If True, prints the hit counts for each list.

    Returns:
        float: The calculated slop index.
    """
    # 1. Load Slop Lists
    slop_words_set = load_slop_list_to_set('data/slop_list.json')
    slop_bigrams_set = load_slop_list_to_set('data/slop_list_bigrams.json')
    slop_trigrams_set = load_slop_list_to_set('data/slop_list_trigrams.json')

    # Check if any lists were loaded
    if not slop_words_set and not slop_bigrams_set and not slop_trigrams_set:
        print("Error: No slop lists could be loaded. Returning slop index 0.")
        return 0.0

    if not extracted_text or not isinstance(extracted_text, str):
        if debug:
            print("Input text is empty or invalid.")
            print(f"Word Hits: 0")
            print(f"Bigram Hits: 0")
            print(f"Trigram Hits: 0")
        return 0.0

    # 2. Preprocess Text and Count Total Words
    lower_text = extracted_text.lower()
    # Use NLTK tokenizer if available for better handling of punctuation,
    # otherwise use a simple regex split for words.
    if nltk_available:
        tokens = [token for token in word_tokenize(lower_text) if token.isalnum()] # Keep alphanumeric
    else:
        tokens = re.findall(r'\b\w+\b', lower_text) # Simple word split

    total_words = len(tokens)
    if total_words == 0:
        if debug:
            print("No valid words found in the text after tokenization.")
            print(f"Word Hits: 0")
            print(f"Bigram Hits: 0")
            print(f"Trigram Hits: 0")
        return 0.0

    # 3. Count Hits
    word_hits = 0
    bigram_hits = 0
    trigram_hits = 0

    # Count word hits
    if slop_words_set:
        word_hits = sum(1 for token in tokens if token in slop_words_set)

    # Count bigram hits
    if slop_bigrams_set and len(tokens) >= 2:
        text_bigrams = ngrams(tokens, 2) if nltk_available else zip(tokens, tokens[1:])
        for bigram_tuple in text_bigrams:
            bigram_str = ' '.join(bigram_tuple)
            if bigram_str in slop_bigrams_set:
                bigram_hits += 1

    # Count trigram hits
    if slop_trigrams_set and len(tokens) >= 3:
        text_trigrams = ngrams(tokens, 3) if nltk_available else zip(tokens, tokens[1:], tokens[2:])
        for trigram_tuple in text_trigrams:
            trigram_str = ' '.join(trigram_tuple)
            if trigram_str in slop_trigrams_set:
                trigram_hits += 1

    # 4. Calculate Final Score
    total_slop_score = word_hits + 2*bigram_hits + 8*trigram_hits
    # Use the same normalization factor as the original function for consistency
    slop_index = (total_slop_score / total_words) * 1000 if total_words > 0 else 0

    # 5. Debug Output
    if debug:
        print("--- Slop Index Debug ---")
        print(f"Total Words Analyzed: {total_words}")
        print(f"Word Hits: {word_hits} (using {len(slop_words_set)} slop words)")
        print(f"Bigram Hits: {bigram_hits} (using {len(slop_bigrams_set)} slop bigrams)")
        print(f"Trigram Hits: {trigram_hits} (using {len(slop_trigrams_set)} slop trigrams)")
        print(f"Total Hits: {total_slop_score}")
        print(f"Calculated Slop Index: {slop_index:.4f}")
        print("------------------------")

    return slop_index

