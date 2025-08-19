import string
import re
from rapidfuzz import fuzz


def soundex(word: str) -> str:
    """
    Returns the Soundex code for a single word.
    Soundex is an older approach to compare "sound-alike" words.
    Note: This is a simplistic implementation and might not handle edge cases (e.g. non-ASCII).
    """
    # 1. Uppercase the word, remove non-alpha chars
    word = re.sub(r'[^a-zA-Z]', '', word.upper())
    if not word:
        return ""

    # 2. Soundex algorithm:
    #    - First letter is stored directly
    #    - Then encode the rest with digit mapping
    #    - Remove duplicates
    #    - Pad or truncate to length 4
    first_letter = word[0]

    # Dictionary for letter to digit mapping
    mappings = {
        'B': '1', 'F': '1', 'P': '1', 'V': '1',
        'C': '2', 'G': '2', 'J': '2', 'K': '2', 'Q': '2', 'S': '2', 'X': '2', 'Z': '2',
        'D': '3', 'T': '3',
        'L': '4',
        'M': '5', 'N': '5',
        'R': '6'
    }

    # Encode the rest of the letters
    encoded_digits = []
    for char in word[1:]:
        digit = mappings.get(char, '0')
        encoded_digits.append(digit)

    # Remove duplicates
    filtered = []
    previous = None
    for digit in encoded_digits:
        if digit != previous and digit != '0':
            filtered.append(digit)
        previous = digit

    # Combine with the first letter
    soundex_code = first_letter + "".join(filtered)

    # Pad or truncate to length 4
    soundex_code = soundex_code[:4].ljust(4, '0')
    return soundex_code


def remove_special_chars(s: str) -> str:
    """
    Removes punctuation, whitespace, and other special characters from a string.
    Returns lowercase (for case-insensitivity).
    """
    # Use translate or regex; here we use regex to remove all punctuation+whitespace
    return re.sub(r'[^a-zA-Z0-9]+', '', s).lower()


def approximate_substring_search(main_string: str, query: str, threshold: int = 80) -> bool:
    """
    Uses RapidFuzz's `fuzz.partial_ratio` to check if `query` is
    "approximately" in `main_string` with at least `threshold` match (0-100).

    We do a rolling scan over the main string (only for substrings of length
    around len(query)± a buffer). Since `query` <= 20 chars, this is still feasible
    for large main strings, though it can be expensive if main_string >> 100k characters.

    If performance is critical and `main_string` is huge, consider:
    - Segmenting main_string by words or sentences and then applying approximate matching.
    - Using specialized indexing or advanced substring search algorithms.
    """
    len_query = len(query)
    if len_query == 0:
        # Edge case: empty query => trivially found
        return True

    # Lowercase everything for case-insensitive
    main_l = main_string.lower()
    query_l = query.lower()

    # If main_l is extremely large, you may want to break it into chunks or lines
    # to avoid scanning the entire string in one pass.

    # We'll attempt a small +/- buffer around the query length to handle short expansions
    # For example, if query = 10 chars, we might check substrings of length 7 to 13.
    # This can help if the fuzzy match might skip or align differently (like "helxo woorld").
    min_len, max_len = max(1, len_query - 3), len_query + 3

    # Rolling approach: for each index in main_l, extract substring slices
    # in [min_len, max_len], compare with query using partial_ratio.
    # Because query <= 20, we do at most 7 expansions per index (from len_query-3 to len_query+3).
    # This is still O(N * 7) ~ O(N), which might be slow for N >> 100k, but feasible in some cases.
    for start_idx in range(len(main_l)):
        # If there's not enough room left to reach min_len or max_len, break early
        if start_idx + min_len > len(main_l):
            break

        for length in range(min_len, max_len + 1):
            end_idx = start_idx + length
            if end_idx > len(main_l):
                break
            substring = main_l[start_idx:end_idx]
            # RapidFuzz partial_ratio
            score = fuzz.partial_ratio(substring, query_l)
            if score >= threshold:
                return True

    return False


def string_exists(
        main_string: str,
        query: str,
        fuzzy_threshold: int = 80,
        soundex_distance_threshold: int = 1
) -> bool:
    """
    Attempts to find 'query' in 'main_string' by:
      1) Case-insensitive exact match
      2) Match ignoring special characters (punctuation, whitespace, etc.)
      3) Approximate (fuzzy) substring match with RapidFuzz (partial_ratio)
      4) Soundex-based "sound-alike" check for words
         (i.e., we compare the query's Soundex to each word in main_string).
         Because main_string can be huge, we do a tokenized approach.

    :param main_string: Potentially very large text (hundreds of thousands of characters).
    :param query: A short string (<= 20 characters).
    :param fuzzy_threshold: The partial_ratio threshold for approximate substring match (0-100).
    :param soundex_distance_threshold: Max difference allowed in Soundex codes for "sound-alike".
    :return: True if match is found by any of the above conditions, otherwise False.
    """
    # -----------------------
    # 1) Case-insensitive Exact Match
    # -----------------------
    if query.lower() in main_string.lower():
        return True

    # -----------------------
    # 2) Match ignoring special characters
    # -----------------------
    main_clean = remove_special_chars(main_string)
    query_clean = remove_special_chars(query)
    if query_clean in main_clean:
        return True

    # -----------------------
    # 3) Approximate (Fuzzy) Substring Match
    # -----------------------
    if approximate_substring_search(main_string, query, threshold=fuzzy_threshold):
        return True

    # -----------------------
    # 4) Soundex-based check
    #    We'll do a simple approach: split main_string into words, compare each word's Soundex
    #    to the query's Soundex. We'll return True if the Soundex codes are "close enough".
    #    By default, "close enough" means codes are identical or differ by at most 1 character
    #    (this is arbitrary—Soundex is not always used this way).
    # -----------------------
    query_soundex = soundex(query)
    # Tokenize main_string on whitespace/punctuation; compare soundex codes
    words = re.findall(r"[A-Za-z0-9]+", main_string)
    for w in words:
        w_soundex = soundex(w)
        # Check if the codes match or are "close enough" in terms of raw Hamming distance
        # (since both codes are length 4).
        distance = sum(1 for a, b in zip(query_soundex, w_soundex) if a != b)
        if distance <= soundex_distance_threshold:
            return True

    return False


# Example usage (quick test)
if __name__ == "__main__":
    text = (
        "This is an example: 'Hello, W0rld!!'. Sometimes spelled as 'Wrld' or 'Wurld'.\n"
        "We also have a tricky phrase: 'H3ll0 W0rld'."
    )
    queries = [
        "W0rld",  # direct (exact ignoring case)
        "world",  # ignoring special chars or partial fuzzy
        "wurld",  # fuzzy or soundex
        "hello w0rld"  # ignoring punctuation or fuzzy
    ]

    for q in queries:
        found = string_exists(text, q, fuzzy_threshold=75, soundex_distance_threshold=1)
        print(f"'{q}' found in text? {found}")
