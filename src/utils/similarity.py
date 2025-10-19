import textdistance

def compute_similarity(word1, word2):
    """Returns similarity between two words (range 0-1)"""
    return textdistance.levenshtein.normalized_similarity(word1, word2)