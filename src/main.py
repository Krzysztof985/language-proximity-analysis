import os
from src.analysis_backend import (
    run_levenshtein_analysis,
    run_embedding_analysis
)

BASE_LANGUAGE = "en"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "../data")
RESULTS_DIR = os.path.join(BASE_DIR, "../results")


def uruchom_analize(languages, method="levenshtein"):
    """
    Funkcja wywoływana przez GUI (oknoMAIN.py)
    """
    if not languages:
        raise ValueError("Nie wybrano żadnych języków.")

    # zawsze dodaj język bazowy
    if BASE_LANGUAGE not in languages:
        languages.append(BASE_LANGUAGE)

    if method == "embedding":
        run_embedding_analysis(
            languages=languages,
            data_dir=DATA_DIR,
            results_dir=RESULTS_DIR,
            base_language=BASE_LANGUAGE
        )
    else:
        run_levenshtein_analysis(
            languages=languages,
            data_dir=DATA_DIR,
            results_dir=RESULTS_DIR,
            base_language=BASE_LANGUAGE
        )
