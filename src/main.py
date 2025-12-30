import os
from src.analysis_backend import (
    run_levenshtein_analysis,
    run_embedding_analysis
)
from deep_translator import GoogleTranslator
from typing import List, Dict

# USUNIĘTO: BASE_LANGUAGE = "en" - nie ma już domyślnego języka

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "../data")
RESULTS_DIR = os.path.join(BASE_DIR, "../results")


def tlumacz_slowka(slowka: List[str], jezyki: List[str]) -> Dict[str, List[str]]:
    """
    Tłumaczy listę słów na wybrane języki.

    Args:
        slowka: Lista słów do przetłumaczenia
        jezyki: Lista kodów języków docelowych

    Returns:
        Słownik z tłumaczeniami w formacie {język: [tłumaczenia]}
    """
    tlumaczenia = {}

    for jezyk in jezyki:
        try:
            print(f"Tłumaczenie na {jezyk}...")
            tlumaczone = []

            for i, slowo in enumerate(slowka):
                try:
                    # Tłumaczenie każdego słowa
                    tlumaczenie = GoogleTranslator(
                        source="auto",
                        target=jezyk
                    ).translate(slowo)
                    tlumaczone.append(tlumaczenie)

                    # Wyświetlaj postęp co 10 słów
                    if (i + 1) % 10 == 0:
                        print(f"  Przetłumaczono {i + 1}/{len(slowka)} słów")
                except Exception as e:
                    print(f"  Błąd tłumaczenia '{slowo}' na {jezyk}: {e}")
                    tlumaczone.append(f"(błąd: {slowo})")

            tlumaczenia[jezyk] = tlumaczone
            print(f"✓ Przetłumaczono na {jezyk}: {len(tlumaczone)} słów")

        except Exception as e:
            print(f"✗ Błąd podczas tłumaczenia na {jezyk}: {e}")
            tlumaczenia[jezyk] = [f"(błąd tłumaczenia)" for _ in slowka]

    return tlumaczenia


def pokaz_tlumaczenia_gui(jezyki: List[str], kategoria: str = None):
    """
    Funkcja wywoływana przez GUI do wyświetlania tłumaczeń.

    Args:
        jezyki: Lista kodów języków
        kategoria: Nazwa kategorii (opcjonalnie)
    """
    if not jezyki:
        raise ValueError("Nie wybrano żadnych języków.")

    # USUNIĘTO: automatyczne dodawanie języka angielskiego

    # Wczytaj słówka z plików tematycznych
    slowka = []
    if kategoria:
        plik = os.path.join(DATA_DIR, f"{kategoria}.txt")
        if os.path.exists(plik):
            with open(plik, "r", encoding="utf-8") as f:
                slowka = [linia.strip() for linia in f if linia.strip()]

    # Jeśli nie ma kategorii, użyj pierwszego dostępnego pliku
    if not slowka:
        for filename in os.listdir(DATA_DIR):
            if filename.endswith(".txt"):
                plik = os.path.join(DATA_DIR, filename)
                with open(plik, "r", encoding="utf-8") as f:
                    slowka = [linia.strip() for linia in f if linia.strip()]
                break

    if not slowka:
        raise ValueError("Nie znaleziono plików z danymi w folderze data/")

    print(f"Znaleziono {len(slowka)} słów")

    # Przeprowadź tłumaczenie
    return tlumacz_slowka(slowka, jezyki)


def uruchom_analize(languages, method="levenshtein"):
    """
    Funkcja wywoływana przez GUI (oknoMAIN.py)

    Args:
        languages: Lista kodów języków z checkboxów
        method: Metoda analizy ('levenshtein' lub 'embedding')
    """
    if not languages:
        raise ValueError("Nie wybrano żadnych języków.")

    # USUNIĘTO: automatyczne dodawanie języka angielskiego

    if len(languages) < 2:
        raise ValueError("Wybierz co najmniej dwa języki do analizy.")

    print(f"Uruchamianie analizy dla języków: {languages}")
    print(f"Metoda: {method}")

    # Wybór języka bazowego - pierwszy z listy
    base_language = languages[0]
    print(f"Język bazowy: {base_language}")

    if method == "embedding":
        run_embedding_analysis(
            languages=languages,
            data_dir=DATA_DIR,
            results_dir=RESULTS_DIR,
            base_language=base_language  # Dynamiczny język bazowy
        )
    else:
        run_levenshtein_analysis(
            languages=languages,
            data_dir=DATA_DIR,
            results_dir=RESULTS_DIR,
            base_language=base_language  # Dynamiczny język bazowy
        )


def get_available_categories() -> List[str]:
    """
    Zwraca listę dostępnych kategorii (plików txt w data/).

    Returns:
        Lista nazw kategorii
    """
    kategorie = []
    for filename in os.listdir(DATA_DIR):
        if filename.endswith(".txt"):
            kategoria = filename.replace(".txt", "")
            kategorie.append(kategoria)
    return sorted(kategorie)


def get_words_from_category(kategoria: str) -> List[str]:
    """
    Pobiera słowa z wybranej kategorii.

    Args:
        kategoria: Nazwa kategorii

    Returns:
        Lista słów z kategorii
    """
    plik = os.path.join(DATA_DIR, f"{kategoria}.txt")
    if not os.path.exists(plik):
        raise ValueError(f"Plik {plik} nie istnieje.")

    with open(plik, "r", encoding="utf-8") as f:
        slowka = [linia.strip() for linia in f if linia.strip()]

    return slowka


# Testowa funkcja do uruchomienia bezpośrednio
if __name__ == "__main__":
    # Przykład użycia - dowolne języki
    test_jezyki = ["pl", "de", "fr", "es"]  # Bez angielskiego
    test_kategoria = "careers"

    print("Test tłumaczenia...")
    tlumaczenia = pokaz_tlumaczenia_gui(test_jezyki, test_kategoria)

    # Wyświetl wyniki
    print("\n=== TŁUMACZENIA ===")
    for jezyk, slowa in tlumaczenia.items():
        print(f"\n{jezyk.upper()}:")
        for i, slowo in enumerate(slowa[:10]):  # Pierwsze 10 jako przykład
            print(f"  {i + 1}. {slowo}")
        if len(slowa) > 10:
            print(f"  ... i {len(slowa) - 10} więcej")
