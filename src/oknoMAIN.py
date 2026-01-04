import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import os
import csv
from pathlib import Path

from main import uruchom_analize, pokaz_tlumaczenia_gui, get_available_categories, get_words_from_category

# =========================
# ŚCIEŻKI
# =========================

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = Path(__file__).parent.parent / "results"
TRANSLATIONS_DIR = RESULTS_DIR / "translations"
GRAPHS_LEVENSHTEIN_DIR = RESULTS_DIR / "graphs_levenshtein"
GRAPHS_EMBEDDING_DIR = RESULTS_DIR / "graphs_embedding"


# =========================
# FUNKCJE
# =========================

def pobierz_wybrane_jezyki():
    return [kod for (_, kod, var) in lista_checkboxow if var.get() == 1]


def start_analizy():
    try:
        jezyki = pobierz_wybrane_jezyki()

        if len(jezyki) < 2:
            messagebox.showwarning("Błąd", "Wybierz co najmniej dwa języki do analizy.")
            return

        metoda = wybrana_metoda.get()
        uruchom_analize(jezyki, metoda)

        messagebox.showinfo("Sukces", "Analiza zakończona pomyślnie!")

        pokaz_grafy()

    except Exception as e:
        messagebox.showerror("Błąd", str(e))


def tlumacz_i_pokaz():
    """Tłumaczy słowa dla zaznaczonych języków i wyświetla wyniki."""
    try:
        jezyki = pobierz_wybrane_jezyki()

        if not jezyki:
            messagebox.showwarning("Błąd", "Zaznacz przynajmniej jeden język.")
            return

        kategoria = wybrana_kategoria.get()
        if kategoria == "Wybierz kategorię":
            kategoria = None

        # Wyświetl komunikat o rozpoczynaniu tłumaczenia
        status_label.config(text="Trwa tłumaczenie...")
        root.update()

        # Wywołaj funkcję tłumaczącą
        tlumaczenia = pokaz_tlumaczenia_gui(jezyki, kategoria)

        # Wyświetl wyniki w tabeli
        pokaz_tlumaczenia_w_tabeli(tlumaczenia)

        status_label.config(text=f"Przetłumaczono na {len(jezyki)} języków")

    except Exception as e:
        messagebox.showerror("Błąd tłumaczenia", str(e))
        status_label.config(text="Błąd tłumaczenia")


def pokaz_tlumaczenia_w_tabeli(tlumaczenia: dict):
    """Wyświetla tłumaczenia w tabeli."""
    # Usuń wszystkie istniejące elementy z ramki tłumaczeń
    for w in ramka_tlumaczenia.winfo_children():
        w.destroy()

    if not tlumaczenia:
        ttk.Label(ramka_tlumaczenia, text="Brak danych do wyświetlenia").pack()
        return

    # Pobierz klucze (języki) i ustal liczbę wierszy
    jezyki = list(tlumaczenia.keys())
    slowka_listy = list(tlumaczenia.values())
    liczba_wierszy = len(slowka_listy[0]) if slowka_listy else 0

    # Utwórz przewijalny obszar dla tabeli
    canvas = tk.Canvas(ramka_tlumaczenia)
    scrollbar_y = ttk.Scrollbar(ramka_tlumaczenia, orient="vertical", command=canvas.yview)
    scrollbar_x = ttk.Scrollbar(ramka_tlumaczenia, orient="horizontal", command=canvas.xview)

    scrollable_frame = ttk.Frame(canvas)

    scrollable_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
    )

    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar_y.set, xscrollcommand=scrollbar_x.set)

    # Nagłówki (języki)
    for c, jezyk in enumerate(jezyki):
        ttk.Label(
            scrollable_frame, text=jezyk.upper(),
            borderwidth=2, relief="solid", padding=8,
            font=('Arial', 11, 'bold'), background="#e0e0ff"
        ).grid(row=0, column=c, sticky="nsew")

    # Dane (tłumaczenia)
    for r in range(liczba_wierszy):
        for c, jezyk in enumerate(jezyki):
            slowo = tlumaczenia[jezyk][r] if r < len(tlumaczenia[jezyk]) else ""
            bg_color = "#ffffff" if r % 2 == 0 else "#f0f0f0"
            ttk.Label(
                scrollable_frame, text=slowo,
                borderwidth=1, relief="solid", padding=6,
                font=('Arial', 10), background=bg_color
            ).grid(row=r + 1, column=c, sticky="nsew")

    # Konfiguruj wagi kolumn dla równomiernego rozłożenia
    for c in range(len(jezyki)):
        scrollable_frame.grid_columnconfigure(c, weight=1, uniform="col")

    # Ustawienie grid dla elementów przewijalnych
    canvas.grid(row=0, column=0, sticky="nsew")
    scrollbar_y.grid(row=0, column=1, sticky="ns")
    scrollbar_x.grid(row=1, column=0, sticky="ew")

    # Konfiguracja grid dla ramki tłumaczeń
    ramka_tlumaczenia.grid_rowconfigure(0, weight=1)
    ramka_tlumaczenia.grid_columnconfigure(0, weight=1)

    # Dodaj informację o liczbie wierszy
    info_label = ttk.Label(
        ramka_tlumaczenia,
        text=f"Wyświetlono {liczba_wierszy} słów w {len(jezyki)} językach",
        font=('Arial', 9)
    )
    info_label.grid(row=2, column=0, pady=5, sticky="w")


def pokaz_grafy():
    """Wyświetla wygenerowane grafy."""
    for w in ramka_wykres.winfo_children():
        w.destroy()

    # Wybierz odpowiedni folder na podstawie metody
    metoda = wybrana_metoda.get()
    if metoda == "levenshtein":
        graphs_dir = GRAPHS_LEVENSHTEIN_DIR
    else:
        graphs_dir = GRAPHS_EMBEDDING_DIR

    # Sprawdź czy folder istnieje
    if not graphs_dir.exists():
        messagebox.showwarning("Brak grafów", f"Folder {graphs_dir.name} nie istnieje.\nUruchom najpierw analizę.")
        return

    # Znajdź wszystkie pliki PNG
    png_files = list(graphs_dir.glob("*.png"))
    if not png_files:
        messagebox.showwarning("Brak grafów", f"Brak plików PNG w folderze {graphs_dir.name}.")
        return

    # Utwórz przewijalną ramkę dla grafik
    canvas = tk.Canvas(ramka_wykres)
    scrollbar_y = ttk.Scrollbar(ramka_wykres, orient="vertical", command=canvas.yview)
    scrollable_frame = ttk.Frame(canvas)

    scrollable_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
    )

    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar_y.set)

    # Wyświetl każdy graf
    for i, plik in enumerate(png_files):
        try:
            # Załaduj i przeskaluj obraz
            img = Image.open(plik)
            # Zachowaj proporcje
            img.thumbnail((600, 450))
            photo = ImageTk.PhotoImage(img)

            # Etykieta z nazwą pliku
            nazwa_label = ttk.Label(
                scrollable_frame,
                text=f"{plik.stem}",
                font=('Arial', 10, 'bold')
            )
            nazwa_label.grid(row=i * 2, column=0, pady=(10, 0))

            # Etykieta z obrazem
            lbl = ttk.Label(scrollable_frame, image=photo)
            lbl.image = photo  # przechowaj referencję
            lbl.grid(row=i * 2 + 1, column=0, pady=(0, 20))

        except Exception as e:
            error_label = ttk.Label(
                scrollable_frame,
                text=f"Błąd ładowania {plik.name}: {str(e)}",
                foreground="red"
            )
            error_label.grid(row=i * 2, column=0, pady=10)

    # Pakuj elementy przewijalne
    canvas.grid(row=0, column=0, sticky="nsew")
    scrollbar_y.grid(row=0, column=1, sticky="ns")

    # Konfiguracja grid
    ramka_wykres.grid_rowconfigure(0, weight=1)
    ramka_wykres.grid_columnconfigure(0, weight=1)

    # Dodaj informację o liczbie grafik
    info_label = ttk.Label(
        ramka_wykres,
        text=f"Znaleziono {len(png_files)} grafik w {graphs_dir.name}",
        font=('Arial', 9)
    )
    info_label.grid(row=1, column=0, pady=5, sticky="w")


def aktualizuj_kategorie():
    """Aktualizuje listę kategorii w dropdown menu."""
    kategorie = get_available_categories()
    menu_kategorii['menu'].delete(0, 'end')

    menu_kategorii['menu'].add_command(
        label="Wybierz kategorię",
        command=lambda: wybrana_kategoria.set("Wybierz kategorię")
    )

    for kat in kategorie:
        menu_kategorii['menu'].add_command(
            label=kat,
            command=lambda v=kat: wybrana_kategoria.set(v)
        )


def wybierz_wszystkie():
    """Zaznacz wszystkie języki"""
    for _, _, var in lista_checkboxow:
        var.set(1)


def odznacz_wszystkie():
    """Odznacz wszystkie języki"""
    for _, _, var in lista_checkboxow:
        var.set(0)


# =========================
# GUI
# =========================

root = tk.Tk()
root.title("Language Proximity Analysis")
root.geometry("1300x900")

wybrana_metoda = tk.StringVar(value="levenshtein")
wybrana_kategoria = tk.StringVar(value="Wybierz kategorię")

# --- JĘZYKI ---
ramka_jezyki = ttk.LabelFrame(root, text="Wybierz języki (co najmniej 2 dla analizy)")
ramka_jezyki.pack(fill="x", padx=10, pady=10)

jezyki = [
    ("Angielski", "en"),
    ("Polski", "pl"),
    ("Niemiecki", "de"),
    ("Francuski", "fr"),
    ("Hiszpański", "es"),
    ("Włoski", "it"),
    ("Portugalski", "pt"),
    ("Holenderski", "nl"),
    ("Szwedzki", "sv"),
    ("Słoweński", "sl"),
    ("Fiński", "fi"),
    ("Duński", "da"),
    ("Norweski", "no"),
    ("Irlandzki", "ga"),
    ("Esperanto", "eo"),
    ("Baskijski", "eu"),
    ("Maltański", "mt"),
    ("Słowacki", "sk"),
    ("Walijski", "cy"),
    ("Szkocki gaelicki", "gd"),
    ("Bretoński", "br"),
    ("Litewski", "lt"),
    ("Łotewski", "lv"),
    ("Turecki", "tr"),
    ("Azerbejdżański", "az"),
    ("Uzbecki", "uz"),
    ("Turkmeński", "tk"),
    ("Suahili", "sw"),
    ("Zulu", "zu"),
    ("Joruba", "yo"),
    ("Hausa", "ha"),
    ("Njandża", "ny"),
    ("Rosyjski", "ru"),
    ("Ukraiński", "uk")
]

lista_checkboxow = []
frame_buttons = ttk.Frame(ramka_jezyki)
frame_buttons.pack(pady=5)

ttk.Button(frame_buttons, text="Zaznacz wszystkie", command=wybierz_wszystkie).pack(side=tk.LEFT, padx=5)
ttk.Button(frame_buttons, text="Odznacz wszystkie", command=odznacz_wszystkie).pack(side=tk.LEFT, padx=5)

frame_checkboxes = ttk.Frame(ramka_jezyki)
frame_checkboxes.pack()

for i, (nazwa, kod) in enumerate(jezyki):
    var = tk.IntVar()
    # USUNIĘTO: domyślne zaznaczenie angielskiego
    chk = ttk.Checkbutton(frame_checkboxes, text=f"{nazwa} ({kod})", variable=var)
    chk.grid(row=i // 5, column=i % 5, padx=5, pady=5)
    lista_checkboxow.append((nazwa, kod, var))

# --- KATEGORIA ---
ramka_kategoria = ttk.LabelFrame(root, text="Wybierz kategorię słów")
ramka_kategoria.pack(fill="x", padx=10, pady=10)

ttk.Label(ramka_kategoria, text="Kategoria:").pack(side=tk.LEFT, padx=5)
menu_kategorii = ttk.OptionMenu(ramka_kategoria, wybrana_kategoria, "Wybierz kategorię")
menu_kategorii.pack(side=tk.LEFT, padx=5)
ttk.Button(ramka_kategoria, text="Odśwież listę", command=aktualizuj_kategorie).pack(side=tk.LEFT, padx=5)

# --- METODA ---
ramka_metoda = ttk.LabelFrame(root, text="Metoda analizy")
ramka_metoda.pack(fill="x", padx=10, pady=10)

ttk.Radiobutton(ramka_metoda, text="Levenshtein (szybka, znakowa)",
                variable=wybrana_metoda, value="levenshtein").pack(side=tk.LEFT, padx=10)

ttk.Radiobutton(ramka_metoda, text="Embedding (zaawansowana, fonemowa)",
                variable=wybrana_metoda, value="embedding").pack(side=tk.LEFT, padx=10)

# --- PRZYCISKI ---
frame_przyciski = ttk.Frame(root)
frame_przyciski.pack(pady=15)

ttk.Button(frame_przyciski, text="Tłumacz słowa", command=tlumacz_i_pokaz,
           style="Accent.TButton").pack(side=tk.LEFT, padx=5)
ttk.Button(frame_przyciski, text="Uruchom analizę", command=start_analizy).pack(side=tk.LEFT, padx=5)
ttk.Button(frame_przyciski, text="Odśwież grafy", command=pokaz_grafy).pack(side=tk.LEFT, padx=5)

# Status bar
status_label = ttk.Label(root, text="Wybierz języki i kategorię", relief=tk.SUNKEN, anchor=tk.W)
status_label.pack(side=tk.BOTTOM, fill=tk.X)

# --- WYNIKI ---
notebook = ttk.Notebook(root)
notebook.pack(fill="both", expand=True, padx=10, pady=10)

# Karta Tłumaczenia
ramka_tlumaczenia = ttk.Frame(notebook)
notebook.add(ramka_tlumaczenia, text="Tłumaczenia")

# Karta Grafy
ramka_wykres = ttk.Frame(notebook)
notebook.add(ramka_wykres, text="Grafy podobieństwa")

# Styl dla przycisku akcentującego
style = ttk.Style()
style.configure("Accent.TButton", font=('Arial', 10, 'bold'))

# Załaduj listę kategorii przy starcie
aktualizuj_kategorie()

root.mainloop()