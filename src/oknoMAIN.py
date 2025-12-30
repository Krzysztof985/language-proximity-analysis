import tkinter as tk
from tkinter import ttk, messagebox
from deep_translator import GoogleTranslator
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"


def wczytaj_slowka(plik):
    """Wczytuje słówka z pliku tekstowego do listy."""
    try:
        with open(plik, "r", encoding="utf-8") as f:
            slowka = [linia.strip() for linia in f if linia.strip()]
        return slowka
    except FileNotFoundError:
        messagebox.showerror("Błąd", f"Nie znaleziono pliku: {plik}")
        return []

def aktualizuj_slowka(*args):
    """Aktualizuje listę słówek po zmianie kategorii."""
    kategoria = wybrana_kategoria.get()
    plik = kategorie_pliki.get(kategoria, None)

    if not plik:
        lista_slowek = []
    else:
        lista_slowek = wczytaj_slowka(plik)

    menu_slowek['menu'].delete(0, 'end')
    menu_slowek['menu'].add_command(label="Wybierz słowo",
                                    command=lambda: wybrane_slowko.set("Wybierz słowo"))

    if lista_slowek:
        for slowo in lista_slowek:
            menu_slowek['menu'].add_command(
                label=slowo,
                command=lambda v=slowo: wybrane_slowko.set(v)
            )

    wybrane_slowko.set("Wybierz słowo")
    etykieta_wyboru.config(text="Wybierz słowo z listy")

def on_select(event=None):
    """Tworzy tabelę: górny wiersz = języki, dolny = tłumaczenia."""
    wybrane = wybrane_slowko.get()

    # wyczyszczenie tabeli
    for widget in ramka_tabelka.winfo_children():
        widget.destroy()

    if wybrane == "Wybierz słowo":
        etykieta_wyboru.config(text="Nie wybrałeś jeszcze słowa.")
        return

    # Pobranie wybranych języków
    zaznaczone = [(nazwa, kod) for (nazwa, kod, var) in lista_checkboxow if var.get() == 1]

    if not zaznaczone:
        messagebox.showwarning("Brak języków", "Zaznacz przynajmniej jeden język!")
        return

    etykieta_wyboru.config(text=f"Tłumaczenia dla: {wybrane}")

    # --- TWORZENIE GÓRNEGO WIERSZA (JĘZYKI) ---
    for i, (jezyk, kod) in enumerate(zaznaczone):
        naglowek = ttk.Label(
            ramka_tabelka,
            text=jezyk.capitalize(),
            borderwidth=1,
            relief="solid",
            padding=5
        )
        naglowek.grid(row=0, column=i, sticky="nsew")

    # --- DRUGI WIERSZ (TŁUMACZENIA) ---
    for i, (jezyk, kod) in enumerate(zaznaczone):
        try:
            if kod == "en":
                tlumaczenie = wybrane
            else:
                tlumaczenie = GoogleTranslator(source="auto", target=kod).translate(wybrane)
        except:
            tlumaczenie = "(błąd)"

        komorka = ttk.Label(
            ramka_tabelka,
            text=tlumaczenie,
            borderwidth=1,
            relief="solid",
            padding=5
        )
        komorka.grid(row=1, column=i, sticky="nsew")

    # rozciąganie kolumn
    for i in range(len(zaznaczone)):
        ramka_tabelka.columnconfigure(i, weight=1)


# --- Okno ---
root = tk.Tk()
root.title("Lista słówek z tłumaczeniami")
root.geometry("1250x600")

# --- Kategorie (NAPRAWIONE ŚCIEŻKI) ---
kategorie_pliki = {
    "Zawody": DATA_DIR / "careers.txt",
    "Miejsca": DATA_DIR / "placesInCity.txt",
    "Kraje": DATA_DIR / "countries.txt",
    "Sporty": DATA_DIR / "sports_and_activities.txt"
}


# --- Zmienne ---
wybrana_kategoria = tk.StringVar(value="Wybierz kategorię")
wybrane_slowko = tk.StringVar(value="Wybierz słowo")

# --- Ramka menu ---
ramka_menu = ttk.Frame(root)
ramka_menu.pack(pady=20)

ttk.Label(ramka_menu, text="Kategoria:").pack(side=tk.LEFT, padx=5)
menu_kategorii = ttk.OptionMenu(ramka_menu, wybrana_kategoria,
                                *["Wybierz kategorię"] + list(kategorie_pliki.keys()))
menu_kategorii.pack(side=tk.LEFT, padx=10)

ttk.Label(ramka_menu, text="Słówko:").pack(side=tk.LEFT, padx=5)
menu_slowek = ttk.OptionMenu(ramka_menu, wybrane_slowko, wybrane_slowko.get())
menu_slowek.pack(side=tk.LEFT, padx=10)

# --- Lista języków (checkboxy) ---
ramka_jezyki = ttk.Frame(root)
ramka_jezyki.pack(pady=10)

ttk.Label(ramka_jezyki, text="Wybierz języki do tłumaczenia:").pack()

jezyki = [
    ("angielski", "en"), ("niemiecki", "de"), ("francuski", "fr"),
    ("polski", "pl"), ("włoski", "it"), ("hiszpański", "es"),
    ("portugalski", "pt"), ("słowacki", "sk"), ("czeski", "cs"),
    ("holenderski", "nl"), ("szwedzki", "sv"), ("duński", "da"),
    ("norweski", "no"), ("słoweński", "sl")
]

lista_checkboxow = []
frame_row = ttk.Frame(ramka_jezyki)
frame_row.pack()

for i, (naz, kod) in enumerate(jezyki):
    var = tk.IntVar()
    chk = ttk.Checkbutton(frame_row, text=naz.capitalize(), variable=var)
    chk.grid(row=i // 7, column=i % 7, padx=5, pady=3)
    lista_checkboxow.append((naz, kod, var))

# --- Etykieta ---
etykieta_wyboru = ttk.Label(root, text="Wybierz kategorię, słowo i języki.")
etykieta_wyboru.pack(pady=10)

# --- Przycisk ---
ttk.Button(root, text="Pokaż tłumaczenia", command=on_select).pack(pady=10)

# --- Ramka tabeli ---
ramka_tabelka = ttk.Frame(root)
ramka_tabelka.pack(pady=10, fill="both", expand=True)

# --- Trace kategorii ---
wybrana_kategoria.trace("w", aktualizuj_slowka)

root.mainloop()
