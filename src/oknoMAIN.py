import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import os
import csv
from pathlib import Path

from main import uruchom_analize


# =========================
# ŚCIEŻKI
# =========================

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
TRANSLATIONS_DIR = RESULTS_DIR / "translations"
GRAPHS_DIR = RESULTS_DIR / "graphs_levenshtein"


# =========================
# FUNKCJE
# =========================

def pobierz_wybrane_jezyki():
    return [kod for (_, kod, var) in lista_checkboxow if var.get() == 1]


def start_analizy():
    try:
        jezyki = pobierz_wybrane_jezyki()

        if len(jezyki) < 2:
            messagebox.showwarning("Błąd", "Zaznacz co najmniej dwa języki.")
            return

        uruchom_analize(jezyki, wybrana_metoda.get())

        messagebox.showinfo("Sukces", "Analiza zakończona pomyślnie!")

        pokaz_tlumaczenia()
        pokaz_grafy()

    except Exception as e:
        messagebox.showerror("Błąd", str(e))


def pokaz_tlumaczenia():
    for w in ramka_tlumaczenia.winfo_children():
        w.destroy()

    csv_files = list(TRANSLATIONS_DIR.glob("*.csv"))
    if not csv_files:
        messagebox.showwarning("Brak danych", "Brak plików CSV z tłumaczeniami.")
        return

    with open(csv_files[0], encoding="utf-8") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames

        for c, h in enumerate(headers):
            ttk.Label(
                ramka_tlumaczenia, text=h,
                borderwidth=1, relief="solid", padding=4
            ).grid(row=0, column=c, sticky="nsew")

        for r, row in enumerate(reader, start=1):
            for c, h in enumerate(headers):
                ttk.Label(
                    ramka_tlumaczenia, text=row[h],
                    borderwidth=1, relief="solid", padding=4
                ).grid(row=r, column=c, sticky="nsew")


def pokaz_grafy():
    for w in ramka_wykres.winfo_children():
        w.destroy()

    if not GRAPHS_DIR.exists():
        messagebox.showwarning("Brak grafów", "Folder graphs_levenshtein nie istnieje.")
        return

    for plik in GRAPHS_DIR.glob("*.png"):
        img = Image.open(plik)
        img = img.resize((600, 450))
        photo = ImageTk.PhotoImage(img)

        lbl = ttk.Label(ramka_wykres, image=photo)
        lbl.image = photo
        lbl.pack(pady=10)


# =========================
# GUI
# =========================

root = tk.Tk()
root.title("Language Proximity Analysis")
root.geometry("1300x900")

wybrana_metoda = tk.StringVar(value="levenshtein")


# --- JĘZYKI ---
ramka_jezyki = ttk.LabelFrame(root, text="Wybierz języki")
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
    ("Słoweński", "sl")
]

lista_checkboxow = []
frame = ttk.Frame(ramka_jezyki)
frame.pack()

for i, (nazwa, kod) in enumerate(jezyki):
    var = tk.IntVar()
    chk = ttk.Checkbutton(frame, text=nazwa, variable=var)
    chk.grid(row=i // 5, column=i % 5, padx=5, pady=5)
    lista_checkboxow.append((nazwa, kod, var))


# --- METODA ---
ramka_metoda = ttk.LabelFrame(root, text="Metoda")
ramka_metoda.pack(fill="x", padx=10, pady=10)

ttk.Radiobutton(ramka_metoda, text="Levenshtein",
                variable=wybrana_metoda, value="levenshtein").pack(side=tk.LEFT, padx=10)

ttk.Radiobutton(ramka_metoda, text="Embedding",
                variable=wybrana_metoda, value="embedding").pack(side=tk.LEFT, padx=10)


# --- PRZYCISK ---
ttk.Button(root, text="Uruchom analizę", command=start_analizy).pack(pady=15)


# --- WYNIKI ---
ramka_tlumaczenia = ttk.LabelFrame(root, text="Tłumaczenia")
ramka_tlumaczenia.pack(fill="both", expand=True, padx=10, pady=10)

ramka_wykres = ttk.LabelFrame(root, text="Grafy podobieństwa")
ramka_wykres.pack(fill="both", expand=True, padx=10, pady=10)

root.mainloop()