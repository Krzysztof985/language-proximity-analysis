"""
Main entry point for Language Proximity Analysis
Can run via CLI or GUI interface.
"""

import os
import sys
import argparse


from src.analysis_backend import run_levenshtein_analysis, run_embedding_analysis
from src.logger.logging_config import setup_logger

# Set up logger
logger = setup_logger(__name__, "main.log")

# Config
BASE_LANGUAGE = "en"
languages = ["fi", "pt", "pl", "es", "en", "fr", "it", "nl", "sv", "sl"]
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(BASE_DIR, "../data")
results_dir = os.path.join(BASE_DIR, "../results")


def main(method: str = "levenshtein", selected_languages: list = None) -> None:
    """
    Main function for language proximity analysis

    Args:
        method: Comparison method ('levenshtein' or 'embedding')
        selected_languages: List of language codes to analyze (if None, uses default)
    """
    # Use provided languages or default
    langs_to_use = selected_languages if selected_languages else languages

    print(f"Starting Language Proximity Analysis using {method.upper()} method")
    print(f"Languages: {', '.join(langs_to_use)}")
    print(f"Data directory: {data_dir}")
    print(f"Results directory: {results_dir}")
    print("-" * 60)

    try:
        if method == "embedding":
            run_embedding_analysis(
                languages=langs_to_use,
                data_dir=data_dir,
                results_dir=results_dir,
                base_language=BASE_LANGUAGE
            )
        elif method == "levenshtein":
            run_levenshtein_analysis(
                languages=langs_to_use,
                data_dir=data_dir,
                results_dir=results_dir,
                base_language=BASE_LANGUAGE
            )
        else:
            raise ValueError(f"Unknown method: {method}. Use 'levenshtein' or 'embedding'")

    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        print(f"\nAnalysis failed: {e}")
        raise


def launch_gui():
    """Launch the GUI interface"""
    try:
        import tkinter as tk
        from tkinter import ttk, messagebox, scrolledtext
        import threading
    except ImportError:
        print("Error: tkinter is not available. Please run in CLI mode.")
        sys.exit(1)

    class AnalysisGUI:
        def __init__(self, root):
            self.root = root
            self.root.title("Language Proximity Analysis")
            self.root.geometry("800x700")

            # Available languages
            self.available_languages = [
                ("English", "en"), ("Finnish", "fi"), ("Portuguese", "pt"),
                ("Polish", "pl"), ("Spanish", "es"), ("French", "fr"),
                ("Italian", "it"), ("Dutch", "nl"), ("Swedish", "sv"),
                ("Slovenian", "sl"), ("German", "de"), ("Slovak", "sk"),
                ("Czech", "cs"), ("Danish", "da"), ("Norwegian", "no")
            ]

            self.language_vars = []
            self.analysis_running = False

            self._create_widgets()

        def _create_widgets(self):
            # Title
            title_frame = ttk.Frame(self.root)
            title_frame.pack(pady=20)

            ttk.Label(
                title_frame,
                text="Language Proximity Analysis",
                font=("Arial", 16, "bold")
            ).pack()

            ttk.Label(
                title_frame,
                text="Select languages and analysis method",
                font=("Arial", 10)
            ).pack()

            # Language selection
            lang_frame = ttk.LabelFrame(self.root, text="Select Languages (minimum 2)", padding=10)
            lang_frame.pack(pady=10, padx=20, fill="both", expand=True)

            button_frame = ttk.Frame(lang_frame)
            button_frame.pack(pady=5)

            ttk.Button(button_frame, text="Select All", command=self._select_all).pack(side=tk.LEFT, padx=5)
            ttk.Button(button_frame, text="Deselect All", command=self._deselect_all).pack(side=tk.LEFT, padx=5)
            ttk.Button(button_frame, text="Select Default", command=self._select_default).pack(side=tk.LEFT, padx=5)

            checkbox_container = ttk.Frame(lang_frame)
            checkbox_container.pack(pady=10, fill="both", expand=True)

            for i, (name, code) in enumerate(self.available_languages):
                var = tk.IntVar()
                if code in languages:  # Pre-select default languages
                    var.set(1)

                chk = ttk.Checkbutton(
                    checkbox_container,
                    text=f"{name} ({code})",
                    variable=var
                )
                chk.grid(row=i // 3, column=i % 3, sticky="w", padx=10, pady=3)
                self.language_vars.append((code, var))

            # Method selection
            method_frame = ttk.LabelFrame(self.root, text="Analysis Method", padding=10)
            method_frame.pack(pady=10, padx=20, fill="x")

            self.method_var = tk.StringVar(value="levenshtein")

            ttk.Radiobutton(
                method_frame,
                text="Levenshtein Distance (Fast, character-based)",
                variable=self.method_var,
                value="levenshtein"
            ).pack(anchor="w", pady=5)

            ttk.Radiobutton(
                method_frame,
                text="Embedding-based (Advanced, phoneme-based)",
                variable=self.method_var,
                value="embedding"
            ).pack(anchor="w", pady=5)

            # Control buttons
            control_frame = ttk.Frame(self.root)
            control_frame.pack(pady=10, padx=20, fill="x")

            self.start_button = ttk.Button(
                control_frame,
                text="Start Analysis",
                command=self._start_analysis
            )
            self.start_button.pack(side=tk.LEFT, padx=5)

            ttk.Button(
                control_frame,
                text="Exit",
                command=self.root.quit
            ).pack(side=tk.RIGHT, padx=5)

            # Status
            status_frame = ttk.LabelFrame(self.root, text="Status", padding=10)
            status_frame.pack(pady=10, padx=20, fill="both", expand=True)

            self.status_text = scrolledtext.ScrolledText(
                status_frame,
                height=8,
                state="disabled",
                wrap=tk.WORD
            )
            self.status_text.pack(fill="both", expand=True)

            self._log("Ready. Select languages and click 'Start Analysis'.")

        def _select_all(self):
            for _, var in self.language_vars:
                var.set(1)

        def _deselect_all(self):
            for _, var in self.language_vars:
                var.set(0)

        def _select_default(self):
            for code, var in self.language_vars:
                var.set(1 if code in languages else 0)

        def _log(self, message):
            self.status_text.config(state="normal")
            self.status_text.insert(tk.END, message + "\n")
            self.status_text.see(tk.END)
            self.status_text.config(state="disabled")
            self.root.update()

        def _start_analysis(self):
            selected = [code for code, var in self.language_vars if var.get() == 1]

            if len(selected) < 2:
                messagebox.showwarning("Invalid Selection", "Please select at least 2 languages.")
                return

            if BASE_LANGUAGE not in selected:
                messagebox.showwarning("Missing Base Language", f"English ({BASE_LANGUAGE}) must be selected.")
                return

            method = self.method_var.get()

            self.start_button.config(state="disabled")
            self._log("=" * 50)
            self._log(f"Starting {method.upper()} analysis...")
            self._log(f"Languages: {', '.join(selected)}")
            self._log("=" * 50)

            threading.Thread(
                target=self._run_analysis,
                args=(selected, method),
                daemon=True
            ).start()

        def _run_analysis(self, selected_langs, method):
            try:
                main(method=method, selected_languages=selected_langs)
                self._log("=" * 50)
                self._log("✓ Analysis completed!")
                self._log(f"Results in: {results_dir}")
                self._log("=" * 50)
                messagebox.showinfo("Success", f"Analysis completed!\n\nResults: {results_dir}")
            except Exception as e:
                self._log(f"✗ Error: {str(e)}")
                messagebox.showerror("Error", f"Analysis failed:\n{str(e)}")
            finally:
                self.start_button.config(state="normal")

    root = tk.Tk()
    app = AnalysisGUI(root)
    root.mainloop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Language Proximity Analysis - Compare language similarity using different methods'
    )
    parser.add_argument(
        '--method',
        choices=['levenshtein', 'embedding'],
        default='levenshtein',
        help='Comparison method: levenshtein (default, fast) or embedding (advanced, slower)'
    )
    parser.add_argument(
        '--languages',
        nargs='+',
        default=None,
        help=f'Languages to analyze (default: {" ".join(languages)})'
    )
    parser.add_argument(
        '--gui',
        action='store_true',
        help='Launch GUI interface instead of CLI'
    )

    args = parser.parse_args()

    # Launch GUI if requested
    if args.gui:
        launch_gui()
    else:
        # Override languages if provided
        selected_langs = args.languages if args.languages else languages

        if args.languages:
            print(f"Using custom languages: {', '.join(selected_langs)}")

        try:
            main(method=args.method, selected_languages=selected_langs)
            print(f"Analysis completed successfully using {args.method.upper()} method!")
        except KeyboardInterrupt:
            print("Analysis interrupted by user")
            sys.exit(1)
        except Exception as e:
            print(f"Critical error: {e}")
            sys.exit(1)
