import argparse
import os
import requests
import sys


from src.logger.logging_config import setup_logger

# Set up logger for this module
logger = setup_logger(__name__, 'phoneme_extractor.log')

# Mapping from language code to WikiPron filename
# Using "broad" transcription (phonemic) by default
# This map includes all major Latin alphabet languages available in WikiPron
LANGUAGE_MAP = {
    # Germanic
    "en": "eng_latn_us_broad.tsv",
    "de": "deu_latn_broad.tsv",
    "nl": "nld_latn_broad.tsv",
    "sv": "swe_latn_broad.tsv",
    "da": "dan_latn_broad.tsv",
    "no": "nob_latn_broad.tsv",     # Bokmål = default
    "is": "isl_latn_broad.tsv",
    "fo": "fao_latn_broad.tsv",
    "lb": "ltz_latn_broad.tsv",
    "af": "afr_latn_broad.tsv",
    "fy": "fry_latn_broad.tsv",
    "yi": "yid_hebr_broad.tsv",     # Yiddish uses Hebrew script

    # Romance
    "fr": "fra_latn_broad.tsv",
    "es": "spa_latn_la_broad.tsv",  # Latin America = most common
    "it": "ita_latn_broad.tsv",
    "pt": "por_latn_bz_broad.tsv",  # Brazilian Portuguese = most common
    "ro": "ron_latn_broad.tsv",
    "ca": "cat_latn_broad.tsv",
    "gl": "glg_latn_broad.tsv",
    "oc": "oci_latn_broad.tsv",
    "la": "lat_latn_clas_broad.tsv",  # classical as default
    "sc": "srd_latn_broad.tsv",
    "co": "cos_latn_broad.tsv",
    "wa": "wln_latn_broad.tsv",
    "rm": "roh_latn_broad.tsv",
    "ast": "ast_latn_broad.tsv",

    # Slavic (Latin)
    "pl": "pol_latn_broad.tsv",
    "cs": "ces_latn_broad.tsv",
    "sk": "slk_latn_broad.tsv",
    "sl": "slv_latn_broad.tsv",
    "hr": "hrv_latn_broad.tsv",
    "bs": "bos_latn_broad.tsv",
    "sh": "hbs_latn_broad.tsv",

    # Celtic
    "ga": "gle_latn_broad.tsv",
    "gd": "gla_latn_broad.tsv",
    "cy": "cym_latn_sw_broad.tsv",     # South Wales = larger set
    "br": "bre_latn_broad.tsv",
    "gv": "glv_latn_broad.tsv",
    "kw": "cor_latn_broad.tsv",

    # Baltic
    "lt": "lit_latn_broad.tsv",
    "lv": "lav_latn_narrow.tsv",       # no broad LATN in table

    # Uralic
    "fi": "fin_latn_broad.tsv",
    "et": "est_latn_broad.tsv",
    "hu": "hun_latn_broad.tsv",        # NO broad available—closest is narrow

    # Turkic (Latin)
    "tr": "tur_latn_broad.tsv",
    "az": "aze_latn_broad.tsv",
    "uz": "uzb_latn_broad.tsv",
    "tk": "tuk_latn_broad.tsv",
    "tt": "tat_cyrl_broad.tsv",        # no LATN version → Cyrillic

    # Albanian
    "sq": "sqi_latn_broad.tsv",

    # Basque
    "eu": "eus_latn_broad.tsv",

    # Maltese
    "mt": "mlt_latn_broad.tsv",

    # Austronesian
    "tl": "tgl_latn_broad.tsv",
    "id": "ind_latn_broad.tsv",
    "ms": "msa_latn_broad.tsv",
    "jv": "jav_java_broad.tsv",  # Javanese script, LATN not available
    "su": "sun_latn_broad.tsv", # doesn't exist; nearest is Makassar etc.
    "mg": "mlg_latn_broad.tsv",
    "ceb": "ceb_latn_broad.tsv",
    "haw": "haw_latn_broad.tsv",
    "mi": "mri_latn_broad.tsv",
    "sm": "smo_latn_broad.tsv",
    "to": "ton_latn_broad.tsv",

    # Vietnamese
    "vi": "vie_latn_saigon_narrow.tsv",  # no broad LATN exists

    # African
    "sw": "swa_latn_broad.tsv",
    "zu": "zul_latn_broad.tsv",
    "xh": "xho_latn_narrow.tsv",
    "sn": "sna_latn_broad.tsv",
    "st": "sot_latn_broad.tsv",
    "tn": "tsn_latn_broad.tsv",
    "yo": "yor_latn_broad.tsv",
    "ig": "ibo_latn_broad.tsv",
    "ha": "hau_latn_broad.tsv",
    "rw": "kin_latn_broad.tsv",
    "lg": "lug_latn_broad.tsv",
    "ny": "nya_latn_broad.tsv",

    # Esperanto
    "eo": "epo_latn_broad.tsv",

    # Cyrillic-only languages
    "ru": "rus_cyrl_narrow.tsv",
    "uk": "ukr_cyrl_narrow.tsv",
    "bg": "bul_cyrl_narrow.tsv",
    "sr": "srp_cyrl_narrow.tsv",
    "mk": "mkd_cyrl_narrow.tsv",
    "be": "bel_cyrl_narrow.tsv",
}

BASE_URL = "https://raw.githubusercontent.com/CUNY-CL/wikipron/master/data/scrape/tsv/"

def download_phonemes(lang, output_dir):
    if lang not in LANGUAGE_MAP:
        logger.warning(f"Language '{lang}' not supported in current map. Please add it to LANGUAGE_MAP in phoneme_extractor.py")
        return

    filename = LANGUAGE_MAP[lang]
    url = BASE_URL + filename
    
    logger.info(f"Downloading phonemes for {lang} from {url}...")
    
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "phonemes.txt")
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        
        # WikiPron TSV format: word <tab> ipa
        # We will save it as is, or maybe filter/clean if needed.
        # Let's save it directly but ensure it's valid.
        
        lines = response.text.strip().split('\n')
        count = 0
        with open(output_path, 'w', encoding='utf-8') as f:
            for line in lines:
                parts = line.split('\t')
                if len(parts) >= 2:
                    word = parts[0].lstrip("'")  # Remove leading apostrophes
                    ipa = parts[1]
                    # Basic filtering: skip if word contains numbers (consistent with previous task)
                    if not any(char.isdigit() for char in word) and word:  # Also check word is not empty
                        f.write(f"{word}\t{ipa}\n")
                        count += 1
        
        logger.info(f"Saved {count} phoneme entries to {output_path}")
        
    except Exception as e:  
        logger.error(f"Error downloading/processing {lang}: {e}")

def download_languages():
    parser = argparse.ArgumentParser(description="Download phoneme dictionaries from WikiPron.")
    parser.add_argument("languages", nargs="+", help="List of language codes (e.g., en pl de)")
    
    args = parser.parse_args()
    
    base_data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data")
    
    for lang in args.languages:
        logger.info(f"Processing language: {lang}")
        lang_dir = os.path.join(base_data_dir, lang)
        download_phonemes(lang, lang_dir)

if __name__ == "__main__":
    download_languages()
