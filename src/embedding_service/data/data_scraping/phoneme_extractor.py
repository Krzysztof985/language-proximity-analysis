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
    # Germanic languages
    "en": "eng_latn_us_broad.tsv",    # English (US)
    "de": "deu_latn_broad.tsv",       # German
    "nl": "nld_latn_broad.tsv",       # Dutch
    "sv": "swe_latn_broad.tsv",       # Swedish
    "da": "dan_latn_broad.tsv",       # Danish
    "no": "nor_latn_broad.tsv",       # Norwegian
    "is": "isl_latn_broad.tsv",       # Icelandic
    "fo": "fao_latn_broad.tsv",       # Faroese
    "lb": "ltz_latn_broad.tsv",       # Luxembourgish
    "af": "afr_latn_broad.tsv",       # Afrikaans
    "fy": "fry_latn_broad.tsv",       # Frisian
    "yi": "yid_latn_broad.tsv",       # Yiddish
    
    # Romance languages
    "fr": "fra_latn_broad.tsv",       # French
    "es": "spa_latn_broad.tsv",       # Spanish
    "it": "ita_latn_broad.tsv",       # Italian
    "pt": "por_latn_broad.tsv",       # Portuguese
    "ro": "ron_latn_broad.tsv",       # Romanian
    "ca": "cat_latn_broad.tsv",       # Catalan
    "gl": "glg_latn_broad.tsv",       # Galician
    "oc": "oci_latn_broad.tsv",       # Occitan
    "la": "lat_latn_broad.tsv",       # Latin
    "sc": "srd_latn_broad.tsv",       # Sardinian
    "co": "cos_latn_broad.tsv",       # Corsican
    "wa": "wln_latn_broad.tsv",       # Walloon
    "rm": "roh_latn_broad.tsv",       # Romansh
    "ast": "ast_latn_broad.tsv",      # Asturian
    
    # Slavic languages (Latin script)
    "pl": "pol_latn_broad.tsv",       # Polish
    "cs": "ces_latn_broad.tsv",       # Czech
    "sk": "slk_latn_broad.tsv",       # Slovak
    "sl": "slv_latn_broad.tsv",       # Slovenian
    "hr": "hrv_latn_broad.tsv",       # Croatian
    "bs": "bos_latn_broad.tsv",       # Bosnian
    "sh": "hbs_latn_broad.tsv",       # Serbo-Croatian
    
    # Celtic languages
    "ga": "gle_latn_broad.tsv",       # Irish
    "gd": "gla_latn_broad.tsv",       # Scottish Gaelic
    "cy": "cym_latn_broad.tsv",       # Welsh
    "br": "bre_latn_broad.tsv",       # Breton
    "gv": "glv_latn_broad.tsv",       # Manx
    "kw": "cor_latn_broad.tsv",       # Cornish
    
    # Baltic languages
    "lt": "lit_latn_broad.tsv",       # Lithuanian
    "lv": "lav_latn_broad.tsv",       # Latvian
    
    # Uralic languages
    "fi": "fin_latn_broad.tsv",       # Finnish
    "et": "est_latn_broad.tsv",       # Estonian
    "hu": "hun_latn_broad.tsv",       # Hungarian
    
    # Turkic languages (Latin script)
    "tr": "tur_latn_broad.tsv",       # Turkish
    "az": "aze_latn_broad.tsv",       # Azerbaijani
    "uz": "uzb_latn_broad.tsv",       # Uzbek
    "tk": "tuk_latn_broad.tsv",       # Turkmen
    "tt": "tat_latn_broad.tsv",       # Tatar
    
    # Albanian
    "sq": "sqi_latn_broad.tsv",       # Albanian
    
    # Basque
    "eu": "eus_latn_broad.tsv",       # Basque
    
    # Maltese
    "mt": "mlt_latn_broad.tsv",       # Maltese
    
    # Filipino/Austronesian languages
    "tl": "tgl_latn_broad.tsv",       # Tagalog
    "id": "ind_latn_broad.tsv",       # Indonesian
    "ms": "msa_latn_broad.tsv",       # Malay
    "jv": "jav_latn_broad.tsv",       # Javanese
    "su": "sun_latn_broad.tsv",       # Sundanese
    "mg": "mlg_latn_broad.tsv",       # Malagasy
    "ceb": "ceb_latn_broad.tsv",      # Cebuano
    "haw": "haw_latn_broad.tsv",      # Hawaiian
    "mi": "mri_latn_broad.tsv",       # Maori
    "sm": "smo_latn_broad.tsv",       # Samoan
    "to": "ton_latn_broad.tsv",       # Tongan
    
    # Vietnamese
    "vi": "vie_latn_broad.tsv",       # Vietnamese
    
    # Swahili and African languages
    "sw": "swa_latn_broad.tsv",       # Swahili
    "zu": "zul_latn_broad.tsv",       # Zulu
    "xh": "xho_latn_broad.tsv",       # Xhosa
    "sn": "sna_latn_broad.tsv",       # Shona
    "st": "sot_latn_broad.tsv",       # Sotho
    "tn": "tsn_latn_broad.tsv",       # Tswana
    "yo": "yor_latn_broad.tsv",       # Yoruba
    "ig": "ibo_latn_broad.tsv",       # Igbo
    "ha": "hau_latn_broad.tsv",       # Hausa
    "rw": "kin_latn_broad.tsv",       # Kinyarwanda
    "lg": "lug_latn_broad.tsv",       # Luganda
    "ny": "nya_latn_broad.tsv",       # Chichewa
    
    # Esperanto
    "eo": "epo_latn_broad.tsv",       # Esperanto
    
    # Cyrillic script languages (for reference, not Latin)
    "ru": "rus_cyrl_broad.tsv",       # Russian
    "uk": "ukr_cyrl_broad.tsv",       # Ukrainian
    "bg": "bul_cyrl_broad.tsv",       # Bulgarian
    "sr": "srp_cyrl_broad.tsv",       # Serbian
    "mk": "mkd_cyrl_broad.tsv",       # Macedonian
    "be": "bel_cyrl_broad.tsv",       # Belarusian
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
