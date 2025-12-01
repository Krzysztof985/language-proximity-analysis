import os
import sys

# Add the project root to sys.path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

# Add the data_scraping directory to sys.path
data_scraping_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data_scraping')
sys.path.insert(0, data_scraping_dir)

from src.data.data_scraping.phoneme_extractor import download_phonemes
from src.logging.logging_config import setup_logger

# Set up logger for this module
logger = setup_logger(__name__, 'data_runner.log')

def run_data_pipeline():
    # Default languages if none provided via command line
    # You can edit this list to include the languages you want to process
    languages = ["pl", "en"]
    
    # Check if languages are passed as command line arguments
    if len(sys.argv) > 1:
        languages = sys.argv[1:]
        
    logger.info(f"Languages to process: {languages}")
    
    # Define data directory relative to this script
    # src/data/runner.py -> .../data
    base_data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data")

    for lang in languages:
        logger.info(f"Starting extraction for: {lang}")
        lang_dir = os.path.join(base_data_dir, lang)
        
        try:
            # Download phonemes
            download_phonemes(lang, lang_dir)
        except Exception as e:
            logger.error(f"Error processing {lang}: {e}")

    # 3. Load data into PyTorch Datasets
    logger.info("Loading data into PyTorch Datasets")
    try:
        # Import from the datasets module
        from src.data.datasets.multilingual_dataset import MultilingualWordDataset, MultilingualPhonemeDataset
        
        # Word Dataset
        word_dataset = MultilingualWordDataset(languages, base_data_dir)
        logger.info(f"Loaded Word Dataset with {len(word_dataset)} characters.")
        if len(word_dataset) > 0:
            logger.debug(f"Sample word char: {word_dataset[0]}")

        # Phoneme Dataset
        phoneme_dataset = MultilingualPhonemeDataset(languages, base_data_dir)
        logger.info(f"Loaded Phoneme Dataset with {len(phoneme_dataset)} phonemes.")
        if len(phoneme_dataset) > 0:
            logger.debug(f"Sample phoneme: {phoneme_dataset[0]}")
            
    except ImportError as e:
        logger.error(f"Could not import datasets: {e}")
        logger.error("Make sure you are running from the project root.")
    except Exception as e:
        logger.error(f"Error loading datasets: {e}")

if __name__ == "__main__":
    run_data_pipeline()
