"""
Master pipeline script to run data extraction and model training.
"""
import os
import sys
import glob

import logging


from src.logger.logging_config import setup_logger
from src.embedding_service.data.data_pipeline import run_data_pipeline
from src.embedding_service.embeding.train_cbow import train_model
from src.embedding_service.embeding import hyperparamiters as hp


# Set up logger
logger = setup_logger(__name__, "pipeline.log")

def model_exists(languages, data_type, models_dir):
    """
    Check if a model for the given languages and data type already exists.
    """
    # Construct pattern to match model files
    # Naming convention: cbow_{data_type}_{langs_joined}_{timestamp}.pt
    langs_joined = '_'.join(languages)
    pattern = os.path.join(models_dir, f"cbow_{data_type}_{langs_joined}_*.pt")
    
    matches = glob.glob(pattern)
    return len(matches) > 0, matches

def run_pipeline(languages=None):
    """
    Run the full pipeline: data extraction -> model training.
    """
    if languages is None:
        languages = hp.LANGUAGES
        
    logger.info("=" * 60)
    logger.info(f"Starting pipeline for languages: {languages}")
    logger.info("=" * 60)
    
    # 1. Run Data Pipeline (Download/Extract)
    logger.info("[Step 1/2] Running Data Pipeline...")
    try:
        run_data_pipeline(languages)
        logger.info("Data pipeline completed successfully.")
    except Exception as e:
        logger.error(f"Data pipeline failed: {e}")
        return
    
    # 2. Train models (both phonemes and words)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.normpath(os.path.join(script_dir, hp.OUTPUT_DIR))
    
    # Train phoneme model
    logger.info("[Step 2/3] Training phoneme model...")
    phoneme_exists, phoneme_matches = model_exists(languages, 'phonemes', models_dir)
    
    if phoneme_exists:
        logger.info(f"Phoneme model already exists for {languages}.")
        logger.info(f"Found model(s): {phoneme_matches}")
        logger.info("Skipping phoneme training.")
    else:
        logger.info("No existing phoneme model found. Starting training...")
        try:
            phoneme_model_path = train_model(languages=languages, data_type='phonemes')
            logger.info(f"Phoneme training completed. Model saved to: {phoneme_model_path}")
        except Exception as e:
            logger.error(f"Phoneme training failed: {e}")
            return
    
    # Train word model
    logger.info("\n[Step 3/3] Training word model...")
    word_exists, word_matches = model_exists(languages, 'words', models_dir)
    
    if word_exists:
        logger.info(f"Word model already exists for {languages}.")
        logger.info(f"Found model(s): {word_matches}")
        logger.info("Skipping word training.")
    else:
        logger.info("No existing word model found. Starting training...")
        try:
            word_model_path = train_model(languages=languages, data_type='words')
            logger.info(f"Word training completed. Model saved to: {word_model_path}")
        except Exception as e:
            logger.error(f"Word training failed: {e}")
            return

    logger.info("\n" + "=" * 60)
    logger.info("Pipeline completed successfully!")
    logger.info("=" * 60)

if __name__ == "__main__":
    # Allow passing languages via command line
    if len(sys.argv) > 1:
        languages = sys.argv[1:]
    else:
        languages = None
        
    run_pipeline(languages)
