"""
Master pipeline script to run data extraction and model training.
"""
import os
import sys
import glob
import torch

import logging


from src.logger.logging_config import setup_logger
from src.embedding_service.data.data_pipeline import run_data_pipeline
from src.embedding_service.embeding.train_cbow import train_model
from src.embedding_service.embeding import hyperparameters as hp


# Set up logger
logger = setup_logger(__name__, "pipeline.log")

def model_exists(languages, data_type, models_dir):
    """
    Check if a model for the given languages and data type already exists.
    """
    # Construct pattern to match model files
    # Naming convention: cbow_{data_type}_{langs_joined}.pt
    langs_joined = '_'.join(languages)
    model_path = os.path.join(models_dir, f"cbow_{data_type}_{langs_joined}.pt")
    
    exists = os.path.exists(model_path)
    return exists, [model_path] if exists else []

def train_with_retries(languages, data_type, models_dir):
    """
    Helper function to train model with OOM retries.
    """
    logger.info(f"[Step] Training {data_type} model...")
    exists, matches = model_exists(languages, data_type, models_dir)
    
    if exists:
        logger.info(f"{data_type.capitalize()} model already exists for {languages}.")
        logger.info(f"Found model(s): {matches}")
        logger.info(f"Skipping {data_type} training.")
        return

    logger.info(f"No existing {data_type} model found. Starting training...")
    current_batch_size = hp.BATCH_SIZE
    while True:
        try:
            model_path = train_model(languages=languages, data_type=data_type, batch_size=current_batch_size)
            logger.info(f"{data_type.capitalize()} training completed. Model saved to: {model_path}")
            break
        except torch.cuda.OutOfMemoryError:
            logger.error(f"CUDA Out of Memory Error during {data_type} training!")
            current_batch_size //= 2
            if current_batch_size <= 0:
                logger.error("Batch size reduced to 0. Cannot continue.")
                return
            logger.error(f"Retrying with reduced batch size: {current_batch_size}")
            torch.cuda.empty_cache()
            continue
        except Exception as e:
            logger.error(f"{data_type.capitalize()} training failed: {e}")
            return

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
    # src/embedding_service/run_train_data_pipeline.py -> project_root
    project_root = os.path.dirname(os.path.dirname(script_dir))
    models_dir = os.path.join(project_root, hp.OUTPUT_DIR)
    
    # Train phoneme model
    train_with_retries(languages, 'phonemes', models_dir)
    
    # Train word model
    train_with_retries(languages, 'words', models_dir)

    logger.info("\n" + "=" * 60)
    logger.info("Pipeline finished.")
    logger.info("=" * 60)

