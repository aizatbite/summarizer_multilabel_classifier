#!/usr/bin/env python3
"""
Complete pipeline script for multi-label text classification
This script runs the complete workflow from data preparation to model training/inference
"""

import os
import sys
import subprocess
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_file_exists(filepath):
    """Check if a file exists and log appropriate message"""
    if os.path.exists(filepath):
        logger.info(f"✓ Found: {filepath}")
        return True
    else:
        logger.warning(f"✗ Missing: {filepath}")
        return False

def run_script(script_name):
    """Run a Python script and handle errors"""
    try:
        logger.info(f"Running {script_name}...")
        result = subprocess.run([sys.executable, script_name], 
                              capture_output=True, 
                              text=True, 
                              check=True)
        logger.info(f"✓ {script_name} completed successfully")
        if result.stdout:
            logger.info(f"Output: {result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"✗ {script_name} failed with error: {e}")
        if e.stdout:
            logger.error(f"STDOUT: {e.stdout}")
        if e.stderr:
            logger.error(f"STDERR: {e.stderr}")
        return False
    except Exception as e:
        logger.error(f"✗ Unexpected error running {script_name}: {e}")
        return False

def main():
    """Main pipeline function"""
    
    logger.info("="*60)
    logger.info("Multi-Label Text Classification Pipeline")
    logger.info("="*60)
    
    # Check if required files exist
    excel_file = "Private Network Tracker - Masterfile for support team 1Q25 - 23-April-25_vOL.xlsx"
    training_script = "training_data.py"
    classifier_script = "multilabel_classifier.py"
    
    logger.info("Checking required files...")
    
    files_exist = True
    if not check_file_exists(excel_file):
        logger.error("Excel data file is required for data preparation")
        files_exist = False
    
    if not check_file_exists(training_script):
        logger.error("training_data.py script is missing")
        files_exist = False
        
    if not check_file_exists(classifier_script):
        logger.error("multilabel_classifier.py script is missing")
        files_exist = False
    
    if not files_exist:
        logger.error("Missing required files. Please ensure all files are present.")
        return False
    
    # Step 1: Data Preparation
    logger.info("\n" + "="*40)
    logger.info("STEP 1: Data Preparation")
    logger.info("="*40)
    
    csv_file = "training_data_for_classifier.csv"
    
    if os.path.exists(csv_file):
        logger.info(f"Training data CSV already exists: {csv_file}")
        response = input("Do you want to regenerate the training data? (y/n): ").lower().strip()
        if response == 'y':
            logger.info("Regenerating training data...")
            if not run_script(training_script):
                logger.error("Data preparation failed!")
                return False
        else:
            logger.info("Using existing training data CSV")
    else:
        logger.info("Training data CSV not found. Generating...")
        if not run_script(training_script):
            logger.error("Data preparation failed!")
            return False
    
    # Verify CSV was created
    if not check_file_exists(csv_file):
        logger.error("Training data CSV was not created successfully")
        return False
    
    # Step 2: Model Training/Inference
    logger.info("\n" + "="*40)
    logger.info("STEP 2: Model Training/Inference")
    logger.info("="*40)
    
    model_dir = "trained_multilabel_model"
    
    if os.path.exists(model_dir) and os.path.exists(os.path.join(model_dir, "config.json")):
        logger.info("Trained model already exists. Will run inference mode.")
    else:
        logger.info("No trained model found. Will train new model.")
    
    if not run_script(classifier_script):
        logger.error("Model training/inference failed!")
        return False
    
    # Summary
    logger.info("\n" + "="*40)
    logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
    logger.info("="*40)
    
    logger.info("Generated files:")
    if os.path.exists(csv_file):
        logger.info(f"✓ Training data: {csv_file}")
    
    if os.path.exists(model_dir):
        logger.info(f"✓ Trained model: {model_dir}/")
        
    logger.info("\nTo run inference on new text:")
    logger.info("1. Edit the sample_texts in multilabel_classifier.py")
    logger.info("2. Run: python multilabel_classifier.py")
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
