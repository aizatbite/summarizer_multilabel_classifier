<<<<<<< HEAD
# summarizer_multilabel_classifier
summarize news and classify it based on multiple parameters
=======
# Multi-Label Text Classification Pipeline

A production-ready Python pipeline for multi-label text classification using Hugging Face transformers and the DeBERTa model.

## Overview

This pipeline processes Excel data containing text descriptions and classifies them into 18 predefined parameters such as Organization, Type, Activity, Technology, etc.

## Files

- `training_data.py` - Data preparation script that processes Excel files and creates training data CSV
- `multilabel_classifier.py` - Main classifier script for training and inference
- `run_pipeline.py` - Complete pipeline runner script
- `requirements.txt` - Python dependencies

## Quick Start

### 1. Setup Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/Mac:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Data

Ensure your Excel file `Private Network Tracker - Masterfile for support team 1Q25 - 23-April-25_vOL.xlsx` is in the same directory.

### 3. Run Complete Pipeline

```bash
python run_pipeline.py
```

This will automatically:
- Process the Excel data and create `training_data_for_classifier.csv`
- Train the multi-label classifier (if no trained model exists)
- Run inference on sample texts
- Save the trained model for future use

### 4. Manual Steps (Alternative)

#### Step 1: Data Preparation
```bash
python training_data.py
```

#### Step 2: Training/Inference
```bash
python multilabel_classifier.py
```

## Features

### Data Processing
- Loads data from Excel files with custom sheet and column specifications
- Handles special concatenation logic for Technology, Use cases, and Additional technologies
- Creates binary labels for 18 classification parameters
- Preprocesses text data and removes invalid entries

### Model Training
- Uses microsoft/deberta-v3-small for multi-label classification
- Automatic train/validation split
- Customizable hyperparameters (learning rate, batch size, epochs)
- Per-parameter and macro metrics calculation
- Model checkpointing and saving

### Inference
- Loads trained models for inference
- Confidence score output for each prediction
- Batch prediction support
- Configurable prediction threshold

## Classification Parameters

The model classifies text into these 18 parameters:

1. Organization
2. Type
3. Activity
4. Geography
5. Network
6. Use cases
7. Technology
8. Vendor
9. Application
10. Data
11. Integration
12. Infrastructure
13. Additional technologies
14. Budget
15. Timeline
16. KPI
17. Challenges
18. Solution

## Configuration

### Data File Requirements
- Excel file with 'Data' sheet
- Text descriptions in 'Description' column
- Parameter columns for training labels

### Model Parameters
- Model: microsoft/deberta-v3-small
- Default learning rate: 2e-5
- Default batch size: 4
- Default epochs: 3
- Validation split: 20%

### Output Files
- `training_data_for_classifier.csv` - Processed training data
- `trained_multilabel_model/` - Saved model directory
- Training logs with metrics and progress

## Usage Examples

### Training a New Model
```python
from multilabel_classifier import MultiLabelTextClassifier

classifier = MultiLabelTextClassifier()
df = classifier.load_and_preprocess_data("training_data_for_classifier.csv")
train_dataset, val_dataset = classifier.prepare_dataset(df)
classifier.train(train_dataset, val_dataset)
classifier.save_model()
```

### Running Inference
```python
classifier = MultiLabelTextClassifier()
classifier.load_model()

texts = ["AI-powered chatbot for customer service"]
predictions = classifier.predict(texts)

for pred in predictions:
    print(f"Text: {pred['text']}")
    for label, prediction in pred['predictions'].items():
        if prediction == 1:
            confidence = pred['confidence_scores'][label]
            print(f"  {label}: {confidence:.3f}")
```

## Requirements

- Python 3.7+
- PyTorch
- Transformers (Hugging Face)
- pandas
- scikit-learn
- openpyxl
- numpy

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce batch_size in the training configuration
2. **Excel file not found**: Ensure the Excel file is in the correct directory
3. **Missing columns**: Verify the Excel file has the expected structure
4. **Virtual environment issues**: Make sure the virtual environment is activated

### Performance Tips

- Use GPU for faster training (automatically detected)
- Adjust batch size based on available memory
- Use smaller models for faster inference
- Enable mixed precision training for memory efficiency

## License

This project is for internal use and research purposes.
>>>>>>> 43c374c (Initial commit: summarizer & multi label classification)
