import pandas as pd
import numpy as np
import logging
import os
import json
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer, 
    DataCollatorWithPadding
)
from datasets import Dataset
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MultiLabelTextClassifier:
    def __init__(self, model_name='distilbert-base-uncased', model_save_path='./trained_model'):
        """
        Initialize the multi-label text classifier
        
        Args:
            model_name: HuggingFace model name to use
            model_save_path: Path to save/load trained models
        """
        self.model_name = model_name
        self.model_save_path = model_save_path
        self.tokenizer = None
        self.model = None
        self.label_columns = [
            'Organization', 'Type', 'Activity', 'Industry', 'Country', 'Region',
            'Telecoms operator', 'Alternative service provider', 'Vendor', 
            'Other partners', 'Other partner type', 'Technology', 'Use cases',
            'Additional technologies', 'Main network type', 'Deal structure',
            'Value (US$m)', 'Date'
        ]
        
        # Special concatenation columns
        self.tech_cols = ['Private 5G', 'Private LTE', 'CBRS', 'MulteFire', 
                         'Network Slicing', 'Fixed Wireless Access (FWA)']
        self.use_case_cols = ['IoT', 'Enterprise workforce']
        self.add_tech_cols = ['Edge computing', 'Slice', 'AI', 'Other']
        
        logger.info(f"Initialized classifier with {len(self.label_columns)} label columns")
    
    def load_and_preprocess_data(self, file_path):
        """
        Load and preprocess data from CSV file (output from training_data.py)
        
        Args:
            file_path: Path to CSV file
            
        Returns:
            Preprocessed DataFrame
        """
        try:
            logger.info(f"Loading data from {file_path}")
            
            # Check if file is CSV or Excel
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            else:
                # Fallback to Excel format for backward compatibility
                df = pd.read_excel(file_path, sheet_name='Data', header=4, usecols='D:AW')
                # Process special concatenation columns for Excel input
                df = self._process_concatenation_columns(df)
            
            logger.info(f"Loaded {len(df)} rows with columns: {df.columns.tolist()}")
            
            # Handle missing Description column
            if 'Description' not in df.columns:
                raise ValueError("Description column not found in the data")
            
            # Remove rows with empty descriptions
            df = df.dropna(subset=['Description'])
            logger.info(f"After removing empty descriptions: {len(df)} rows")
            
            # Create binary labels for each parameter
            df = self._create_binary_labels(df)
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def _process_concatenation_columns(self, df):
        """Process special concatenation logic for Technology, Use cases, Additional technologies"""
        
        # Convert relevant columns to binary (1 if ticked, 0 otherwise)
        all_special_cols = self.tech_cols + self.use_case_cols + self.add_tech_cols
        
        for col in all_special_cols:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: 1 if x == 1 else 0)
        
        # Create Technology column
        if all(col in df.columns for col in self.tech_cols):
            df['Technology'] = df[self.tech_cols].apply(
                lambda row: ', '.join([col for col, val in row.items() if val == 1]), axis=1
            )
        
        # Create Use cases column
        if all(col in df.columns for col in self.use_case_cols):
            df['Use cases'] = df[self.use_case_cols].apply(
                lambda row: ', '.join([col for col, val in row.items() if val == 1]), axis=1
            )
        
        # Create Additional technologies column
        if all(col in df.columns for col in self.add_tech_cols):
            df['Additional technologies'] = df[self.add_tech_cols].apply(
                lambda row: ', '.join([col for col, val in row.items() if val == 1]), axis=1
            )
        
        logger.info("Processed concatenation columns")
        return df
    
    def _create_binary_labels(self, df):
        """Create binary labels for each parameter (1 if mentioned, 0 if not)"""
        
        for col in self.label_columns:
            if col in df.columns:
                # Convert to binary: 1 if not empty/null, 0 otherwise
                df[f'{col}_label'] = df[col].apply(lambda x: 1 if pd.notna(x) and str(x).strip() != '' else 0)
            else:
                # If column doesn't exist, set all labels to 0
                df[f'{col}_label'] = 0
                logger.warning(f"Column '{col}' not found, setting all labels to 0")
        
        return df
    
    def prepare_dataset(self, df, test_size=0.2, random_state=42):
        """
        Prepare dataset for training
        
        Args:
            df: Preprocessed DataFrame
            test_size: Fraction of data to use for validation
            random_state: Random seed for reproducibility
            
        Returns:
            train_dataset, val_dataset
        """
        # Prepare features and labels
        texts = df['Description'].tolist()
        
        # Create multi-label matrix
        label_cols = [f'{col}_label' for col in self.label_columns]
        labels = df[label_cols].values.astype(float)
        
        logger.info(f"Label distribution:")
        for i, col in enumerate(self.label_columns):
            pos_count = int(labels[:, i].sum())
            logger.info(f"  {col}: {pos_count}/{len(labels)} ({pos_count/len(labels)*100:.1f}%)")
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            texts, labels, test_size=test_size, random_state=random_state, stratify=None
        )
        
        # Create datasets
        train_dataset = Dataset.from_dict({
            'text': X_train,
            'labels': y_train.tolist()
        })
        
        val_dataset = Dataset.from_dict({
            'text': X_val,
            'labels': y_val.tolist()
        })
        
        logger.info(f"Train dataset: {len(train_dataset)} samples")
        logger.info(f"Validation dataset: {len(val_dataset)} samples")
        
        return train_dataset, val_dataset
    
    def tokenize_dataset(self, dataset):
        """Tokenize dataset for model input"""
        
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=False)
            logger.info(f"Loaded tokenizer: {self.model_name}")
        
        def tokenize_function(examples):
            return self.tokenizer(
                examples['text'],
                truncation=True,
                padding=False,
                max_length=512
            )
        
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        return tokenized_dataset
    
    def compute_metrics(self, eval_pred):
        """Compute metrics for evaluation"""
        predictions, labels = eval_pred
        
        # Apply sigmoid to get probabilities
        predictions = torch.sigmoid(torch.from_numpy(predictions)).numpy()
        
        # Convert to binary predictions (threshold = 0.5)
        binary_predictions = (predictions > 0.5).astype(int)
        
        # Calculate metrics for each label
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, binary_predictions, average=None, zero_division=0
        )
        
        # Calculate macro averages
        macro_precision = np.mean(precision)
        macro_recall = np.mean(recall)
        macro_f1 = np.mean(f1)
        
        # Calculate accuracy per sample (exact match)
        exact_match = accuracy_score(labels, binary_predictions)
        
        metrics = {
            'macro_precision': macro_precision,
            'macro_recall': macro_recall,
            'macro_f1': macro_f1,
            'exact_match_accuracy': exact_match
        }
        
        # Add per-label metrics
        for i, label in enumerate(self.label_columns):
            metrics[f'{label}_precision'] = precision[i]
            metrics[f'{label}_recall'] = recall[i]
            metrics[f'{label}_f1'] = f1[i]
        
        return metrics
    
    def train(self, train_dataset, val_dataset, 
              learning_rate=2e-5, batch_size=8, num_epochs=3, 
              weight_decay=0.01, warmup_steps=500):
        """
        Train the model
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            learning_rate: Learning rate for training
            batch_size: Batch size for training
            num_epochs: Number of training epochs
            weight_decay: Weight decay for regularization
            warmup_steps: Number of warmup steps
        """
        
        # Initialize model
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=len(self.label_columns),
            problem_type="multi_label_classification"
        )
        
        logger.info(f"Loaded model: {self.model_name}")
        
        # Tokenize datasets
        train_dataset = self.tokenize_dataset(train_dataset)
        val_dataset = self.tokenize_dataset(val_dataset)
        
        # Data collator
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.model_save_path,
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=num_epochs,
            weight_decay=weight_decay,
            warmup_steps=warmup_steps,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="macro_f1",
            greater_is_better=True,
            logging_dir=f'{self.model_save_path}/logs',
            logging_steps=50,
            save_total_limit=2,
            report_to=None
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics
        )
        
        logger.info("Starting training...")
        
        # Train the model
        trainer.train()
        
        # Save the model
        trainer.save_model()
        self.tokenizer.save_pretrained(self.model_save_path)
        
        # Save label mapping
        label_mapping = {
            'label_columns': self.label_columns,
            'tech_cols': self.tech_cols,
            'use_case_cols': self.use_case_cols,
            'add_tech_cols': self.add_tech_cols
        }
        
        with open(f'{self.model_save_path}/label_mapping.json', 'w') as f:
            json.dump(label_mapping, f, indent=2)
        
        logger.info(f"Model saved to {self.model_save_path}")
        
        return trainer
    
    def load_model(self, model_path=None):
        """Load a trained model"""
        
        if model_path is None:
            model_path = self.model_save_path
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
            
            # Load label mapping
            with open(f'{model_path}/label_mapping.json', 'r') as f:
                label_mapping = json.load(f)
                self.label_columns = label_mapping['label_columns']
                self.tech_cols = label_mapping['tech_cols']
                self.use_case_cols = label_mapping['use_case_cols']
                self.add_tech_cols = label_mapping['add_tech_cols']
            
            logger.info(f"Model loaded from {model_path}")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def predict(self, texts, threshold=0.5):
        """
        Make predictions on new texts
        
        Args:
            texts: List of texts to classify
            threshold: Threshold for binary classification
            
        Returns:
            Dictionary with predictions and confidence scores
        """
        
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not loaded. Please train or load a model first.")
        
        if isinstance(texts, str):
            texts = [texts]
        
        # Tokenize inputs
        inputs = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors='pt'
        )
        
        # Make predictions
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            
        # Apply sigmoid to get probabilities
        probabilities = torch.sigmoid(logits).numpy()
        
        # Convert to binary predictions
        binary_predictions = (probabilities > threshold).astype(int)
        
        results = []
        for i, text in enumerate(texts):
            result = {
                'text': text,
                'predictions': {},
                'confidence_scores': {}
            }
            
            for j, label in enumerate(self.label_columns):
                result['predictions'][label] = int(binary_predictions[i, j])
                result['confidence_scores'][label] = float(probabilities[i, j])
            
            results.append(result)
        
        return results

def main():
    """Main function for training and inference"""
    
    # Configuration
    DATA_FILE = "training_data_for_classifier.csv"  # CSV output from training_data.py
    MODEL_SAVE_PATH = "./trained_multilabel_model"
    
    # Check if data file exists
    if not os.path.exists(DATA_FILE):
        logger.error(f"Data file {DATA_FILE} not found!")
        logger.info("Please run training_data.py first to generate the training data CSV file.")
        return
    
    # Initialize classifier
    classifier = MultiLabelTextClassifier(model_save_path=MODEL_SAVE_PATH)
    
    # Check if we should train or load existing model
    if os.path.exists(MODEL_SAVE_PATH) and os.path.exists(f"{MODEL_SAVE_PATH}/config.json"):
        logger.info("Found existing model, loading...")
        classifier.load_model()
        
        # Example inference
        sample_texts = [
            "Verizon announced a new 5G private network deployment for manufacturing.",
            "AT&T partners with Microsoft for edge computing solutions using network slicing."
        ]
        
        predictions = classifier.predict(sample_texts)
        
        for pred in predictions:
            print(f"\nText: {pred['text']}")
            print("Predictions:")
            for label, prediction in pred['predictions'].items():
                if prediction == 1:
                    confidence = pred['confidence_scores'][label]
                    print(f"  {label}: {prediction} (confidence: {confidence:.3f})")
    
    else:
        logger.info("No existing model found, starting training...")
        
        # Load and preprocess data
        df = classifier.load_and_preprocess_data(DATA_FILE)
        
        # Prepare datasets
        train_dataset, val_dataset = classifier.prepare_dataset(df)
        
        # Train model
        classifier.train(
            train_dataset, 
            val_dataset,
            learning_rate=2e-5,
            batch_size=4,  # Adjust based on GPU memory
            num_epochs=3
        )
        
        logger.info("Training completed!")

if __name__ == "__main__":
    main()
