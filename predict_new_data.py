#!/usr/bin/env python3
"""
Script to predict classifications for new text descriptions using the trained model
and export results to Excel format
"""

import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime
from multilabel_classifier import MultiLabelTextClassifier
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def predict_new_descriptions(input_file, output_file=None, model_path="./trained_multilabel_model", threshold=0.5):
    """
    Predict classifications for new descriptions and save to Excel
    
    Args:
        input_file: Path to input file (CSV or Excel) with Description column
        output_file: Path to output Excel file (optional)
        model_path: Path to trained model directory
        threshold: Confidence threshold for predictions (default: 0.5)
        
    Returns:
        DataFrame with predictions
    """
    
    # Check if model exists
    if not os.path.exists(model_path):
        logger.error(f"Trained model not found at {model_path}")
        logger.info("Please train the model first by running: python multilabel_classifier.py")
        return None
    
    # Load the trained model
    logger.info("Loading trained model...")
    classifier = MultiLabelTextClassifier(model_save_path=model_path)
    classifier.load_model()
    
    # Load input data
    logger.info(f"Loading input data from {input_file}")
    if input_file.endswith('.csv'):
        df = pd.read_csv(input_file)
    elif input_file.endswith(('.xlsx', '.xls')):
        df = pd.read_excel(input_file)
    else:
        logger.error("Input file must be CSV or Excel format")
        return None
    
    # Check if Description column exists
    if 'Description' not in df.columns:
        logger.error("Input file must contain a 'Description' column")
        logger.info(f"Available columns: {df.columns.tolist()}")
        return None
    
    # Remove rows with empty descriptions
    original_count = len(df)
    df = df.dropna(subset=['Description'])
    df = df[df['Description'].str.strip() != '']
    logger.info(f"Processing {len(df)} descriptions (removed {original_count - len(df)} empty descriptions)")
    
    # Make predictions
    logger.info("Making predictions...")
    descriptions = df['Description'].tolist()
    predictions = classifier.predict(descriptions, threshold=threshold)
    
    # Create results DataFrame
    results_df = df.copy()
    
    # Add prediction columns
    for i, label in enumerate(classifier.label_columns):
        # Binary prediction (0 or 1)
        results_df[f'{label}_predicted'] = [pred['predictions'][label] for pred in predictions]
        # Confidence score (0.0 to 1.0)
        results_df[f'{label}_confidence'] = [pred['confidence_scores'][label] for pred in predictions]
    
    # Add summary columns
    results_df['total_predicted_labels'] = results_df[[f'{label}_predicted' for label in classifier.label_columns]].sum(axis=1)
    results_df['max_confidence'] = results_df[[f'{label}_confidence' for label in classifier.label_columns]].max(axis=1)
    results_df['avg_confidence'] = results_df[[f'{label}_confidence' for label in classifier.label_columns]].mean(axis=1)
    
    # Create a summary of predicted labels for each row
    predicted_labels_summary = []
    for i, pred in enumerate(predictions):
        predicted_labels = [label for label, prediction in pred['predictions'].items() if prediction == 1]
        predicted_labels_summary.append('; '.join(predicted_labels) if predicted_labels else 'None')
    
    results_df['predicted_labels_summary'] = predicted_labels_summary
    
    # Generate output filename if not provided
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"predictions_{timestamp}.xlsx"
    
    # Save to Excel with multiple sheets
    logger.info(f"Saving results to {output_file}")
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        # Main results sheet
        results_df.to_excel(writer, sheet_name='Predictions', index=False)
        
        # Summary statistics sheet
        summary_stats = create_summary_statistics(results_df, classifier.label_columns)
        summary_stats.to_excel(writer, sheet_name='Summary_Statistics', index=True)
        
        # Model configuration sheet
        model_config = create_model_config_sheet(classifier, threshold, len(descriptions))
        model_config.to_excel(writer, sheet_name='Model_Info', index=False)
    
    logger.info(f"✅ Predictions saved to {output_file}")
    logger.info(f"📊 Processed {len(descriptions)} descriptions")
    logger.info(f"🎯 Average predictions per description: {results_df['total_predicted_labels'].mean():.1f}")
    logger.info(f"📈 Average confidence: {results_df['avg_confidence'].mean():.3f}")
    
    return results_df

def create_summary_statistics(results_df, label_columns):
    """Create summary statistics for the predictions"""
    
    summary_data = []
    
    for label in label_columns:
        pred_col = f'{label}_predicted'
        conf_col = f'{label}_confidence'
        
        predicted_count = results_df[pred_col].sum()
        total_count = len(results_df)
        prediction_rate = predicted_count / total_count * 100
        avg_confidence = results_df[conf_col].mean()
        max_confidence = results_df[conf_col].max()
        min_confidence = results_df[conf_col].min()
        
        summary_data.append({
            'Label': label,
            'Predicted_Count': int(predicted_count),
            'Total_Descriptions': total_count,
            'Prediction_Rate_%': round(prediction_rate, 2),
            'Avg_Confidence': round(avg_confidence, 4),
            'Max_Confidence': round(max_confidence, 4),
            'Min_Confidence': round(min_confidence, 4)
        })
    
    return pd.DataFrame(summary_data).set_index('Label')

def create_model_config_sheet(classifier, threshold, num_descriptions):
    """Create model configuration information sheet"""
    
    config_data = [
        ['Model Information', ''],
        ['Model Name', classifier.model_name],
        ['Model Path', classifier.model_save_path],
        ['Number of Labels', len(classifier.label_columns)],
        ['', ''],
        ['Prediction Settings', ''],
        ['Confidence Threshold', threshold],
        ['Number of Descriptions Processed', num_descriptions],
        ['Prediction Date', datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
        ['', ''],
        ['Labels Used', ''],
    ]
    
    # Add all labels
    for i, label in enumerate(classifier.label_columns, 1):
        config_data.append([f'Label {i}', label])
    
    return pd.DataFrame(config_data, columns=['Parameter', 'Value'])

def main():
    """Main function to run predictions"""
    
    print("🤖 Multi-Label Text Classification - Prediction Tool")
    print("=" * 60)
    
    # Check for input arguments
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else None
        threshold = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5
    else:
        # Interactive mode
        print("\n📋 Input Options:")
        print("1. Use existing training data CSV for testing")
        print("2. Specify custom input file")
        
        choice = input("\nEnter your choice (1 or 2): ").strip()
        
        if choice == "1":
            input_file = "training_data_for_classifier.csv"
            if not os.path.exists(input_file):
                print(f"❌ File {input_file} not found!")
                return
        elif choice == "2":
            input_file = input("Enter path to input file (CSV or Excel): ").strip()
            if not os.path.exists(input_file):
                print(f"❌ File {input_file} not found!")
                return
        else:
            print("❌ Invalid choice!")
            return
        
        output_file = input("Enter output Excel filename (press Enter for auto-generated): ").strip()
        if not output_file:
            output_file = None
        
        threshold_input = input("Enter confidence threshold (0.0-1.0, default 0.5): ").strip()
        threshold = float(threshold_input) if threshold_input else 0.5
    
    # Run predictions
    results_df = predict_new_descriptions(
        input_file=input_file,
        output_file=output_file,
        threshold=threshold
    )
    
    if results_df is not None:
        print("\n✅ Prediction completed successfully!")
        
        # Show sample predictions
        print(f"\n📝 Sample Predictions (first 3 descriptions):")
        print("-" * 80)
        
        for i in range(min(3, len(results_df))):
            desc = results_df.iloc[i]['Description'][:100] + "..." if len(results_df.iloc[i]['Description']) > 100 else results_df.iloc[i]['Description']
            predicted = results_df.iloc[i]['predicted_labels_summary']
            confidence = results_df.iloc[i]['max_confidence']
            
            print(f"\nDescription {i+1}: {desc}")
            print(f"Predicted Labels: {predicted}")
            print(f"Max Confidence: {confidence:.3f}")
    
    else:
        print("❌ Prediction failed!")

if __name__ == "__main__":
    main()
