# For the testing with finetuned model

import numpy as np
import re
import string
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import pandas as pd
import os
from dateutil import parser
from datetime import datetime
import torch
from peft import PeftModel

# ------------------------------
# 1. Load Fine-Tuned Model
# ------------------------------
print("Loading the fine-tuned model...")
# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained('llama_finetuned_user_peft', use_fast=False)
tokenizer.pad_token = tokenizer.eos_token  # Set pad_token_id

# Load the base model
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B",
    device_map="auto",
    load_in_8bit=True,
    torch_dtype=torch.float16,
)

# Load the PEFT fine-tuned model
model = PeftModel.from_pretrained(model, 'llama_finetuned_user_peft')
model.to('cuda')

# Set the model to evaluation mode
model.eval()

# ------------------------------
# 2. Load and Preprocess Datasets
# ------------------------------
test_files = {
    "test_l1": "./test_l1.json",
    "test_l1_future": "./test_l1_future.json",
    "test_l2": "./test_l2.json",
    "test_l3": "./test_l3.json"
}

test_datasets = {}
for split, path in test_files.items():
    print(f"Loading {split} from {path}...")
    test_datasets[split] = load_dataset("json", data_files=path)["train"]

def preprocess_example(example):
    """
    Preprocess each example by structuring it into a prompt format.
    """
    # Construct the prompt
    prompt = f"Question: {example['question']}\nAnswer:"
    example["prompt"] = prompt
    return example

for split in test_datasets:
    print(f"Preprocessing {split}...")
    test_datasets[split] = test_datasets[split].map(preprocess_example)

# ------------------------------
# 3. Define Inference Functions
# ------------------------------
def generate_prompts(dataset):
    """
    Extract prompts from the dataset.
    """
    return [example["prompt"] for example in dataset]

def batch_generate_responses(prompts, batch_size=8, max_length=100):
    """
    Generate responses in batches using the fine-tuned model.
    """
    responses = []
    for i in tqdm(range(0, len(prompts), batch_size), desc="Generating responses"):
        batch_prompts = prompts[i:i+batch_size]
        # Tokenize inputs
        inputs = tokenizer(batch_prompts, return_tensors='pt', padding=True, truncation=True).to('cuda')
        # Generate outputs
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_new_tokens=max_length,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                do_sample=False,  # For deterministic output
            )
        # Decode outputs
        batch_responses = tokenizer.batch_decode(outputs[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        responses.extend([response.strip() for response in batch_responses])
    return responses

# ------------------------------
# 4. Define Evaluation Metrics
# ------------------------------
def normalize_text(s):
    """Lowercase, remove punctuation, and extra whitespace."""
    s = s.lower()
    s = s.translate(str.maketrans('', '', string.punctuation))
    s = ' '.join(s.split())
    return s

# Text-based metric functions
def compute_exact_match_text(prediction, ground_truth):
    """Compute exact match between prediction and ground truth for text-based datasets."""
    pred_text = normalize_text(prediction)
    gt_text = normalize_text(ground_truth)
    return pred_text == gt_text

def compute_token_f1_text(prediction, ground_truth):
    """Compute token-level F1 score between prediction and ground truth for text-based datasets."""
    pred_tokens = normalize_text(prediction).split()
    gt_tokens = normalize_text(ground_truth).split()
    common_tokens = set(pred_tokens) & set(gt_tokens)
    num_common = len(common_tokens)
    if len(pred_tokens) == 0 or len(gt_tokens) == 0:
        return 0.0
    precision = num_common / len(pred_tokens)
    recall = num_common / len(gt_tokens)
    if precision + recall == 0:
        return 0.0
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1

def compute_average_metrics_text(predictions, ground_truths):
    """
    Compute average EM and F1 for text-based datasets.
    """
    em_scores = []
    f1_scores = []
    for pred, gt in zip(predictions, ground_truths):
        em = compute_exact_match_text(pred, gt)
        f1 = compute_token_f1_text(pred, gt)
        em_scores.append(em)
        f1_scores.append(f1)
    avg_em = np.mean(em_scores) * 100
    avg_f1 = np.mean(f1_scores) * 100
    return avg_em, avg_f1, em_scores, f1_scores

# Date-based metric functions
def extract_all_dates(text):
    """
    Extract all date strings from the given text using regex.
    Supports multiple formats:
    - Month and Year (e.g., 'Dec, 2041', 'December 2041')
    - ISO Format (e.g., '1026-08-03')
    - Year Only (e.g., '1474')
    - Full Dates with Time (e.g., '2035-07-05 00:00:00')
    """
    # Regex patterns for different date formats
    pattern_month_year = r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December|' \
                         r'Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[.,]?\s*\d{4}\b'
    pattern_iso = r'\b\d{4}-\d{2}-\d{2}\b'
    pattern_iso_datetime = r'\b\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\b'
    pattern_year = r'\b\d{4}\b'

    patterns = [pattern_month_year, pattern_iso_datetime, pattern_iso, pattern_year]

    dates = []
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        dates.extend(matches)
    return dates

def clean_prediction(prediction):
    """
    Further clean the extracted date if necessary.
    Removes trailing periods, commas, and extra whitespace.
    """
    if prediction:
        return prediction.strip("., ").strip()
    return prediction

def parse_date(date_str):
    """Parse a date string into a timezone-naive datetime object."""
    if date_str is None:
        return None
    try:
        dt = parser.parse(date_str, fuzzy=True, default=datetime(1900, 1, 1))
        # Convert to timezone-naive by removing tzinfo
        if dt.tzinfo:
            dt = dt.replace(tzinfo=None)
        return dt
    except (parser.ParserError, TypeError, ValueError):
        return None  # If parsing fails

def extract_answer_date(prediction):
    """
    Extract the calculated answer's date from the model's prediction.
    """
    dates = extract_all_dates(prediction)
    if not dates:
        return None
    # Take the first date found as the answer
    return clean_prediction(dates[0])

def compute_exact_match_date(prediction, ground_truth):
    """
    Compute exact match for date-based datasets.
    """
    pred_date_str = extract_answer_date(prediction)
    gt_dates = extract_all_dates(ground_truth)
    gt_date_str = clean_prediction(gt_dates[0]) if gt_dates else None

    if pred_date_str is None or gt_date_str is None:
        return False  # If extraction fails

    pred_date = parse_date(pred_date_str)
    gt_date = parse_date(gt_date_str)

    if pred_date and gt_date:
        # Compare only year and month
        return (pred_date.year == gt_date.year) and (pred_date.month == gt_date.month)
    return False  # If parsing fails

def compute_mae(prediction, ground_truth):
    """
    Compute the Mean Absolute Error in months between predicted and ground truth dates.
    """
    pred_date_str = extract_answer_date(prediction)
    gt_dates = extract_all_dates(ground_truth)
    gt_date_str = clean_prediction(gt_dates[0]) if gt_dates else None

    if pred_date_str is None or gt_date_str is None:
        return np.nan  # If extraction fails

    pred_date = parse_date(pred_date_str)
    gt_date = parse_date(gt_date_str)

    if pred_date and gt_date:
        # Calculate difference in months
        diff_years = pred_date.year - gt_date.year
        diff_months = pred_date.month - gt_date.month
        total_diff = diff_years * 12 + diff_months
        return abs(total_diff)
    return np.nan  # If parsing fails

def compute_trend_acc(prediction, ground_truth):
    """
    Check if the trend (before/after) is correct.
    """
    pred_date_str = extract_answer_date(prediction)
    gt_dates = extract_all_dates(ground_truth)
    gt_date_str = clean_prediction(gt_dates[0]) if gt_dates else None

    if pred_date_str is None or gt_date_str is None:
        return False  # If extraction fails

    pred_date = parse_date(pred_date_str)
    gt_date = parse_date(gt_date_str)

    if pred_date and gt_date:
        return (pred_date > gt_date)
    return False  # If parsing fails

def compute_average_metrics_date(predictions, ground_truths):
    """
    Compute per-example Exact Match, MAE, and Trend Accuracy for date-based datasets.
    """
    em_scores = []
    mae_scores = []
    trend_acc_scores = []
    for pred, gt in zip(predictions, ground_truths):
        em = compute_exact_match_date(pred, gt)
        mae = compute_mae(pred, gt)
        trend = compute_trend_acc(pred, gt)
        em_scores.append(em)
        mae_scores.append(mae)
        trend_acc_scores.append(trend)
    avg_em = np.mean(em_scores) * 100
    avg_mae = np.nanmean(mae_scores)
    trend_acc = np.mean(trend_acc_scores) * 100
    return avg_em, avg_mae, trend_acc, em_scores, mae_scores, trend_acc_scores

# ------------------------------
# 5. Define Saving Functions
# ------------------------------
def save_predictions(dataset, predictions, ground_truths, metrics, split_name, metric_type='f1', threshold=50.0):
    """
    Save all predictions and optionally the failed cases.
    """
    df = pd.DataFrame({
        "Question": [example["question"] for example in dataset],
        "Ground Truth": ground_truths,
        "Prediction": predictions
    })

    if metric_type == 'f1':
        f1_scores = metrics['f1_scores']
        em_scores = metrics['em_scores']
        df["F1 Score"] = f1_scores
        df["Exact Match"] = em_scores
        df["Pass"] = [f1 >= (threshold / 100) and em for f1, em in zip(f1_scores, em_scores)]
    elif metric_type == 'em_mae_trend':
        em_scores = metrics['em_scores']
        mae_scores = metrics['mae_scores']
        trend_acc_scores = metrics['trend_acc_scores']
        df["Exact Match"] = em_scores
        df["MAE (Months)"] = mae_scores
        df["Trend Acc (%)"] = trend_acc_scores
    else:
        raise ValueError("Unsupported metric type.")

    # Ensure output directory exists
    os.makedirs("results", exist_ok=True)
    
    # Save all predictions
    all_preds_path = os.path.join("results", f"{split_name}_predictions.csv")
    df.to_csv(all_preds_path, index=False)
    print(f"Saved all predictions to {all_preds_path}")

    # Save failed predictions based on metric type
    if metric_type == 'f1':
        # Define failure as EM==False or F1 < threshold
        failed_df = df[(df["Exact Match"] == False) | (df["F1 Score"] < (threshold / 100))]
    elif metric_type == 'em_mae_trend':
        # Define failure as not Exact Match
        failed_df = df[~df["Exact Match"]]
    
    if not failed_df.empty:
        failed_preds_path = os.path.join("results", f"{split_name}_failed_predictions.csv")
        failed_df.to_csv(failed_preds_path, index=False)
        print(f"Saved failed predictions to {failed_preds_path}")
    else:
        print(f"No failed predictions for {split_name}.")

# ------------------------------
# 6. Define Evaluation Functions
# ------------------------------
def evaluate_f1_em_dataset(dataset, split_name):
    """
    Evaluate text-based datasets (test_l2, test_l3) using custom EM and F1.
    """
    print(f"\nEvaluating {split_name} with Exact Match and F1 Score...")
    prompts = generate_prompts(dataset)
    predictions = batch_generate_responses(prompts, batch_size=8)
    
    ground_truths = [example["text_answers"]["text"][0] for example in dataset]

    # Compute EM and F1 scores using the new functions
    avg_em, avg_f1, em_scores, f1_scores = compute_average_metrics_text(predictions, ground_truths)

    print(f"{split_name} - Average Exact Match: {avg_em:.2f}%")
    print(f"{split_name} - Average F1 Score: {avg_f1:.2f}%")

    # Prepare metrics dictionary
    metrics = {
        "em_scores": em_scores,
        "f1_scores": f1_scores
    }

    # Save predictions and failed cases based on EM and F1
    save_predictions(
        dataset, 
        predictions, 
        ground_truths, 
        metrics=metrics, 
        split_name=split_name, 
        metric_type='f1', 
        threshold=50.0
    )

    # Return metrics
    return {
        "Average Exact Match": avg_em,
        "Average F1 Score": avg_f1
    }

def evaluate_date_dataset_extended(dataset, split_name):
    """
    Evaluate date-based datasets (test_l1, test_l1_future) using custom EM, MAE, and Trend Accuracy.
    """
    print(f"\nEvaluating {split_name} with EM, MAE, and Trend Accuracy...")
    prompts = generate_prompts(dataset)
    predictions = batch_generate_responses(prompts, batch_size=8)
    
    ground_truths = [example["text_answers"]["text"][0] for example in dataset]
    
    # Compute EM, MAE, and Trend Accuracy using the updated functions
    avg_em, avg_mae, trend_acc, em_scores, mae_scores, trend_acc_scores = compute_average_metrics_date(predictions, ground_truths)
    
    print(f"{split_name} - Average Exact Match: {avg_em:.2f}%")
    print(f"{split_name} - Average MAE (Months): {avg_mae:.2f}")
    print(f"{split_name} - Trend Accuracy: {trend_acc:.2f}%")
    
    # Prepare metrics dictionary
    metrics = {
        "em_scores": em_scores,
        "mae_scores": mae_scores,
        "trend_acc_scores": trend_acc_scores
    }
    
    # Save predictions and failed cases based on Exact Match
    save_predictions(
        dataset, 
        predictions, 
        ground_truths, 
        metrics=metrics, 
        split_name=split_name, 
        metric_type='em_mae_trend'
    )
    
    # Return metrics
    return {
        "Average Exact Match": avg_em,
        "Average MAE (Months)": avg_mae,
        "Trend Accuracy": trend_acc
    }


## Run Evaluations
results = {}
for split in test_datasets:
    if split in ["test_l1", "test_l1_future"]:
        metrics = evaluate_date_dataset_extended(test_datasets[split], split)
    else:
        metrics = evaluate_f1_em_dataset(test_datasets[split], split)
    results[split] = metrics

print("\nEvaluation Results:")
for split, metrics in results.items():
    print(f"{split}:")
    for metric_name, value in metrics.items():
        print(f"  {metric_name}: {value:.2f}")
