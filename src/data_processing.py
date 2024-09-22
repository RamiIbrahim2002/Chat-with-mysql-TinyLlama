import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer
import os
import pickle

# Define constants
DATASET_NAME = "AzizBelaweid/Tunisian_Language_Dataset"
MODEL_NAME = "google/flan-t5-large"
SAVE_PATH = "data/processed/"
MAX_LENGTH = 512  # Set this to an appropriate value for your use case

def load_dataset_full():
    """
    Loads the entire dataset without filtering.
    """
    print("Loading the dataset...")
    try:
        dataset = load_dataset(DATASET_NAME, split='train')
        return dataset
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

def process_batch(batch, tokenizer):
    """
    Process a batch of data.
    """
    # Handle None values by replacing them with empty strings
    texts = [text if text is not None else '' for text in batch['text']]
    
    # Tokenize the texts
    tokenized = tokenizer(texts, truncation=True, padding='max_length', max_length=MAX_LENGTH)
    
    # Add tokenized data to the batch
    batch['input_ids'] = tokenized['input_ids']
    batch['attention_mask'] = tokenized['attention_mask']
    
    return batch

def tokenize_dataset(dataset, tokenizer):
    """
    Tokenizes the dataset using the FLAN-T5 tokenizer.
    """
    print("Tokenizing dataset...")
    try:
        tokenized_dataset = dataset.map(
            lambda batch: process_batch(batch, tokenizer),
            batched=True,
            batch_size=1000  # Adjust this based on your memory constraints
        )
        return tokenized_dataset
    except Exception as e:
        print(f"Error tokenizing dataset: {e}")
        return None

def save_processed_data(dataset):
    """
    Saves the processed data to disk for later use in training.
    """
    os.makedirs(SAVE_PATH, exist_ok=True)
    processed_file = os.path.join(SAVE_PATH, "processed_data.pkl")
    print(f"Saving processed data to {processed_file}...")
    with open(processed_file, 'wb') as f:
        pickle.dump(dataset, f)

if __name__ == "__main__":
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Load full dataset
    dataset = load_dataset_full()
    if dataset is None:
        print("Failed to load dataset. Exiting.")
        exit(1)

    # Tokenize the dataset
    tokenized_dataset = tokenize_dataset(dataset, tokenizer)
    if tokenized_dataset is None:
        print("Failed to tokenize dataset. Exiting.")
        exit(1)

    # Save processed data
    save_processed_data(tokenized_dataset)
    
    print("Data processing completed successfully!")