import torch
from transformers import T5ForConditionalGeneration, AutoTokenizer, Trainer, TrainingArguments
from datasets import Dataset
import pickle
import numpy as np

# Define constants
MODEL_NAME = "google/flan-t5-large"
PROCESSED_DATA_PATH = "data/processed/processed_data.pkl"
OUTPUT_DIR = "./results"
LOGGING_DIR = "./logs"
FINAL_MODEL_DIR = "./fine_tuned_model"

def load_processed_data():
    """
    Load the processed dataset from a pickle file.
    """
    print("Loading processed data...")
    with open(PROCESSED_DATA_PATH, 'rb') as f:
        return pickle.load(f)

def prepare_dataset(dataset):
    """
    Prepare the dataset for training by setting the format and creating train/val splits.
    """
    print("Preparing dataset for training...")
    
    # Function to prepare inputs for T5
    def prepare_t5_input(examples):
        input_ids = examples['input_ids']
        attention_mask = examples['attention_mask']
        
        # For T5, we need to provide decoder_input_ids
        decoder_input_ids = np.roll(input_ids, 1, axis=1)
        decoder_input_ids[:, 0] = 0  # Set first token to pad token
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'decoder_input_ids': decoder_input_ids,
            'labels': input_ids  # For text generation, we use the input as the label
        }
    
    # Apply the preparation function
    dataset = dataset.map(prepare_t5_input, batched=True)
    
    # Set the format of the dataset to PyTorch tensors
    dataset = dataset.with_format("torch")
    
    # Create a validation split (10% of the data)
    dataset = dataset.train_test_split(test_size=0.1)
    
    return dataset['train'], dataset['test']

def model_init():
    """
    Initialize the model. This function will be passed to the Trainer.
    """
    return T5ForConditionalGeneration.from_pretrained(MODEL_NAME)

def compute_metrics(eval_pred):
    """
    Compute metrics for evaluation.
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    # Compute accuracy
    accuracy = (predictions == labels).astype(np.float32).mean().item()
    
    return {"accuracy": accuracy}

def main():
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Load and prepare dataset
    dataset = load_processed_data()
    train_dataset, eval_dataset = prepare_dataset(dataset)
    
    # Set training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        evaluation_strategy="steps",
        eval_steps=500,
        logging_dir=LOGGING_DIR,
        per_device_train_batch_size=2,  # Increase this if your GPU can handle it
        per_device_eval_batch_size=2,    # Increase this if your GPU can handle it
        num_train_epochs=3,
        save_steps=1000,
        save_total_limit=2,
        logging_steps=100,
        report_to="none",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        fp16=True,  # Enable mixed precision training
        gradient_accumulation_steps=4,   # Gradients are accumulated over 4 steps

    )
    
    # Initialize the Trainer
    trainer = Trainer(
        model_init=model_init,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    
    # Start training
    print("Starting training...")
    trainer.train()
    
    # Save the best model
    print(f"Saving the best model to {FINAL_MODEL_DIR}...")
    trainer.save_model(FINAL_MODEL_DIR)
    tokenizer.save_pretrained(FINAL_MODEL_DIR)
    
    print("Training completed successfully!")

if __name__ == "__main__":
    main()
