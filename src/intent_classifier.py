import torch
from torch.utils.data import Dataset
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments
)
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from typing import Dict, List, Tuple
import os

class IntentDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], tokenizer: DistilBertTokenizer, max_length: int = 128):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_length)
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

class IntentClassifier:
    def __init__(self, model_path: str = None):
        print(f"Initializing IntentClassifier with model_path: {model_path}")
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.model = None
        self.label2id = None
        self.id2label = None
        
        if model_path and os.path.exists(model_path):
            try:
                print(f"Loading model from {model_path}")
                # First load the config to get the number of labels
                config_path = os.path.join(model_path, 'config.json')
                if os.path.exists(config_path):
                    import json
                    with open(config_path, 'r') as f:
                        config = json.load(f)
                        if 'id2label' in config:
                            self.id2label = config['id2label']
                            self.label2id = {v: int(k) for k, v in self.id2label.items()}
                
                # Then load the model
                self.model = DistilBertForSequenceClassification.from_pretrained(
                    model_path,
                    local_files_only=True  # Important: only look for local files
                )
                print("Model loaded successfully")
            except Exception as e:
                print(f"Error loading model: {str(e)}")
                raise
        else:
            print(f"Model path {model_path} does not exist")

    def prepare_data(self, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[IntentDataset, IntentDataset, IntentDataset]:
        # Create label encoder
        unique_labels = sorted(train_df['intent'].unique())
        self.label2id = {label: i for i, label in enumerate(unique_labels)}
        self.id2label = {i: label for label, i in self.label2id.items()}

        # Convert labels to IDs
        train_labels = [self.label2id[label] for label in train_df['intent']]
        val_labels = [self.label2id[label] for label in val_df['intent']]
        test_labels = [self.label2id[label] for label in test_df['intent']]

        # Create datasets
        train_dataset = IntentDataset(train_df['text'].tolist(), train_labels, self.tokenizer)
        val_dataset = IntentDataset(val_df['text'].tolist(), val_labels, self.tokenizer)
        test_dataset = IntentDataset(test_df['text'].tolist(), test_labels, self.tokenizer)

        return train_dataset, val_dataset, test_dataset

    def train(self, train_dataset: IntentDataset, val_dataset: IntentDataset, output_dir: str = './models/intent_classifier'):
        # Initialize model if not already done
        if self.model is None:
            self.model = DistilBertForSequenceClassification.from_pretrained(
                'distilbert-base-uncased',
                num_labels=len(self.label2id),
                id2label=self.id2label,
                label2id=self.label2id
            )

        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=3,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=64,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir=f'{output_dir}/logs',
            logging_steps=10,
            eval_strategy="steps",
            save_strategy="steps",
            eval_steps=100,
            save_steps=100,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            logging_first_step=True,
            disable_tqdm=False
        )

        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset
        )

        # Train model
        trainer.train()

        # Save model
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

    def predict(self, text: str) -> Dict[str, float]:
        """Predict intent for a single text input"""
        if self.model is None:
            raise ValueError("Model not initialized. Please train or load a model first.")

        # Prepare input
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        
        # Get prediction
        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
        # Convert to dictionary of label: probability
        probs = probabilities[0].numpy()
        return {self.id2label[str(i)]: float(prob) for i, prob in enumerate(probs)}
