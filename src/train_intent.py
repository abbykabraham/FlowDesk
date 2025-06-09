import pandas as pd
from intent_classifier import IntentClassifier

def main():
    # Load data
    print("Loading processed data...")
    train_df = pd.read_csv('../data/processed/intent_train.csv')
    val_df = pd.read_csv('../data/processed/intent_val.csv')
    test_df = pd.read_csv('../data/processed/intent_test.csv')

    # Initialize classifier
    classifier = IntentClassifier()

    # Prepare data
    print("Preparing data...")
    train_dataset, val_dataset, test_dataset = classifier.prepare_data(train_df, val_df, test_df)

    # Train model
    print("Training model...")
    classifier.train(train_dataset, val_dataset)

    print("Training complete! Model saved to ./models/intent_classifier")

if __name__ == "__main__":
    main()
