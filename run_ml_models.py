import argparse
import os

import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder

from utils import *


def main():
    '''
    Benchmarking Ml models for user story quality classification.
    '''

    parser = argparse.ArgumentParser(description='main run script')

    parser.add_argument('-i', '--data_dir', type=str, default="dataset/metric",
                        help="path to dataset folder")
    parser.add_argument('-e', '--experiment', type=str, default="results/metric",
                        help='experiment version to prefix the results')
    parser.add_argument('-d', '--dataset', type=str, default="Porru_Dataset",
                        help="Dataset name to work on ('Porru_Dataset' | 'Tawosi_Dataset' | 'Choet_Dataset')")
    parser.add_argument('-s', '--seed', type=int, default=42)

    args = parser.parse_args()
    out_dir_path = args.experiment
    data_dir = args.data_dir
    seed = args.seed

    np.random.seed(seed)

    if os.path.exists(out_dir_path):
        print(f"\nModels have been benchmarked at: {out_dir_path}")
        return None
    else:
        os.makedirs(out_dir_path, exist_ok=True)

    dataset_name = args.dataset

    print(f"\nProcessing {dataset_name}...")
    dataset_dir = os.path.join(data_dir, dataset_name)

    dataframes = []
    for file in os.listdir(dataset_dir):
        dataframes.append(pd.read_csv(f"{dataset_dir}/{file}"))

    df = pd.concat(dataframes, axis=0, ignore_index=True)
    df['text'] = df['title'] + ' ' + df['description']
    df['text'] = df['text'].astype(str)
    df['text'] = df['text'].str.replace('nan', '')
    df['text'] = df['text'].str.strip().replace('', np.nan)
    df = df[df['text'].notna()]

    train_data, test_data = train_test_split(df[['text', 'label']], test_size=0.2, random_state=seed)

    vectorizer_combined = TfidfVectorizer(max_features=20000)
    tfidf_train_combined = vectorizer_combined.fit_transform(train_data["text"])
    tfidf_test_combined = vectorizer_combined.transform(test_data["text"])

    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(train_data['label'])
    y_test_encoded = label_encoder.transform(test_data['label'])

    print("Training set shape:", train_data.shape)
    print("Testing set shape:", test_data.shape)

    models = [
        LogisticRegression(max_iter=1000),
        RandomForestClassifier(random_state=seed),
        GradientBoostingClassifier(random_state=seed),
        MLPClassifier(random_state=seed)
    ]

    results = []

    for model in models:
        model_name = model.__class__.__name__
        print(f"model: {model_name}")

        model.fit(tfidf_train_combined, y_train_encoded)
        predicted_categories = model.predict(tfidf_test_combined)

        precision = precision_score(y_test_encoded, predicted_categories, average='weighted')
        recall = recall_score(y_test_encoded, predicted_categories, average='weighted')
        f1 = f1_score(y_test_encoded, predicted_categories, average='weighted')

        results.append({
            'Model': model_name,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1,
        })

    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(out_dir_path, f'{dataset_name}_benchmark_results.csv'), index=False)


if __name__ == '__main__':
    main()
