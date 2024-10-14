import re

import numpy as np
import ollama
from textstat import textstat


def count_words_chars(text):
    words = text.split()
    num_words = len(words)
    num_chars = sum(len(word) for word in words)
    return num_words, num_chars


def calculate_word_char_counts(df, col_name):
    if col_name not in df.columns:
        raise ValueError(f"Column '{col_name}' does not exist in the DataFrame.")

    word_counts = [len(text.split()) for text in df[col_name]]
    total_words = sum(word_counts)
    min_words = min(word_counts) if word_counts else 0
    max_words = max(word_counts) if word_counts else 0
    avg_words = total_words / len(df[col_name]) if df[col_name].notnull().sum() else 0
    std_words = np.std(word_counts)

    char_counts = [len(''.join(text.split())) for text in df[col_name]]
    total_chars = sum(char_counts)
    min_chars = min(char_counts) if char_counts else 0
    max_chars = max(char_counts) if char_counts else 0
    avg_chars = total_chars / len(df[col_name]) if df[col_name].notnull().sum() else 0
    std_chars = np.std(char_counts)

    return {'total_words': total_words, 'min_words': min_words, 'max_words': max_words, 'avg_words': avg_words, 'std_words': std_words,
            'total_chars': total_chars, 'min_chars': min_chars, 'max_chars': max_chars, 'avg_chars': avg_chars, 'std_chars': std_chars}


def is_independent(text):
    keywords = ["and", "then", "before", "after", "or", "depends on", "requires", "linked to", "needs"]
    return 0 if any(keyword in text.lower() for keyword in keywords) else 1


def is_negotiable(text):
    pattern = (
        r'\b(?:must|exactly|mandatory|requirement|fixed|specific)\b'
    )
    return 0 if re.search(pattern, text, re.IGNORECASE) else 1


def is_valuable(text):
    pattern = (
        r'\b(?:value|benefit|advantage|improve|enhance|gain|profit|beneficial|worthwhile|reward|gain|'
        r'(?:fix|resolve|address)\s*(?:bug|issue|problem)|'
        r'release|deploy|launch|'
        r'(?:find|identify|detect)\s*bug|'
        r'(?:patch|update|upgrade|refine)\s*(?:software|system|application))\b'
    )
    return 1 if re.search(pattern, text, re.IGNORECASE) else 0


def is_estimable(text):
    keywords = ["unknown", "unclear", "ambiguous", "vague", "complex", "hard to estimate"]
    return 0 if any(keyword in text.lower() for keyword in keywords) else 1


def is_small(text, word_limit=10):
    return 1 if len(text.split()) <= word_limit else 0


def is_large(text, word_limit=10):
    return 1 if len(text.split()) >= word_limit else 0


def is_testable(text):
    combined_pattern = (
        r'\b(?:done\s+when|testable|verification|criteria|acceptance\s+criteria|pass|fail|'
        r'test|testing|qa|quality\s*assurance|validation|unit\s*test|integration\s*test)\b'
    )

    return 1 if re.search(combined_pattern, str(text), re.IGNORECASE) else 0


def completeness_check(text):
    text = text.lower()
    phrases = ['as a', 'i want', 'so that']

    scores = []
    for phrase in phrases:
        if phrase in text:
            scores.append(1)

    completeness_score = sum(scores) / len(phrases)

    return 1 if completeness_score > 0.5 else 0


def clarity_check(text):
    weights = {
        'flesch_kincaid_grade': 0.5,
        'gunning_fog': 0.5,
    }

    metrics = {
        'flesch_kincaid_grade': textstat.flesch_kincaid_grade(text),
        'gunning_fog': textstat.gunning_fog(text),
    }

    overall_clarity = 0
    for metric, value in metrics.items():
        if weights[metric] > 0:
            overall_clarity += weights[metric] * value

    min_score = min(value for value in metrics.values() if value is not None)
    max_score = max(value for value in metrics.values() if value is not None)

    normalized_clarity = (overall_clarity - min_score) / (max_score - min_score) if max_score != min_score else 0

    return 1 if normalized_clarity > 0.5 else 0


def calculate_high_low_quality(df, threshold=0.5, weights={}):
    total_weight = sum(weights.values())

    df['score'] = df.apply(lambda row: sum([row[feature] * weights[feature] for feature in weights]), axis=1) / total_weight

    min_value = df['score'].min()
    max_value = df['score'].max()
    range_value = max_value - min_value

    df['normalized_score'] = (df['score'] - min_value) / range_value
    df['label'] = df.apply(lambda row: 'high' if row['normalized_score'] >= threshold else 'low', axis=1)

    return df


def llm_pred(model_name, user_story, template, max_try=3):
    num_try = 0
    while num_try <= max_try:
        num_try += 1
        response = ollama.chat(model=model_name, messages=[
            {'role': 'system', 'content': template},
            {'role': 'user', 'content': user_story},
        ])

        pred = response['message']['content']
        pred = pred.strip().lower()
        if pred == 'low' or pred == 'high':
            return pred

    return None
