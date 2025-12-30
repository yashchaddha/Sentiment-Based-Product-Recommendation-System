"""
Recommendation Engine Module
Extracted logic from Jupyter notebook for Flask application
"""

import pandas as pd
import numpy as np
import re
import string
import nltk
import pickle
from pathlib import Path

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.metrics.pairwise import cosine_similarity


# Global variables to store loaded data and models
reviews_df = None
item_user_matrix = None
item_similarity_df = None
product_id_to_name = None
stop_words = None
lemmatizer = None
lr_model = None
tfidf_vectorizer = None


def download_nltk_data():
    """Download required NLTK data if not already available"""
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
    
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', quiet=True)
    
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet', quiet=True)


def clean_text(text):
    """
    Clean and preprocess text for sentiment analysis
    - Lowercase
    - Remove punctuation and digits
    - Remove stopwords
    - Lemmatize words
    """
    if pd.isna(text):
        return ""
    
    # Lowercase
    text = str(text).lower()
    
    # Remove punctuation and digits
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Tokenize and remove stopwords + lemmatize
    tokens = text.split()
    tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
        if word not in stop_words and len(word) > 1
    ]
    
    return " ".join(tokens)


def load_and_prepare_data(csv_path='Reviews.csv'):
    """
    Load Reviews.csv and prepare all necessary data structures
    - Load and clean the dataset
    - Create item-user matrix
    - Compute item similarity matrix
    - Create product ID to name mapping
    """
    global reviews_df, item_user_matrix, item_similarity_df, product_id_to_name
    
    print("Loading Reviews.csv...")
    reviews_df = pd.read_csv(csv_path)
    
    # Data cleaning - drop unnecessary columns
    drop_cols = [
        "reviews_userProvince",
        "reviews_userCity",
        "reviews_didPurchase",
        "reviews_date",
        "manufacturer",
        "reviews_doRecommend"
    ]
    reviews_df = reviews_df.drop(columns=drop_cols)
    
    # Drop rows with missing values in critical columns
    reviews_df = reviews_df.dropna(
        subset=['reviews_title', 'reviews_username', 'user_sentiment']
    )
    reviews_df = reviews_df.reset_index(drop=True)
    
    print(f"Dataset loaded: {len(reviews_df)} rows")
    
    # Create item-user matrix
    print("Creating item-user matrix...")
    item_user_matrix = reviews_df.pivot_table(
        index='id',
        columns='reviews_username',
        values='reviews_rating'
    )
    
    # Fill missing values with 0 for cosine similarity calculation
    item_user_matrix_filled = item_user_matrix.fillna(0)
    
    # Compute item similarity matrix
    print("Computing item similarity matrix...")
    item_similarity = cosine_similarity(item_user_matrix_filled)
    
    item_similarity_df = pd.DataFrame(
        item_similarity,
        index=item_user_matrix_filled.index,
        columns=item_user_matrix_filled.index
    )
    
    # Create product ID to name mapping
    print("Creating product ID to name mapping...")
    product_id_to_name = (
        reviews_df[['id', 'name']]
        .drop_duplicates()
        .set_index('id')['name']
        .to_dict()
    )
    
    print("Data preparation complete!")


def initialize_text_preprocessing():
    """Initialize NLTK stopwords and lemmatizer"""
    global stop_words, lemmatizer
    
    download_nltk_data()
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    print("Text preprocessing initialized")


def load_models(models_dir='models'):
    """Load pre-trained sentiment analysis models"""
    global lr_model, tfidf_vectorizer
    
    models_path = Path(models_dir)
    
    print(f"Loading models from {models_dir}...")
    
    # Load logistic regression model
    with open(models_path / 'logistic_model.pkl', 'rb') as f:
        lr_model = pickle.load(f)
    
    # Load TF-IDF vectorizer
    with open(models_path / 'tfidf_vectorizer.pkl', 'rb') as f:
        tfidf_vectorizer = pickle.load(f)
    
    print("Models loaded successfully")


def get_similar_items(item_id, top_n=10):
    """Get similar items for a given item ID"""
    return (
        item_similarity_df[item_id]
        .sort_values(ascending=False)
        .iloc[1:top_n+1]
    )


def item_based_recommendation(user, top_n=20):
    """
    Generate item-based recommendations for a user
    
    Args:
        user: Username string
        top_n: Number of recommendations to return
    
    Returns:
        List of product IDs (recommended items)
    """
    if user not in item_user_matrix.columns:
        return []
    
    user_ratings = item_user_matrix[user].dropna()
    
    if len(user_ratings) == 0:
        return []
    
    recommendations = {}
    
    # Focus on items the user rated 4 or 5 stars
    for item_id, rating in user_ratings.items():
        if rating >= 4:
            similar_items = item_similarity_df[item_id]
            
            for sim_item, similarity_score in similar_items.items():
                if sim_item not in user_ratings.index:
                    recommendations[sim_item] = recommendations.get(sim_item, 0) + (similarity_score * rating)
    
    # Sort by recommendation score and return top N
    recommended_items = sorted(
        recommendations.items(),
        key=lambda x: x[1],
        reverse=True
    )
    
    return [item for item, score in recommended_items[:top_n]]


def item_based_recommendation_with_names(user, top_n=20):
    """
    Generate item-based recommendations with product names
    
    Args:
        user: Username string
        top_n: Number of recommendations to return
    
    Returns:
        List of product names
    """
    item_ids = item_based_recommendation(user, top_n)
    
    return [
        product_id_to_name.get(item_id, "Unknown Product")
        for item_id in item_ids
    ]


def filter_by_sentiment(product_names, top_n=5):
    """
    Filter recommended products by sentiment analysis
    Returns top N products with highest average positive sentiment probability
    
    Args:
        product_names: List of product names to filter
        top_n: Number of top products to return (default 5)
    
    Returns:
        List of tuples: (product_name, sentiment_score) sorted by score descending
    """
    if not product_names:
        return []
    
    # Filter reviews for recommended products
    recommended_df = reviews_df[reviews_df['name'].isin(product_names)].copy()
    
    if len(recommended_df) == 0:
        return []
    
    # Combine review title and text
    recommended_df['combined_review'] = (
        recommended_df['reviews_title'] + " " + recommended_df['reviews_text']
    )
    
    # Clean the reviews
    recommended_df['clean_review'] = recommended_df['combined_review'].apply(clean_text)
    
    # Remove empty reviews after cleaning
    recommended_df = recommended_df[recommended_df['clean_review'].str.len() > 0]
    
    if len(recommended_df) == 0:
        return []
    
    # Transform reviews using TF-IDF
    X_recommended_tfidf = tfidf_vectorizer.transform(recommended_df['clean_review'])
    
    # Get positive sentiment probabilities
    recommended_df['positive_sentiment_prob'] = lr_model.predict_proba(
        X_recommended_tfidf
    )[:, 1]
    
    # Calculate average sentiment score per product
    product_sentiment_scores = (
        recommended_df
        .groupby('name')['positive_sentiment_prob']
        .mean()
        .sort_values(ascending=False)
    )
    
    # Return top N products with their scores
    top_products = product_sentiment_scores.head(top_n)
    
    return [(name, float(score)) for name, score in top_products.items()]


def get_recommendations(user, top_n_recommend=20, top_n_final=5):
    """
    Main function to get recommendations for a user
    Combines item-based recommendation with sentiment filtering
    
    Args:
        user: Username string
        top_n_recommend: Number of initial recommendations (default 20)
        top_n_final: Number of final recommendations after sentiment filtering (default 5)
    
    Returns:
        Dictionary with:
            - success: bool
            - user: username
            - recommendations: list of tuples (product_name, sentiment_score)
            - message: error message if any
    """
    # Check if user exists
    if reviews_df is None:
        return {
            'success': False,
            'user': user,
            'recommendations': [],
            'message': 'Data not loaded. Please initialize the recommendation engine.'
        }
    
    if user not in reviews_df['reviews_username'].unique():
        return {
            'success': False,
            'user': user,
            'recommendations': [],
            'message': f'User "{user}" not found in the dataset.'
        }
    
    # Get initial recommendations
    top_20_products = item_based_recommendation_with_names(user, top_n=top_n_recommend)
    
    if not top_20_products:
        return {
            'success': False,
            'user': user,
            'recommendations': [],
            'message': f'No recommendations found for user "{user}". User may have no product ratings.'
        }
    
    # Filter by sentiment
    top_5_products = filter_by_sentiment(top_20_products, top_n=top_n_final)
    
    if not top_5_products:
        return {
            'success': False,
            'user': user,
            'recommendations': [],
            'message': f'No products with sufficient reviews for sentiment analysis.'
        }
    
    return {
        'success': True,
        'user': user,
        'recommendations': top_5_products,
        'message': 'Success'
    }


def initialize_engine(csv_path='Reviews.csv', models_dir='models'):
    """
    Initialize the recommendation engine
    Loads all data, models, and prepares necessary structures
    
    Args:
        csv_path: Path to Reviews.csv
        models_dir: Directory containing saved models
    """
    print("Initializing recommendation engine...")
    initialize_text_preprocessing()
    load_and_prepare_data(csv_path)
    load_models(models_dir)
    print("Recommendation engine ready!")

