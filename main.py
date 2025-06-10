import pandas as pd
from dotenv import load_dotenv
import os
import logging
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from simpleaichat import AIChat
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib # For saving/loading models
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, roc_curve
from sklearn.preprocessing import label_binarize
from openai import OpenAI
import matplotlib.pyplot as plt
from itertools import cycle
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

load_dotenv()

# --- Download NLTK data ---
# Ensure NLTK data is downloaded once
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt', quiet=True)
try:
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    nltk.download('stopwords', quiet=True)

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# Silence noisy HTTPX logs
logging.getLogger("httpx").setLevel(logging.WARNING)

CSV_PATH = 'socioeconomic_dreams.csv'
ENV_PATH = '.env' # This is typically used by load_dotenv(), not directly in script usually
GRIDSEARCH_RESULTS_PATH = 'cache/gridsearch_results.joblib'
CLASS_TFIDF_SCORES_PATH = 'cache/all_words_class_tfidf_scores.csv'
ROC_CURVE_PLOT_PATH = 'cache/roc_auc_curves.png'
RANDOM_STATE = 42
N_SPLITS = 5 # For cross-validation
CATEGORIES = ['blue_collar', 'gig_worker', 'white_collar']


def parse_embedding(embedding_str):
    """Parses the string representation of an embedding into a numpy array."""
    if pd.isna(embedding_str):
        return None
    try:
        # Assuming format like '{0.1, 0.2, ...}' or '[0.1, 0.2, ...]'
        cleaned_str = embedding_str.strip('{}[] ')
        return np.fromstring(cleaned_str, sep=',', dtype=np.float32)
    except Exception as e:
        logging.warning(f"Could not parse embedding string: {embedding_str[:50]}... Error: {e}")
        return None

def load_dreams(path: str):
    logging.info(f"Loading data from {path}...")
    try:
        df = pd.read_csv(path)
        # Ensure 'embedding' column is parsed correctly if it's stored as a string
        if 'embedding' in df.columns:
            # Check if parsing is needed (e.g., if first non-NA embedding is a string)
            first_embedding = df['embedding'].dropna().iloc[0] if not df['embedding'].dropna().empty else None
            if isinstance(first_embedding, str):
                logging.info("Parsing string embeddings in data...")
                df['embedding'] = df['embedding'].apply(parse_embedding)
                # Drop rows where embedding parsing failed
                original_len = len(df)
                df = df.dropna(subset=['embedding'])
                if len(df) < original_len:
                    logging.warning(f"Dropped {original_len - len(df)} rows due to embedding parsing errors.")
            # Ensure embeddings are numpy arrays of consistent shape, fill None with np.nan for consistent stacking later
            # This step might be better handled before passing to ML model, to decide strategy for missing embeddings
            df['embedding'] = df['embedding'].apply(lambda x: x if isinstance(x, np.ndarray) else np.nan)


        df['n_categories'] = df[CATEGORIES].sum(axis=1)
        return df
    except FileNotFoundError:
        logging.error(f"File not found at {path}. Please ensure '{path}' exists.")
        raise
    except Exception as e:
        logging.error(f"Error loading or initially processing data from {path}: {e}")
        raise

def analyze_and_clean(df:pd.DataFrame):
    '''
    Print metrics and save relevant charts describing the dataset.
    Clean up the dataframe to prepare it for modeling.

    df.head().T
    Out[9]:
                                                                  0  ...                                                  4
    dream_id                                                10007hf  ...                                            1009r5z
    created_utc                              2022-12-31 18:19:43-08  ...                             2023-01-01 02:24:25-08
    dream         In the dream I am told that everyone has this ...  ...  I am dreaming of cursing him around a fire wit...
    embedding     {0.011063187383115292,-0.024722669273614883,-0...  ...  {0.016067076474428177,-0.028647320345044136,-0...
    author                                                The_Grelm  ...                                     creapfactorart
    blue_collar                                                   1  ...                                                  1
    gig_worker                                                    1  ...                                                  0
    white_collar                                                  0  ...                                                  0
    n_categories                                                  2  ...                                                  1

    '''

    pass

    # get stats on dataset:
    # number of dreams
    # number of dreams with embeddings
    # number of users
    # distribution of users
    # user counts by category
    # dream counts by category
    # dream counts by year
    # stacked bar chart showing dream counts by category by year
    # overlapping: number of users by n_categories

    # --- Detailed EDA ---
    logging.info("--- Initial EDA on Raw Dataset ---")
    logging.info(f"Raw dataset shape: {df.shape}")
    logging.info(f"Total dreams (raw): {len(df)}")
    logging.info(f"Dreams with embeddings (raw): {df['embedding'].notna().sum()}")
    if 'author' in df.columns:
        logging.info(f"Unique authors (raw): {df['author'].nunique()}")
    else:
        logging.info("'author' column not found for unique author count.")

    logging.info(f"Distribution of 'n_categories' (raw):\n{df['n_categories'].value_counts(dropna=False).sort_index().to_string()}")

    for cat in CATEGORIES:
        dreams_in_cat = df[cat].sum()
        authors_in_cat = 0
        if 'author' in df.columns:
            authors_in_cat = df[df[cat] == 1]['author'].nunique()
        logging.info(f"Category '{cat}' (raw): {dreams_in_cat} dreams, {authors_in_cat} unique authors.")

    if 'created_utc' in df.columns:
        # Attempt to parse 'created_utc' if it's not already datetime
        if not pd.api.types.is_datetime64_any_dtype(df['created_utc']):
            try:
                # Convert to datetime, coercing errors, and standardize to UTC
                df['created_utc_dt'] = pd.to_datetime(df['created_utc'], errors='coerce', utc=True)
            except Exception as e:
                logging.warning(f"Could not parse 'created_utc' for year stats: {e}")
                df['created_utc_dt'] = pd.Series([pd.NaT] * len(df), index=df.index) # Ensure column is NaT series
        else:
            # If already datetime, ensure it's UTC for consistency or handle appropriately
            if df['created_utc'].dt.tz is None:
                df['created_utc_dt'] = df['created_utc'].dt.tz_localize('UTC', ambiguous='NaT', nonexistent='NaT')
            else:
                df['created_utc_dt'] = df['created_utc'].dt.tz_convert('UTC')

        # Check if the conversion was successful and the column is actually datetimelike
        if pd.api.types.is_datetime64_any_dtype(df['created_utc_dt']) and df['created_utc_dt'].notna().any():
            df['year'] = df['created_utc_dt'].dt.year
            logging.info(f"Dream counts by year (raw):\n{df['year'].value_counts().sort_index().to_string()}")
        else:
            logging.info("Could not generate dream counts by year (raw) as 'created_utc_dt' is not a valid datetime series or is all NaT.")
            df['year'] = np.nan # Ensure column exists
    else:
        logging.info("'created_utc' column not found for year-based stats.")
        df['year'] = np.nan # Ensure column exists

    # Cleanup: remove records where n_categories > 1 (i.e., keep only n_categories == 1)
    original_count = len(df)
    df = df[df['n_categories'] == 1].copy() # Use .copy() to avoid SettingWithCopyWarning
    logging.info(f"Removed {original_count - len(df)} records with n_categories > 1. New dataset size: {len(df)}")

    if df.empty:
        logging.error("DataFrame is empty after filtering for n_categories == 1. Cannot proceed.")
        raise ValueError("No data remaining after filtering for single category assignment.")

    # --- EDA on Cleaned Dataset ---
    logging.info("--- EDA on Cleaned Dataset (n_categories == 1) ---")
    logging.info(f"Cleaned dataset shape: {df.shape}")
    logging.info(f"Total dreams (cleaned): {len(df)}")
    logging.info(f"Dreams with embeddings (cleaned): {df['embedding'].notna().sum()}")
    if 'author' in df.columns:
        logging.info(f"Unique authors (cleaned): {df['author'].nunique()}")

    # Create multiclass y var
    # Ensure that after filtering, each row has exactly one category marked as 1
    df['y'] = df[CATEGORIES].idxmax(axis=1)
    logging.info(f"Distribution of target variable 'y' (cleaned):\n{df['y'].value_counts(dropna=False).sort_index().to_string()}")

    for cat in CATEGORIES: # Now refers to the single assigned category in 'y'
        dreams_in_cat_cleaned = (df['y'] == cat).sum()
        authors_in_cat_cleaned = 0
        if 'author' in df.columns:
            authors_in_cat_cleaned = df[df['y'] == cat]['author'].nunique()
        logging.info(f"Category '{cat}' (cleaned): {dreams_in_cat_cleaned} dreams, {authors_in_cat_cleaned} unique authors.")
    
    if 'year' in df.columns and df['year'].notna().any(): # 'year' column was created from 'created_utc_dt'
        logging.info(f"Dream counts by year (cleaned):\n{df['year'].value_counts().sort_index().to_string()}")
    else:
        logging.info("Could not generate dream counts by year (cleaned).")

    # Clean up temporary datetime columns if they were created
    if 'created_utc_dt' in df.columns:
        df.drop(columns=['created_utc_dt'], inplace=True, errors='ignore')
    if 'year' in df.columns: # We might want to keep 'year' if it's useful downstream, or drop it. Let's drop for now.
        df.drop(columns=['year'], inplace=True, errors='ignore')
        
    # return the cleaned dataframe
    return df


def analyze_tfidf_for_category(df_full: pd.DataFrame, X_embeddings_full: np.ndarray, model, target_category: str, text_column_name: str = 'dream'):
    """
    Performs TF-IDF analysis for a specific target_category using a one-vs-rest approach
    based on model prediction probabilities. Asks LLM for themes.
    """
    logging.info(f"--- Running TF-IDF analysis for category: {target_category} ---")

    if text_column_name not in df_full.columns:
        logging.error(f"DataFrame must have '{text_column_name}' column for TF-IDF analysis.")
        return
    if df_full[text_column_name].isnull().all():
        logging.error(f"The '{text_column_name}' column contains all null values.")
        return

    try:
        # Get model's class order to map probabilities correctly
        model_classes = list(model.classes_)
        if target_category not in model_classes:
            logging.error(f"Target category '{target_category}' not found in model classes: {model_classes}")
            return
        target_category_idx = model_classes.index(target_category)

        # Predict probabilities on the full dataset's embeddings
        # Ensure X_embeddings_full is correctly scaled if the model pipeline expects it
        # If model is a pipeline, it handles scaling. If it's just LogisticRegression, scaling needs to be done before.
        # Assuming 'model' is the fitted pipeline from find_best_model, it will handle scaling.
        all_probas = model.predict_proba(X_embeddings_full)
        target_probas = all_probas[:, target_category_idx]

        # Define "top" group (high probability for target_category) vs "rest"
        # Using a threshold, e.g., top 25% percentile of probabilities for this category
        # Or simply compare rows where predicted class IS target_category vs. IS NOT.
        # For simplicity with TF-IDF, let's use predicted class.
        # df_full['predicted_class_for_tfidf'] = model.predict(X_embeddings_full) # Old method

        # Using hard predictions: documents predicted as target_category vs. documents predicted as other categories.
        df_full['predicted_class_for_tfidf_llm'] = model.predict(X_embeddings_full)

        top_mask = (df_full['predicted_class_for_tfidf_llm'] == target_category)
        bottom_mask = (df_full['predicted_class_for_tfidf_llm'] != target_category)

        num_top_docs = np.sum(top_mask)
        num_bottom_docs = np.sum(bottom_mask)

        if num_top_docs == 0:
            logging.warning(f"No dreams predicted as '{target_category}'. Skipping TF-IDF for LLM theming for this category.")
            return
        if num_bottom_docs == 0:
            logging.warning(f"All dreams predicted as '{target_category}'. Cannot perform differential TF-IDF for LLM theming. Skipping.")
            return
        
        logging.info(f"For LLM theming of '{target_category}', using {num_top_docs} documents predicted as '{target_category}' vs. {num_bottom_docs} documents predicted as other categories.")

        # Preprocessing text
        stop_words = set(stopwords.words('english'))
        # Ensure 'processed_text' is created on the df_full that aligns with tfidf_matrix
        df_full['processed_text_for_llm_tfidf'] = df_full[text_column_name].fillna('').astype(str).apply(
            lambda x: ' '.join([word.lower() for word in word_tokenize(x) if word.isalpha() and word.lower() not in stop_words])
        )

        vectorizer = TfidfVectorizer(max_features=5000)
        # Fit on all processed text to have a common vocabulary
        tfidf_matrix = vectorizer.fit_transform(df_full['processed_text_for_llm_tfidf'])
        feature_names = vectorizer.get_feature_names_out()

        # Calculate mean TF-IDF for the two groups
        # top_mask and bottom_mask are boolean arrays aligned with df_full and thus with tfidf_matrix rows

        # Check if masks align with tfidf_matrix shape (should be guaranteed if processed_text is created on the same df_full)
        if tfidf_matrix.shape[0] != len(df_full):
             logging.error("Mismatch between TF-IDF matrix rows and DataFrame rows. TF-IDF cannot be reliably performed.")
             # This can happen if df_full was modified (e.g. rows dropped) AFTER 'processed_text' was created
             # and BEFORE this point, without re-calculating processed_text and tfidf_matrix.
             # For this flow, df_full is passed and processed_text is created on it, so it should align.
             return

        top_tfidf_mean = np.array(tfidf_matrix[top_mask].mean(axis=0)).flatten()
        bottom_tfidf_mean = np.array(tfidf_matrix[bottom_mask].mean(axis=0)).flatten()


        comparison_df = pd.DataFrame({
            'feature': feature_names,
            'target_category_mean_tfidf': top_tfidf_mean,
            'other_categories_mean_tfidf': bottom_tfidf_mean
        })
        comparison_df[f'diff_{target_category}_vs_others'] = comparison_df['target_category_mean_tfidf'] - comparison_df['other_categories_mean_tfidf']

        n_words = 20 # Reduced for clarity per category
        top_words_for_category = comparison_df.sort_values(f'diff_{target_category}_vs_others', ascending=False).head(n_words)['feature'].tolist()
        # Words more associated with "others" when compared to target_category
        # bottom_words_for_category = comparison_df.sort_values(f'diff_{target_category}_vs_others', ascending=True).head(n_words)['feature'].tolist()


        logging.info(f"Top {n_words} words most strongly associated with '{target_category}' (vs other categories):\n{', '.join(top_words_for_category)}")
        # logging.info(f"Top {n_words} words most strongly associated with OTHER categories (vs '{target_category}'):\n{', '.join(bottom_words_for_category)}")


        # LLM Analysis for Theming
        try:
            ai = AIChat(console=False, model='gpt-4o', api_key=os.getenv("OPENAI_API_KEY")) # Ensure API key is loaded
            prompt_category_themes = f"""The following words are most strongly associated with dreams classified as '{target_category}' when compared to dreams classified as other categories. Please group these words into meaningful semantic themes (2-4 themes usually works well). Provide only the theme names and the words belonging to each theme.

Words: {', '.join(top_words_for_category)}"""
            response_themes = ai(prompt_category_themes)
            logging.info(f"LLM Theming for '{target_category}' associated words:\n{response_themes}")

        except ImportError:
            logging.warning("simpleaichat not installed or OPENAI_API_KEY not set. Skipping LLM theming.")
        except Exception as e:
            logging.error(f"Error during LLM theming for '{target_category}': {e}")

    except Exception as e:
        logging.error(f"Error during TF-IDF analysis for category '{target_category}': {e}")
    finally:
        # Clean up temporary columns if added
        if 'processed_text_for_llm_tfidf' in df_full.columns:
            df_full.drop(columns=['processed_text_for_llm_tfidf'], inplace=True)
        if 'predicted_class_for_tfidf_llm' in df_full.columns: 
            df_full.drop(columns=['predicted_class_for_tfidf_llm'], inplace=True, errors='ignore')


def find_best_model(X, y, n_splits=N_SPLITS):
    """
    Performs GridSearchCV to find the best multiclass logistic regression model and hyperparameters.
    Tests Logistic Regression with and without PCA.
    Caches the best found pipeline to avoid re-running.
    """

    if os.path.exists(GRIDSEARCH_RESULTS_PATH):
        logging.info(f"Loading best pipeline from cache: {GRIDSEARCH_RESULTS_PATH}")
        try:
            best_pipeline = joblib.load(GRIDSEARCH_RESULTS_PATH)
            logging.info(f"Successfully loaded cached pipeline.")
            return best_pipeline
        except Exception as e:
            logging.error(f"Failed to load pipeline from cache: {e}. Re-running GridSearchCV.")

    logging.info("No valid cache found. Starting GridSearchCV to find the best model and hyperparameters...")

    pipe_base = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', None)
    ])

    pipe_pca = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(random_state=RANDOM_STATE)),
        ('clf', None)
    ])

    # Logistic Regression setup for multiclass
    param_grid_lr = {
        'clf': [LogisticRegression(
            solver='saga',
            multi_class='multinomial',
            max_iter=1000,
            random_state=RANDOM_STATE,
            class_weight='balanced'
        )],
        'clf__C': [0.001, 0.01, 0.1, 1, 10],
        'clf__penalty': ['l1', 'l2']
    }

    param_grid_pca = {
        'pca__n_components': [50, 125, 200]
    }

    param_grid_lr_pca = {**param_grid_lr, **param_grid_pca}

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    scoring = 'roc_auc_ovr'
    best_score = -1
    best_pipeline = None
    overall_best_params = None

    search_configs = [
        ("Logistic Regression (No PCA)", pipe_base, param_grid_lr),
        ("Logistic Regression (with PCA)", pipe_pca, param_grid_lr_pca),
    ]

    for name, pipe, params in search_configs:
        logging.info(f"--- Running GridSearchCV for: {name} ---")
        search = GridSearchCV(pipe, params, scoring=scoring, cv=cv, n_jobs=-1, verbose=1)
        try:
            search.fit(X, y)
            logging.info(f"Best score ({scoring}) for {name}: {search.best_score_:.4f}")
            logging.info(f"Best params for {name}: {search.best_params_}")

            if search.best_score_ > best_score:
                best_score = search.best_score_
                best_pipeline = search.best_estimator_
                overall_best_params = search.best_params_
                logging.info(f"*** New overall best model found: {name} (Score: {best_score:.4f}) ***")

        except Exception as e:
            logging.error(f"GridSearchCV failed for {name}: {e}")

    if best_pipeline is None:
        logging.error("GridSearchCV failed to find any valid model.")
        raise RuntimeError("Model optimization failed.")

    logging.info(f"--- GridSearchCV Finished ---")
    logging.info(f"Overall best pipeline score ({scoring}): {best_score:.4f}")
    if overall_best_params:
        logging.info(f"Overall best hyperparameters: {overall_best_params}")
    logging.info(f"Overall best pipeline configuration: {best_pipeline}")

    try:
        logging.info(f"Saving the best pipeline to cache: {GRIDSEARCH_RESULTS_PATH}")
        joblib.dump(best_pipeline, GRIDSEARCH_RESULTS_PATH)
    except Exception as e:
        logging.error(f"Failed to save best pipeline to cache: {e}")

    return best_pipeline

def evaluate_performance(model, X_test, y_test, target_names=None):
    '''Demonstrates model performance on the test set.'''
    logging.info("--- Evaluating Model Performance on Test Set ---")
    if not hasattr(model, 'predict'):
        logging.error("Provided model object does not have a 'predict' method.")
        return

    try:
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test) # Get probabilities for ROC AUC

        # 1. Classification Report
        report = classification_report(y_test, y_pred, target_names=target_names, zero_division=0)
        logging.info("Classification Report:\n" + report)

        # 2. Confusion Matrix
        cm = confusion_matrix(y_test, y_pred, labels=target_names)
        logging.info(f"Confusion Matrix (Rows: True, Cols: Predicted):\nLabels: {target_names}\n{cm}")
        
        cm_normalized = confusion_matrix(y_test, y_pred, labels=target_names, normalize='true')
        logging.info(f"Normalized Confusion Matrix (Rows: True, Cols: Predicted):\nLabels: {target_names}\n{cm_normalized}")

        # 3. ROC AUC Score (Multiclass)
        # Binarize the output classes for OvR ROC AUC calculation
        y_test_binarized = label_binarize(y_test, classes=target_names)
        
        # Ensure y_proba columns align with target_names if model.classes_ might differ
        # For safety, reorder y_proba columns according to target_names if necessary
        # This assumes target_names is the definitive order from CATEGORIES
        model_class_order = list(model.classes_)
        if model_class_order != target_names:
            logging.warning(f"Model classes order {model_class_order} differs from target_names {target_names}. Reordering y_proba for ROC AUC.")
            # Create a mapping from model_class_order to indices
            class_to_idx = {cls_name: idx for idx, cls_name in enumerate(model_class_order)}
            # Get the indices in the order of target_names
            ordered_indices = [class_to_idx[cls_name] for cls_name in target_names if cls_name in class_to_idx]
            # Reorder y_proba
            y_proba_ordered = y_proba[:, ordered_indices]
        else:
            y_proba_ordered = y_proba

        if y_test_binarized.shape[1] == y_proba_ordered.shape[1]:
            roc_auc_ovr = roc_auc_score(y_test_binarized, y_proba_ordered, multi_class='ovr', average='weighted')
            logging.info(f"Weighted OvR ROC AUC Score: {roc_auc_ovr:.4f}")
        else:
            logging.error(f"Shape mismatch for ROC AUC: y_test_binarized ({y_test_binarized.shape}) vs y_proba_ordered ({y_proba_ordered.shape}). Skipping ROC AUC score.")


        # 4. Plot and Save ROC Curves (One-vs-Rest for each class)
        plt.figure(figsize=(10, 8))
        colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'red', 'purple']) # Add more if more classes

        for i, (class_name, color) in enumerate(zip(target_names, colors)):
            if i < y_test_binarized.shape[1] and i < y_proba_ordered.shape[1]:
                fpr, tpr, _ = roc_curve(y_test_binarized[:, i], y_proba_ordered[:, i])
                roc_auc = roc_auc_score(y_test_binarized[:, i], y_proba_ordered[:, i])
                plt.plot(fpr, tpr, color=color, lw=2,
                         label=f'ROC curve of class {class_name} (area = {roc_auc:.2f})')
            else:
                logging.warning(f"Skipping ROC curve for class {class_name} due to index mismatch.")

        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) - Multiclass (One-vs-Rest)')
        plt.legend(loc="lower right")
        
        try:
            os.makedirs(os.path.dirname(ROC_CURVE_PLOT_PATH), exist_ok=True)
            plt.savefig(ROC_CURVE_PLOT_PATH)
            logging.info(f"ROC curves saved to {ROC_CURVE_PLOT_PATH}")
        except Exception as e_plot:
            logging.error(f"Failed to save ROC curve plot: {e_plot}")
        plt.close()

    except Exception as e:
        logging.error(f"Error during model evaluation: {e}")

def get_dream_scores(dream_text: str, model_path: str = GRIDSEARCH_RESULTS_PATH) -> dict | None:
    """
    Gets the socioeconomic category scores for a single dream text.

    Args:
        dream_text: The text of the dream.
        model_path: Path to the saved trained model.

    Returns:
        A dictionary mapping category names to scores (probabilities), or None if an error occurs.
    """
    logging.info(f"Getting scores for dream: '{dream_text[:100]}...'")

    # 1. Initialize OpenAI client and get embedding
    try:
        client = OpenAI() # API key is read from OPENAI_API_KEY environment variable
        response = client.embeddings.create(
            input=dream_text,
            model="text-embedding-3-large"
        )
        embedding = np.array(response.data[0].embedding, dtype=np.float32).reshape(1, -1)
        logging.info(f"Successfully obtained embedding of shape {embedding.shape}")
    except Exception as e:
        logging.error(f"Failed to get embedding from OpenAI: {e}")
        return None

    # 2. Load the pre-trained model
    if not os.path.exists(model_path):
        logging.error(f"Model file not found at {model_path}. Please run the main training script first.")
        return None
    try:
        model = joblib.load(model_path)
        logging.info(f"Successfully loaded model from {model_path}")
    except Exception as e:
        logging.error(f"Failed to load model from {model_path}: {e}")
        return None

    # 3. Predict probabilities
    try:
        probabilities = model.predict_proba(embedding)
        # model.classes_ should give the order of classes for the probabilities
        scores = dict(zip(model.classes_, probabilities[0]))
        logging.info(f"Predicted scores: {scores}")
        return scores
    except Exception as e:
        logging.error(f"Failed to predict scores with the model: {e}")
        return None

def score_full_dataset(csv_path: str = CSV_PATH, model_path: str = GRIDSEARCH_RESULTS_PATH) -> pd.DataFrame | None:
    """
    Loads the full dream dataset, the pre-trained model, and adds predicted scores
    for each class to the DataFrame.

    Args:
        csv_path: Path to the dream dataset CSV file.
        model_path: Path to the saved trained model.

    Returns:
        A pandas DataFrame with added score columns (e.g., 'score_blue_collar'),
        or None if an error occurs.
    """
    logging.info("--- Scoring full dataset ---")

    # 1. Load the dataset
    df = load_dreams(csv_path)
    if df is None or df.empty:
        logging.error("Failed to load or dataset is empty.")
        return None

    # Ensure embeddings are present and in correct format (numpy arrays)
    # Drop rows where 'embedding' is NaN as these cannot be scored
    df_valid_embeddings = df.dropna(subset=['embedding']).copy()
    if df_valid_embeddings.empty:
        logging.warning("No rows with valid embeddings found in the dataset to score.")
        # Return original df with no scores, or an empty df with score columns, or None?
        # For now, let's add empty score columns to the original df to indicate attempt.
        for category in CATEGORIES: # Assuming CATEGORIES is globally defined
            df[f'score_{category}'] = np.nan
        return df

    X_embeddings = np.vstack(df_valid_embeddings['embedding'].values)

    # 2. Load the pre-trained model
    if not os.path.exists(model_path):
        logging.error(f"Model file not found at {model_path}. Please run the main training script first.")
        return None
    try:
        model = joblib.load(model_path)
        logging.info(f"Successfully loaded model from {model_path}")
    except Exception as e:
        logging.error(f"Failed to load model from {model_path}: {e}")
        return None

    # 3. Predict probabilities for all valid embeddings
    try:
        all_probabilities = model.predict_proba(X_embeddings)
        logging.info(f"Successfully predicted probabilities for {len(X_embeddings)} dreams.")
    except Exception as e:
        logging.error(f"Failed to predict probabilities with the model: {e}")
        return None

    # 4. Add scores to the DataFrame
    # Create score columns in the original df, then fill for valid rows
    for i, category in enumerate(model.classes_): # Use model.classes_ for correct order
        score_col_name = f'score_{category}'
        # Initialize column in the main df (df_valid_embeddings is a subset)
        df[score_col_name] = np.nan
        # Assign scores to the corresponding rows in the original DataFrame
        # using the index from df_valid_embeddings
        df.loc[df_valid_embeddings.index, score_col_name] = all_probabilities[:, i]

    logging.info(f"Added score columns: {', '.join([f'score_{cat}' for cat in model.classes_])}")
    return df

def calculate_and_save_class_tfidf_scores(
    df_full: pd.DataFrame,
    X_embeddings_full: np.ndarray,
    model,
    text_column_name: str = 'dream',
    output_path: str = CLASS_TFIDF_SCORES_PATH
):
    """
    Calculates mean TF-IDF scores for all words for each predicted class and saves them to a CSV.
    """
    logging.info(f"--- Calculating and saving TF-IDF scores for all words per class to {output_path} ---")

    if text_column_name not in df_full.columns:
        logging.error(f"DataFrame must have '{text_column_name}' column.")
        return
    if df_full[text_column_name].isnull().all():
        logging.error(f"The '{text_column_name}' column contains all null values.")
        return

    # 1. Preprocess text
    stop_words = set(stopwords.words('english'))
    # Create a temporary column for processed text on a copy to avoid SettingWithCopyWarning
    df_processed = df_full.copy()
    df_processed['processed_text_for_global_tfidf'] = df_processed[text_column_name].fillna('').astype(str).apply(
        lambda x: ' '.join([word.lower() for word in word_tokenize(x) if word.isalpha() and word.lower() not in stop_words])
    )

    # 2. Fit TF-IDF Vectorizer on all processed text
    vectorizer = TfidfVectorizer(max_features=5000) # Standardizing to 5000 features
    try:
        tfidf_matrix = vectorizer.fit_transform(df_processed['processed_text_for_global_tfidf'])
        feature_names = vectorizer.get_feature_names_out()
    except ValueError as e:
        logging.error(f"TF-IDF Vectorizer fitting failed, possibly due to empty vocabulary: {e}")
        return
    
    if tfidf_matrix.shape[0] == 0 or tfidf_matrix.shape[1] == 0:
        logging.warning("TF-IDF matrix is empty. Skipping saving scores.")
        return

    # 3. Get model's hard predictions for all documents
    predicted_classes = model.predict(X_embeddings_full)
    df_processed['predicted_class_for_global_tfidf'] = predicted_classes # Add to the df with processed text

    # 4. Calculate mean TF-IDF for each word within documents predicted for each class
    class_tfidf_scores_data = {'feature': feature_names}

    for category_name in model.classes_:
        # Create a mask for documents predicted to be in the current category
        class_mask = (df_processed['predicted_class_for_global_tfidf'] == category_name)
        
        num_predicted_docs = np.sum(class_mask)

        if num_predicted_docs > 0:
            logging.info(f"Calculating mean TF-IDF for '{category_name}' using {num_predicted_docs} documents predicted as this class.")
            # Calculate mean TF-IDF for words in documents belonging to this class
            # Ensure tfidf_matrix rows align with df_processed rows where class_mask is derived
            mean_scores_for_class = np.array(tfidf_matrix[class_mask].mean(axis=0)).flatten()
        else:
            # If no documents are predicted for this class
            logging.warning(f"No documents predicted as '{category_name}'. Mean TF-IDF scores for this class will be NaN.")
            mean_scores_for_class = np.full(len(feature_names), np.nan)
        
        class_tfidf_scores_data[f'mean_tfidf_predicted_as_{category_name}'] = mean_scores_for_class

    # 5. Create DataFrame with mean TF-IDF scores
    tfidf_scores_df = pd.DataFrame(class_tfidf_scores_data)

    # 6. Calculate and add difference columns
    all_class_names = list(model.classes_)
    for target_category_name in all_class_names:
        mean_tfidf_col_target = f'mean_tfidf_predicted_as_{target_category_name}'
        
        other_category_names = [cn for cn in all_class_names if cn != target_category_name]
        
        if not other_category_names: # Should not happen with >1 class
            tfidf_scores_df[f'diff_vs_others_{target_category_name}'] = tfidf_scores_df[mean_tfidf_col_target]
            continue

        # Calculate mean of other categories' TF-IDF scores for each word
        other_cols_to_average = [f'mean_tfidf_predicted_as_{other_cat}' for other_cat in other_category_names]
        
        # Ensure all necessary columns exist before trying to average them
        valid_other_cols_to_average = [col for col in other_cols_to_average if col in tfidf_scores_df.columns]
        
        if not valid_other_cols_to_average:
            logging.warning(f"No valid TF-IDF columns found for 'other' categories when calculating diff for '{target_category_name}'. Skipping diff calculation for this category.")
            tfidf_scores_df[f'diff_vs_others_{target_category_name}'] = np.nan
            continue
            
        mean_of_others = tfidf_scores_df[valid_other_cols_to_average].mean(axis=1)
        
        tfidf_scores_df[f'diff_vs_others_{target_category_name}'] = tfidf_scores_df[mean_tfidf_col_target] - mean_of_others
        logging.info(f"Added difference column: 'diff_vs_others_{target_category_name}'")

    try:
        # Ensure cache directory exists for this new file too
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        tfidf_scores_df.to_csv(output_path, index=False)
        logging.info(f"Successfully saved class TF-IDF scores (including differences) for {len(feature_names)} words to {output_path}")
    except Exception as e:
        logging.error(f"Failed to save class TF-IDF scores: {e}")

def _calculate_point_line_distance(point, line_start, line_end):
    """Calculates the perpendicular distance from a point to a line segment."""
    if np.all(line_start == line_end):
        return np.linalg.norm(point - line_start)
    
    line_vec = line_end - line_start
    point_vec = point - line_start
    line_len_sq = np.dot(line_vec, line_vec)
    
    t = np.dot(point_vec, line_vec) / line_len_sq
    t = np.clip(t, 0, 1) # Project onto the segment
    
    closest_point_on_line = line_start + t * line_vec
    return np.linalg.norm(point - closest_point_on_line)

def _estimate_dbscan_eps(embeddings, min_samples, metric='cosine'):
    """Estimates a good eps value for DBSCAN using the k-distance graph elbow method."""
    if len(embeddings) < min_samples:
        logging.warning(f"Not enough samples ({len(embeddings)}) to estimate eps for min_samples={min_samples}. Returning default eps.")
        return 0.5 # Default fallback eps

    nn = NearestNeighbors(n_neighbors=min_samples, metric=metric)
    nn.fit(embeddings)
    # distances are to the k-th neighbor (min_samples includes the point itself, so k = min_samples-1 index)
    distances, _ = nn.kneighbors(embeddings) 
    k_distances = np.sort(distances[:, min_samples-1], axis=0)

    if len(k_distances) < 2: # Need at least two points to form a line
        logging.warning("Not enough k-distances to determine elbow. Returning default eps.")
        return 0.5

    # Create points for the k-distance curve (index, distance)
    indices = np.arange(len(k_distances))
    curve_points = np.vstack((indices, k_distances)).T

    line_start_pt = curve_points[0]
    line_end_pt = curve_points[-1]

    if np.all(line_start_pt == line_end_pt): # All distances are the same
        return k_distances[0] if k_distances[0] > 0 else 0.5


    # Calculate perpendicular distance of each point to the line connecting start and end of curve
    # This is a simplified way to find the "knee" or "elbow"
    # We are looking for the point on the curve (indices[i], k_distances[i])
    # that is furthest from the line connecting (indices[0], k_distances[0]) and (indices[-1], k_distances[-1])
    
    # Line defined by P1=(x1,y1) and P2=(x2,y2) is (y1-y2)x + (x2-x1)y + (x1y2-x2y1) = 0
    # A = y1-y2, B = x2-x1, C = x1y2-x2y1
    x1, y1 = indices[0], k_distances[0]
    x2, y2 = indices[-1], k_distances[-1]

    # Handle vertical line case (all indices are same, should not happen with np.arange)
    # Handle horizontal line case (all k_distances are same)
    if x1 == x2: # Should not happen
        return np.median(k_distances) 
    if y1 == y2: # All k-distances are the same, any point is fine, or just use the distance
        return y1 if y1 > 1e-6 else 0.5 # Avoid eps=0

    line_A = y1 - y2
    line_B = x2 - x1
    line_C = x1 * y2 - x2 * y1
    
    norm_factor = np.sqrt(line_A**2 + line_B**2)
    if norm_factor == 0: # Should be caught by y1==y2 or x1==x2
        return np.median(k_distances)

    perp_distances = np.abs(line_A * indices + line_B * k_distances + line_C) / norm_factor
    
    elbow_index = np.argmax(perp_distances)
    estimated_eps = k_distances[elbow_index]
    
    logging.info(f"Estimated DBSCAN eps: {estimated_eps:.4f} at index {elbow_index} from k-distance plot.")
    # Ensure eps is not too small, e.g., if all distances are tiny
    return max(estimated_eps, 1e-3)


def cluster_top_words_for_themes(
    tfidf_scores_path: str = CLASS_TFIDF_SCORES_PATH,
    top_n_words: int = 50,
    min_samples: int = 4, # DBSCAN parameter, min words to form a dense region (theme). Increased from 3.
    embedding_model_name: str = 'all-MiniLM-L6-v2' # Efficient and good quality RoBERTa-based model
):
    """
    Loads TF-IDF scores, identifies top differentiating words for each class,
    embeds them, and clusters them using DBSCAN with auto-tuned eps to find semantic themes.
    """
    logging.info(f"--- Clustering top {top_n_words} words for themes using DBSCAN (auto-eps, min_samples={min_samples}) and {embedding_model_name} ---")

    if not os.path.exists(tfidf_scores_path):
        logging.error(f"TF-IDF scores file not found at {tfidf_scores_path}. Cannot perform word clustering.")
        return

    try:
        df_tfidf = pd.read_csv(tfidf_scores_path)
    except Exception as e:
        logging.error(f"Failed to load TF-IDF scores from {tfidf_scores_path}: {e}")
        return

    try:
        logging.info(f"Loading sentence transformer model: {embedding_model_name}...")
        model = SentenceTransformer(embedding_model_name)
        logging.info("Sentence transformer model loaded successfully.")
    except Exception as e:
        logging.error(f"Failed to load sentence transformer model '{embedding_model_name}': {e}. Make sure 'sentence-transformers' is installed.")
        return

    for category_name in CATEGORIES: # Assumes CATEGORIES is globally defined
        diff_col = f'diff_vs_others_{category_name}'
        if diff_col not in df_tfidf.columns:
            logging.warning(f"Difference column '{diff_col}' not found for category '{category_name}'. Skipping clustering for this category.")
            continue

        # Get top N words for the category based on the difference score
        top_words_df = df_tfidf.sort_values(by=diff_col, ascending=False).head(top_n_words)
        
        if top_words_df.empty or 'feature' not in top_words_df.columns:
            logging.warning(f"No top words found or 'feature' column missing for category '{category_name}'. Skipping.")
            continue
            
        words_to_cluster = top_words_df['feature'].tolist()

        if len(words_to_cluster) < min_samples: 
            logging.warning(f"Category '{category_name}' has only {len(words_to_cluster)} top words, which is less than DBSCAN min_samples ({min_samples}). Skipping clustering for this category.")
            continue
        
        logging.info(f"\n--- Themes for Category: {category_name} (Top {len(words_to_cluster)} words) ---")
        
        try:
            word_embeddings = model.encode(words_to_cluster, show_progress_bar=False)
        except Exception as e:
            logging.error(f"Failed to encode words for category '{category_name}': {e}")
            continue

        # Estimate eps for DBSCAN
        estimated_eps = _estimate_dbscan_eps(word_embeddings, min_samples, metric='cosine')
        if estimated_eps is None: # Fallback if estimation failed
            logging.warning(f"Could not estimate eps for '{category_name}', using default 0.5.")
            estimated_eps = 0.5


        # Perform DBSCAN clustering
        dbscan_clusterer = DBSCAN(eps=estimated_eps, min_samples=min_samples, metric='cosine')
        try:
            cluster_labels = dbscan_clusterer.fit_predict(word_embeddings)
        except Exception as e:
            logging.error(f"DBSCAN clustering failed for category '{category_name}': {e}")
            continue

        # Group words by cluster
        num_discovered_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        logging.info(f"Discovered {num_discovered_clusters} themes (clusters) for '{category_name}'.")

        clustered_words = {}
        noise_words = []
        for word, label in zip(words_to_cluster, cluster_labels):
            if label == -1:
                noise_words.append(word)
            else:
                if label not in clustered_words:
                    clustered_words[label] = []
                clustered_words[label].append(word)
        
        sorted_cluster_ids = sorted(clustered_words.keys())

        for cluster_id in sorted_cluster_ids:
            words_in_cluster = clustered_words[cluster_id]
            logging.info(f"  Theme (Cluster {cluster_id}): {', '.join(words_in_cluster)}")
        
        if noise_words:
            logging.info(f"  Noise (words not in any theme): {', '.join(noise_words)}")
            
    logging.info("--- Word clustering for themes finished ---")


def main():
    # Ensure cache directory exists
    os.makedirs(os.path.dirname(GRIDSEARCH_RESULTS_PATH), exist_ok=True)

    df = load_dreams(CSV_PATH)
    if df.empty:
        logging.error("Loaded dataframe is empty. Exiting.")
        return

    df = analyze_and_clean(df)
    if df.empty or 'y' not in df.columns or 'embedding' not in df.columns:
        logging.error("DataFrame is unsuitable for modeling after cleaning (empty or missing 'y'/'embedding'). Exiting.")
        return
    
    # Drop rows where 'embedding' is NaN before splitting, as these cannot be used by the model
    df.dropna(subset=['embedding'], inplace=True)
    if df.empty:
        logging.error("DataFrame is empty after dropping NaN embeddings. Exiting.")
        return
        
    # Prepare data for modeling
    # Convert 'created_utc' to datetime for time-based split
    if 'created_utc' not in df.columns:
        logging.error("'created_utc' column missing, cannot perform time-based split. Consider random split or ensure column exists.")
        # Fallback to random split or raise error
        # For now, let's assume it exists or proceed with caution. A robust solution would handle this.
        # If we must proceed without it, a random split is an option:
        # X_train, X_test, y_train, y_test = train_test_split(
        #     np.vstack(df['embedding'].values), df['y'], test_size=0.2, random_state=RANDOM_STATE, stratify=df['y']
        # )
        # However, the request was for time-based.
        raise ValueError("'created_utc' column is required for time-based splitting but not found.")

    df['created_utc'] = pd.to_datetime(df['created_utc'])
    df = df.sort_values(by='created_utc')

    # Stack embeddings into a numpy array
    X_full = np.vstack(df['embedding'].values) # Embeddings for the entire dataset (after cleaning and NaN drop)
    y_full = df['y'] # Labels for the entire dataset

    # Split data: 80% train, 20% test based on time
    # Using train_test_split with shuffle=False after sorting by time achieves a time-based split
    # where earlier data is for training and later data for testing.
    split_index = int(len(df) * 0.8)
    
    X_train = X_full[:split_index]
    y_train = y_full.iloc[:split_index]
    X_test = X_full[split_index:]
    y_test = y_full.iloc[split_index:]

    if len(X_train) == 0 or len(X_test) == 0:
        logging.error("Training or testing set is empty after split. Check data size and split logic.")
        return

    logging.info(f"Training set size: {len(X_train)}, Test set size: {len(X_test)}")
    
    model = find_best_model(X_train, y_train) # find_best_model uses N_SPLITS for CV on training data

    evaluate_performance(model, X_test, y_test, target_names=CATEGORIES)

    # TF-IDF analysis on the full dataset for each category
    logging.info("--- Starting TF-IDF Analysis for each category on the full dataset ---")
    # The df for analyze_tfidf_for_category should be the one used to generate X_full and y_full
    # which is 'df' after cleaning, sorting, and dropping NaN embeddings.
    for category in CATEGORIES:
        analyze_tfidf_for_category(df.copy(), X_full, model, target_category=category, text_column_name='dream')

    # Calculate and save TF-IDF scores for all words per class
    if model: # Ensure model was trained
        calculate_and_save_class_tfidf_scores(df.copy(), X_full, model, text_column_name='dream')
        # Cluster top words for themes after TF-IDF scores are saved
        cluster_top_words_for_themes()
    
    logging.info("--- Script Finished ---")

if __name__ == '__main__':
    main()

    # Example usage of get_dream_scores
    logging.info("\n--- Example: Scoring a new dream ---")
    example_dream = "I dreamt I was flying over a city made of code, trying to find a bug in the system."
    scores = get_dream_scores(example_dream)
    if scores:
        logging.info(f"Scores for example dream ('{example_dream[:50]}...'): {scores}")
    else:
        logging.info("Could not get scores for the example dream.")

    logging.info("\n--- Example: Scoring the full dataset ---")
    df_with_scores = score_full_dataset()
    if df_with_scores is not None:
        logging.info("Successfully scored the full dataset.")
        # Display info for a few rows with scores
        score_cols = [f'score_{cat}' for cat in CATEGORIES] # Assuming CATEGORIES matches model classes
        # Check if score columns were actually added
        if all(col in df_with_scores.columns for col in score_cols):
             logging.info(f"Sample of DataFrame with scores (first 5 rows with non-NaN embeddings):\n{df_with_scores[df_with_scores[score_cols[0]].notna()].head()}")
        else:
            logging.warning("Score columns may not have been added correctly to all rows.")
            logging.info(f"Full dataset head:\n{df_with_scores.head()}")

    else:
        logging.info("Failed to score the full dataset.")
