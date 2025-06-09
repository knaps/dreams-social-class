import pandas as pd
from dotenv import load_dotenv

load_dotenv()

DB_URL = os.getenv("DB_CONN_STR")

# --- Download NLTK data ---
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    nltk.download('stopwords')

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# Silence noisy HTTPX logs from simpleaichat/openai
logging.getLogger("httpx").setLevel(logging.WARNING)
CSV_PATH = 'socioeconomic_dreams.csv'
ENV_PATH = '.env'
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

def load_dreams(path:str):
    logging.info(f"Loading data from {path}...")
    try:
        df = pd.read_csv(df, index_col=0)
        # Ensure 'embedding' column is parsed correctly if it's stored as a string
        if 'embedding' in df.columns and isinstance(df['embedding'].iloc[0], str):
             logging.info("Parsing string embeddings in synesthesia data...")
             df['embedding'] = df['embedding'].apply(parse_embedding)
             # Drop rows where embedding parsing failed
             original_len = len(df)
             df = df.dropna(subset=['embedding'])
             if len(df) < original_len:
                 logging.warning(f"Dropped {original_len - len(df)} rows due to embedding parsing errors.")

    df = pd.read_csv(path)
    df['n_categories']=df[['blue_collar','gig_worker','white_collar']].sum(axis=1)
    return df

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

    # then cleanup:
    # remove records where n_categories>1

    # then re-run the above stats with new dataset

    # create multiclass y var
    df['y'] = df[['blue_collar', 'gig_worker', 'white_collar']].idxmax(axis=1)
    # return the cleaned dataframe
    return df

def cleanup(df:pd.DataFrame):
    '''Prepare dream dataset for modeling, logging changes to the dataframe'''
    pass
    # remove the dreams of users who belong to more than one dataset

def analyze_tfidf_deciles(df):
    """ Performs TF-IDF analysis on top vs bottom deciles and asks LLM for themes. """

    # TODO: fix this so that it handles the multiclass properly

    logging.info("Running TF-IDF analysis on top/bottom deciles...")
    if 'score_decile' not in df.columns:
        logging.error("DataFrame must have 'score_decile' column for TF-IDF analysis.")
        return

    try:
        # Preprocessing
        stop_words = set(stopwords.words('english'))
        # Ensure text column exists and handle potential NaN values
        if 'text' not in df.columns:
             logging.error("DataFrame must have 'text' column.")
             return
        df['processed_text'] = df['text'].fillna('').apply(
            lambda x: ' '.join([word.lower() for word in word_tokenize(str(x)) if word.isalpha() and word.lower() not in stop_words])
        )

        # Fit TF-IDF Vectorizer
        vectorizer = TfidfVectorizer(max_features=5000) # Limit features for performance
        tfidf_matrix = vectorizer.fit_transform(df['processed_text'])
        feature_names = vectorizer.get_feature_names_out()

        # Get indices for top and bottom deciles
        bottom_decile_indices = df[df['score_decile'] == 0].index
        top_decile_indices = df[df['score_decile'] == df['score_decile'].max()].index # Use max() in case of fewer than 10 deciles

        if len(bottom_decile_indices) == 0 or len(top_decile_indices) == 0:
            logging.warning("Not enough data in top or bottom deciles for TF-IDF analysis.")
            return

        # Calculate mean TF-IDF for each decile
        bottom_tfidf_mean = np.array(tfidf_matrix[bottom_decile_indices].mean(axis=0)).flatten()
        top_tfidf_mean = np.array(tfidf_matrix[top_decile_indices].mean(axis=0)).flatten()

        # Create comparison DataFrame
        comparison_df = pd.DataFrame({
            'feature': feature_names,
            'bottom_mean_tfidf': bottom_tfidf_mean,
            'top_mean_tfidf': top_tfidf_mean
        })
        comparison_df['diff_top_bottom'] = comparison_df['top_mean_tfidf'] - comparison_df['bottom_mean_tfidf']

        # Get top N differentiating words
        n_words = 50
        top_words = comparison_df.sort_values('diff_top_bottom', ascending=False).head(n_words)['feature'].tolist()
        bottom_words = comparison_df.sort_values('diff_top_bottom', ascending=True).head(n_words)['feature'].tolist()

        logging.info(f"Top {n_words} words associated with HIGH scores (Synesthesia):\n{', '.join(top_words)}")
        logging.info(f"Top {n_words} words associated with LOW scores (Baseline):\n{', '.join(bottom_words)}")

        # LLM Analysis for Theming
        try:
            ai = AIChat(console=False, model='gpt-4o', api_key=os.getenv("OPENAI_API_KEY"))
            prompt_top = f"""The following words are most strongly associated with dreams from subreddit A (compared to subreddit B). Please group these words into meaningful semantic themes (3-5 themes usually works well). Provide only the theme names and the words belonging to each theme.

Words: {', '.join(top_words)}"""
            response_top = ai(prompt_top)
            logging.info(f"LLM Theming for HIGH score words:\n{response_top}")

            prompt_bottom = f"""The following words are most strongly associated with dreams from subreddit B (compared to subreddit A). Please group these words into meaningful semantic themes (3-5 themes usually works well). Provide only the theme names and the words belonging to each theme.

Words: {', '.join(bottom_words)}"""
            response_bottom = ai(prompt_bottom)
            logging.info(f"LLM Theming for LOW score words:\n{response_bottom}")

        except ImportError:
            logging.warning("simpleaichat not installed. Skipping LLM theming for TF-IDF words.")
        except Exception as e:
            logging.error(f"Error during LLM theming for TF-IDF words: {e}")

    except Exception as e:
        logging.error(f"Error during TF-IDF analysis: {e}")

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
                logging.info(f"*** New overall best model found: {name} (Score: {best_score:.4f}) ***")

        except Exception as e:
            logging.error(f"GridSearchCV failed for {name}: {e}")

    if best_pipeline is None:
        logging.error("GridSearchCV failed to find any valid model.")
        raise RuntimeError("Model optimization failed.")

    logging.info(f"--- GridSearchCV Finished ---")
    logging.info(f"Overall best pipeline score ({scoring}): {best_score:.4f}")
    logging.info(f"Overall best pipeline configuration: {best_pipeline}")

    try:
        logging.info(f"Saving the best pipeline to cache: {GRIDSEARCH_RESULTS_PATH}")
        joblib.dump(best_pipeline, GRIDSEARCH_RESULTS_PATH)
    except Exception as e:
        logging.error(f"Failed to save best pipeline to cache: {e}")

    return best_pipeline

def evaluate_performance(model):
    '''Demonstrate how this model performs in putting people into each class'''
    pass
def main():
    df = load_dreams(CSV_PATH)
    df = analyze_and_clean(df)
    # split data into 80% train %20 test - using the last 20% of created_utc records as test
    # train, test = 
    pass
    model = find_best_model(X=df['embedding'], y=df['y'])
    evaluate_performance(model)
    # do analyze_tfidf_deciles but need to adapt it for this multiclass. Get top decile of scores for each class and do the analysis
    # might need to do some processing here to get the score deciles for each class etc.
    analyze_tfidf_deciles()
    pass
