# Project Methods Outline

## 1. Data Collection and Preprocessing
- Source of dream narratives (briefly describe, e.g., online forums, specific subreddits if applicable).
- Initial dataset characteristics: 22,897 dreams, associated metadata (e.g., author, timestamp, pre-assigned category labels, pre-computed embeddings).
    - Raw distribution: 'blue_collar': 11,656 dreams (7,892 unique authors); 'gig_worker': 7,444 dreams (4,926 unique authors); 'white_collar': 8,061 dreams (5,534 unique authors).
    - 15,688 unique authors in the raw dataset.
- Embedding generation: Pre-computed text embeddings (e.g., "text-embedding-3-large" via OpenAI API, or similar) were provided with the dataset.
- Data cleaning:
    - Handling of missing or malformed embeddings: 3,509 rows dropped due to embedding parsing errors.
    - Filtering criteria:
        - Exclusion of dreams associated with users belonging to multiple socioeconomic categories (i.e., `n_categories > 1`). This removed 3,961 records.
- Final dataset for analysis: 18,936 dreams after cleaning and filtering for `n_categories = 1`.

## 2. Exploratory Data Analysis (EDA)
- Descriptive statistics of the cleaned dataset (18,936 dreams):
    - Overall dream counts: 18,936.
    - Dream counts per socioeconomic category: 'blue_collar': 8,165; 'gig_worker': 4,814; 'white_collar': 5,957.
    - Unique author counts overall: 13,256.
    - Unique author counts per category: 'blue_collar': 5,787; 'gig_worker': 3,424; 'white_collar': 4,045.
    - Temporal distribution of dreams (e.g., counts by year): Data spans from 2009 to 2025, with increasing dream counts in more recent years (e.g., 2023: 2,501 dreams; 2024: 3,222 dreams in the cleaned set).
- Rationale for focusing on single-category assignments: To ensure clear and unambiguous labeling for model training and analysis.

## 3. Socioeconomic Classification Model
- **Objective**: To develop a model capable of classifying dream narratives into one of three socioeconomic categories: 'blue_collar', 'gig_worker', or 'white_collar', based on their text embeddings.
- **Feature Engineering**:
    - Use of pre-computed text embeddings as input features.
- **Data Splitting**:
    - Chronological split of the 18,936 cleaned dreams: 15,148 for training (80%), 3,788 for testing (20%), based on `created_utc`.
- **Model Selection**:
    - Algorithm: Logistic Regression (multinomial).
    - Preprocessing within pipeline: StandardScaler.
    - Dimensionality Reduction (tested): Principal Component Analysis (PCA).
    - Hyperparameter Optimization: GridSearchCV with 5-fold stratified cross-validation on the training set.
        - Parameters tuned for Logistic Regression (e.g., C, penalty).
        - Parameters tuned for PCA (e.g., n_components).
    - Evaluation Metric for GridSearchCV: Weighted One-vs-Rest (OvR) ROC AUC.
    - Caching: Best model pipeline cached to `cache/gridsearch_results.joblib`.
- **Model Evaluation (on the test set)**:
    - Overall Accuracy: 0.43.
    - Weighted OvR ROC AUC Score: 0.6025.
    - Per-class performance (Weighted Avg): Precision: 0.44, Recall: 0.43, F1-score: 0.43.
        - Blue Collar: P=0.51, R=0.36, F1=0.42
        - Gig Worker: P=0.41, R=0.53, F1=0.46
        - White Collar: P=0.36, R=0.40, F1=0.38
    - Confusion Matrix (raw and normalized) and per-class ROC AUC curves (plotted and saved to `cache/roc_auc_curves.png`) were generated for detailed error analysis.

## 4. Identifying Differentiating Language (TF-IDF Analysis)
- **Objective**: To identify words whose usage frequency significantly differs between dreams predicted to belong to a specific socioeconomic category versus dreams predicted to belong to other categories.
- **Text Preprocessing**:
    - Tokenization (NLTK `word_tokenize`).
    - Conversion to lowercase.
    - Removal of stopwords (NLTK English list).
    - Removal of non-alphabetic tokens.
- **TF-IDF Vectorization**:
    - Applied to the processed dream texts from the full cleaned dataset (18,936 dreams).
    - `max_features` set to 5000.
- **Methodology**:
    - For each socioeconomic category (e.g., 'blue_collar'):
        - Group 1: Dreams predicted by the trained model to belong to this target category (e.g., 6,416 for 'blue_collar').
        - Group 2: Dreams predicted by the trained model to belong to any *other* category (e.g., 12,520 for 'blue_collar').
        - Calculate the mean TF-IDF score for each word within Group 1 and Group 2.
        - Calculate a difference score: `mean_tfidf_Group1 - mean_tfidf_Group2`.
    - Output: A list of all 5000 words with their mean TF-IDF scores for each predicted class and the calculated difference scores, saved to `cache/all_words_class_tfidf_scores.csv`.
- **LLM-based Theming (Initial Exploration)**:
    - Top 20 words with the highest positive difference score for each category were submitted to an LLM (GPT-4o via `simpleaichat`). For example, for 'blue_collar', top words included "wife, car, girlfriend, door, see, house..." which the LLM grouped into themes like "Domestic Life and Relationships" and "Vehicles and Movement".

## 5. Thematic Analysis of Differentiating Language (BERTopic)
- **Objective**: To systematically identify semantic themes among the words most characteristic of each socioeconomic category.
- **Input Word Selection**:
    - For each category, use the `diff_vs_others_<category>` scores from the TF-IDF analysis.
    - Filter words by minimum length (3 characters).
    - Lemmatization: Words lemmatized using NLTK `WordNetLemmatizer` with Part-of-Speech (POS) tagging.
    - Deduplication: For words sharing the same lemma, the original word form with the highest `diff_vs_others_<category>` score was retained.
    - Top 50 unique, highest-scoring, original word forms selected for each category.
- **BERTopic Configuration**:
    - Embedding Model: `all-mpnet-base-v2` (SentenceTransformer).
    - `min_topic_size`: Set to 3.
    - Applied independently to the selected top 50 words for each socioeconomic category.
- **Output**:
    - Identified themes (topics) for each category. For example, for 'blue_collar', 5 themes were found, including "Theme 0 (0_wife_wall_truck_town): wife, car, house, door, truck, garage, town, open, basement, road, wall" and an outlier group.
    - Results (category, topic ID, topic name, list of words) saved to `cache/bertopic_themes.csv`.

## 6. Theme Prevalence Analysis
- **Objective**: To quantify the presence of each identified BERTopic theme within the dream narratives across the entire dataset and within each actual socioeconomic category.
- **Methodology**:
    - For each theme identified by BERTopic:
        - The list of words constituting the theme was used.
        - A regular expression pattern was created to detect the presence of *any* of these theme words (case-insensitive, whole word match) within each dream text in the cleaned dataset.
    - **Calculations**:
        - Overall prevalence: Count and proportion of all dreams containing the theme.
        - Per-category prevalence: Count and proportion of dreams within each actual socioeconomic category (`y` label) containing the theme.
    - **Statistical Significance**:
        - For each theme and each actual socioeconomic category, a Chi-squared test of independence (2x2 contingency table: theme presence/absence vs. category membership/non-membership) was performed to assess if the theme's prevalence in that category was statistically significant compared to its prevalence in other categories. Yates' correction applied.
- **Output**: A CSV file (`cache/theme_prevalence_stats.csv`) containing, for each theme:
    - Its constituent words and original derived category.
    - Overall dream count and proportion.
    - Dream count and proportion for each actual socioeconomic category.
    - P-value from the Chi-squared test for each actual socioeconomic category.
