# Project Methods Outline

## 1. Data Collection and Preprocessing
    - Source of dream narratives (briefly describe, e.g., online forums, specific subreddits if applicable).
    - Initial dataset characteristics: number of dreams, associated metadata (e.g., author, timestamp, pre-assigned category labels, pre-computed embeddings).
    - Embedding generation: Mention the model used for pre-computed embeddings (e.g., "text-embedding-3-large" via OpenAI API if this was the source, or describe if generated otherwise).
    - Data cleaning:
        - Handling of missing or malformed embeddings.
        - Filtering criteria:
            - Exclusion of dreams associated with users belonging to multiple socioeconomic categories (i.e., `n_categories > 1`).
    - Final dataset for analysis: Number of dreams after cleaning.

## 2. Exploratory Data Analysis (EDA)
    - Descriptive statistics of the cleaned dataset:
        - Overall dream counts.
        - Dream counts per socioeconomic category.
        - Unique author counts overall and per category.
        - Temporal distribution of dreams (e.g., counts by year).
    - Rationale for focusing on single-category assignments.

## 3. Socioeconomic Classification Model
    - **Objective**: To develop a model capable of classifying dream narratives into one of three socioeconomic categories: 'blue_collar', 'gig_worker', or 'white_collar', based on their text embeddings.
    - **Feature Engineering**:
        - Use of pre-computed text embeddings as input features.
    - **Data Splitting**:
        - Chronological split: 80% for training, 20% for testing, based on `created_utc` to simulate a realistic prediction scenario.
    - **Model Selection**:
        - Algorithm: Logistic Regression (multinomial).
        - Preprocessing within pipeline: StandardScaler.
        - Dimensionality Reduction (tested): Principal Component Analysis (PCA).
        - Hyperparameter Optimization: GridSearchCV with 5-fold stratified cross-validation on the training set.
            - Parameters tuned for Logistic Regression (e.g., C, penalty).
            - Parameters tuned for PCA (e.g., n_components).
        - Evaluation Metric for GridSearchCV: Weighted One-vs-Rest (OvR) ROC AUC.
        - Caching: Best model pipeline cached to avoid re-computation.
    - **Model Evaluation (on the test set)**:
        - Classification Report (precision, recall, F1-score per class, accuracy).
        - Confusion Matrix (raw and normalized).
        - Weighted OvR ROC AUC score.
        - Per-class ROC AUC curves (plotted and saved).

## 4. Identifying Differentiating Language (TF-IDF Analysis)
    - **Objective**: To identify words whose usage frequency significantly differs between dreams predicted to belong to a specific socioeconomic category versus dreams predicted to belong to other categories.
    - **Text Preprocessing**:
        - Tokenization.
        - Conversion to lowercase.
        - Removal of stopwords (standard English list).
        - Removal of non-alphabetic tokens.
    - **TF-IDF Vectorization**:
        - Applied to the processed dream texts from the full cleaned dataset.
        - `max_features` set to 5000.
    - **Methodology**:
        - For each socioeconomic category (e.g., 'blue_collar'):
            - Group 1: Dreams predicted by the trained model to belong to this target category.
            - Group 2: Dreams predicted by the trained model to belong to any *other* category.
            - Calculate the mean TF-IDF score for each word within Group 1 and Group 2.
            - Calculate a difference score: `mean_tfidf_Group1 - mean_tfidf_Group2`.
        - Output: A list of all words with their mean TF-IDF scores for each predicted class and the calculated difference scores, saved to a CSV file (`all_words_class_tfidf_scores.csv`).
    - **LLM-based Theming (Initial Exploration)**:
        - Top 20 words with the highest positive difference score for each category were submitted to an LLM (GPT-4o via `simpleaichat`) to generate initial semantic themes.

## 5. Thematic Analysis of Differentiating Language (BERTopic)
    - **Objective**: To systematically identify semantic themes among the words most characteristic of each socioeconomic category.
    - **Input Word Selection**:
        - For each category, use the `diff_vs_others_<category>` scores from the TF-IDF analysis.
        - Filter words by minimum length (3 characters).
        - Lemmatization: Words lemmatized using NLTK `WordNetLemmatizer` with Part-of-Speech (POS) tagging to handle word forms (e.g., plurals, verb tenses).
        - Deduplication: For words sharing the same lemma, the original word form with the highest `diff_vs_others_<category>` score was retained.
        - Top 50 unique, highest-scoring, original word forms selected for each category.
    - **BERTopic Configuration**:
        - Embedding Model: `all-mpnet-base-v2` (SentenceTransformer).
        - `min_topic_size`: Set to 3 (minimum number of words to form a topic/theme).
        - Applied independently to the selected top 50 words for each socioeconomic category.
    - **Output**:
        - Identified themes (topics) for each category, consisting of clusters of semantically similar words.
        - Results (category, topic ID, topic name, list of words) saved to a CSV file (`bertopic_themes.csv`).

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
    - **Output**: A CSV file (`theme_prevalence_stats.csv`) containing, for each theme:
        - Its constituent words and original derived category.
        - Overall dream count and proportion.
        - Dream count and proportion for each actual socioeconomic category.
        - P-value from the Chi-squared test for each actual socioeconomic category.
