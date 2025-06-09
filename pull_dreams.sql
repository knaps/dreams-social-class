DROP TABLE IF EXISTS reddit.tmp_emily;

CREATE TABLE reddit.tmp_emily AS (
    WITH categorized_subreddits AS (
        SELECT 
            'Blue-collar' AS category, unnest(ARRAY[
                'SkilledTrades', 'Construction', 'Truckers', 'Carpentry', 'Welding',
                'Electricians', 'Plumbing', 'Lineman', 'BlueCollarWomen', 'MechanicAdvice', 'HVAC'
            ]) AS subreddit
        UNION ALL
        SELECT 
            'Gig-worker', unnest(ARRAY[
                'UberDrivers', 'doordash', 'InstacartShoppers', 'amazonflexdrivers', 'deliveries',
                'TaskRabbit', 'PoplinLaundryPros', 'UberEatsDrivers', 'deliveroos', 'ShiptShoppers'
            ])
        UNION ALL
        SELECT 
            'White-collar', unnest(ARRAY[
                'cscareerquestions', 'Accounting', 'humanresources', 'UXDesign', 'consulting', 'MBA',
                'FinancialCareers', 'Big4', 'TaxPros', 'ITCareerQuestions', 'dataengineering',
                'businessanalysis', 'productmanagement', 'architecture', 'academia', 'actuary'
            ])
    ),
    user_category_map AS (
        SELECT 
            users.user_id,
            c.category
        FROM reddit.subreddit_membership srm
        JOIN reddit.subreddits sr ON srm.subreddit_id = sr.id
        JOIN reddit.user_id_mapping users ON srm.user_id = users.user_id
        JOIN categorized_subreddits c ON sr.subreddit = c.subreddit
    ),
    dreams_raw AS (
        SELECT
            dreams.id AS dream_id,
            dreams.created_utc,
            dreams.dream,
            dreams.embedding,
            users.user_id,
            users.username AS author,
            ucm.category
        FROM reddit.r_dreams_v3 dreams
        JOIN reddit.user_id_mapping users ON dreams.author = users.username
        JOIN user_category_map ucm ON ucm.user_id = users.user_id
        WHERE dreams.dream IS NOT NULL AND length(dreams.dream) > 0
    ),
    deduplicated_dreams AS (
        SELECT *
        FROM (
            SELECT *,
                   ROW_NUMBER() OVER (PARTITION BY dream ORDER BY created_utc ASC) AS rn
            FROM dreams_raw
        ) t
        WHERE rn = 1
    ),
    category_flags AS (
        SELECT
            dream_id,
            MAX(CASE WHEN category = 'Blue-collar' THEN 1 ELSE 0 END) AS Blue_collar,
            MAX(CASE WHEN category = 'Gig-worker' THEN 1 ELSE 0 END) AS Gig_worker,
            MAX(CASE WHEN category = 'White-collar' THEN 1 ELSE 0 END) AS White_collar
        FROM dreams_raw
        GROUP BY dream_id
    ),
    earliest_dreams AS (
        SELECT DISTINCT ON (dream_id)
            dream_id,
            created_utc,
            dream,
            embedding,
            author
        FROM deduplicated_dreams
        ORDER BY dream_id, created_utc
    )
    SELECT 
        e.dream_id,
        e.created_utc,
        e.dream,
        e.embedding,
        e.author,
        c.Blue_collar,
        c.Gig_worker,
        c.White_collar
    FROM earliest_dreams e
    JOIN category_flags c ON e.dream_id = c.dream_id
);
