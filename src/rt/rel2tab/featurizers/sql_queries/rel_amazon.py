"""SQL feature queries for rel-amazon (Amazon Reviews) tasks.

Each query produces ~14 pre-normalized features designed for
few-shot linear prediction in the rel2tab pipeline.

Database tables:
  - customer: customer_id, customer_name
  - product:  product_id, category, brand, title, description, price
  - review:   review_time, customer_id, product_id, rating, verified, review_text, summary

Tasks (user-level):
  - user-churn: (timestamp, customer_id) -> binary
  - user-ltv:   (timestamp, customer_id) -> regression
Tasks (item-level):
  - item-churn: (timestamp, product_id)  -> binary
  - item-ltv:   (timestamp, product_id)  -> regression
"""

# ---------------------------------------------------------------------------
# User-level features (shared by user-churn and user-ltv)
# 14 features: recency, reviews_30d, reviews_90d, reviews_total, avg_rating,
#   rating_std, count_low_rating, verified_rate, avg_price, spend_1y,
#   spend_total, log_unique_products, categories, trend, is_new_user
# ---------------------------------------------------------------------------

USER_FEATURES_SQL = """
WITH task AS (
    SELECT timestamp, customer_id FROM task_table
),
user_reviews AS (
    SELECT
        t.timestamp AS task_ts,
        t.customer_id,
        r.rating,
        r.verified,
        r.product_id,
        DATE_DIFF('day', r.review_time, t.timestamp) AS days_ago
    FROM task t
    JOIN review r
        ON r.customer_id = t.customer_id
        AND r.review_time < t.timestamp
),
user_reviews_with_product AS (
    SELECT ur.*, p.price, p.category
    FROM user_reviews ur
    LEFT JOIN product p ON p.product_id = ur.product_id
),
user_agg AS (
    SELECT
        task_ts, customer_id,

        -- Recency
        MIN(days_ago) AS days_since_last_review,

        -- Volume
        COUNT(*) FILTER (WHERE days_ago <= 30) AS reviews_30d,
        COUNT(*) FILTER (WHERE days_ago <= 90) AS reviews_90d,
        COUNT(*) AS reviews_total,

        -- Ratings
        AVG(rating) AS avg_rating,
        STDDEV_POP(rating) AS rating_std,
        COUNT(*) FILTER (WHERE rating <= 2)::DOUBLE
            / NULLIF(COUNT(*), 0) AS low_rating_ratio,
        AVG(CASE WHEN verified THEN 1.0 ELSE 0.0 END) AS verified_rate,

        -- Monetary
        AVG(price) FILTER (WHERE price > 0) AS avg_price,
        SUM(price) FILTER (WHERE price > 0 AND days_ago <= 365) AS spend_1y,
        SUM(price) FILTER (WHERE price > 0) AS spend_total,

        -- Diversity
        COUNT(DISTINCT product_id) AS unique_products,
        COUNT(DISTINCT category) AS unique_categories,

        -- Trend
        COUNT(*) FILTER (WHERE days_ago <= 90)
            - COUNT(*) FILTER (WHERE days_ago > 90 AND days_ago <= 180) AS trend_90d

    FROM user_reviews_with_product
    GROUP BY task_ts, customer_id
)
SELECT
    t.timestamp,
    t.customer_id,

    -- 1. Recency (exponential decay, half-life ~60 days)
    EXP(-COALESCE(ua.days_since_last_review, 9999) / 87.0)
                                                        AS review_recency,

    -- 2. Reviews in last 30d (log-scaled, cap at ~20)
    LEAST(LN(1 + COALESCE(ua.reviews_30d, 0)) / LN(21), 1.0)
                                                        AS log_reviews_30d_norm,

    -- 3. Reviews in last 90d (log-scaled, cap at ~30)
    LEAST(LN(1 + COALESCE(ua.reviews_90d, 0)) / LN(31), 1.0)
                                                        AS log_reviews_90d_norm,

    -- 4. Total reviews (log-scaled, cap at ~500)
    LEAST(LN(1 + COALESCE(ua.reviews_total, 0)) / LN(501), 1.0)
                                                        AS log_reviews_total_norm,

    -- 5. Average rating (normalized to [0,1] from 1-5 scale)
    (COALESCE(ua.avg_rating, 3.0) - 1.0) / 4.0
                                                        AS avg_rating_norm,

    -- 6. Rating standard deviation (normalized, max stddev ~2.0)
    LEAST(COALESCE(ua.rating_std, 0.0) / 2.0, 1.0)
                                                        AS rating_std_norm,

    -- 7. Low rating ratio (fraction of 1-2 star reviews, [0,1])
    COALESCE(ua.low_rating_ratio, 0.0)                 AS low_rating_ratio,

    -- 8. Verified rate (already [0,1])
    COALESCE(ua.verified_rate, 0.5)                    AS verified_rate,

    -- 9. Average price (log-scaled, cap at ~500)
    LEAST(LN(1 + COALESCE(ua.avg_price, 0)) / LN(501), 1.0)
                                                        AS log_avg_price_norm,

    -- 10. Annual spend (log-scaled, cap at ~5000)
    LEAST(LN(1 + COALESCE(ua.spend_1y, 0)) / LN(5001), 1.0)
                                                        AS log_spend_1y_norm,

    -- 11. Total lifetime spend (log-scaled, cap at ~10000)
    LEAST(LN(1 + COALESCE(ua.spend_total, 0)) / LN(10001), 1.0)
                                                        AS log_spend_total_norm,

    -- 12. Product diversity (log unique products, cap at ~200)
    LEAST(LN(1 + COALESCE(ua.unique_products, 0)) / LN(201), 1.0)
                                                        AS log_unique_products_norm,

    -- 13. Category diversity (log-scaled, cap at ~50)
    LEAST(LN(1 + COALESCE(ua.unique_categories, 0)) / LN(51), 1.0)
                                                        AS log_categories_norm,

    -- 14. Activity trend (90d vs prev 90d, shifted to [0,1])
    LEAST(GREATEST(
        0.5 + COALESCE(ua.trend_90d, 0) / 20.0,
        0.0), 1.0)                                     AS trend_90d_norm,

    -- 15. Is new user (no review history, binary)
    CASE WHEN ua.customer_id IS NULL THEN 1.0 ELSE 0.0 END
                                                        AS is_new_user

FROM task t
LEFT JOIN user_agg ua
    ON ua.customer_id = t.customer_id AND ua.task_ts = t.timestamp
"""

USER_CHURN_SQL = USER_FEATURES_SQL
USER_LTV_SQL = USER_FEATURES_SQL


# ---------------------------------------------------------------------------
# Item-level features (shared by item-churn and item-ltv)
# 14 features: recency, reviews_90d, reviews_total, avg_rating, rating_std,
#   verified_rate, unique_reviewers, unique_reviewers_1y, price,
#   estimated_revenue_1y, item_age, review_trend_30d, trend_90d, is_new_product
# ---------------------------------------------------------------------------

ITEM_FEATURES_SQL = """
WITH task AS (
    SELECT timestamp, product_id FROM task_table
),
product_info AS (
    SELECT product_id, price, category, brand
    FROM product
),
item_reviews AS (
    SELECT
        t.timestamp AS task_ts,
        t.product_id,
        r.rating,
        r.verified,
        r.customer_id,
        DATE_DIFF('day', r.review_time, t.timestamp) AS days_ago
    FROM task t
    JOIN review r
        ON r.product_id = t.product_id
        AND r.review_time < t.timestamp
),
item_reviews_with_price AS (
    SELECT ir.*, p.price
    FROM item_reviews ir
    LEFT JOIN product p ON p.product_id = ir.product_id
),
item_agg AS (
    SELECT
        task_ts, product_id,

        -- Recency
        MIN(days_ago) AS days_since_last_review,

        -- Volume
        COUNT(*) FILTER (WHERE days_ago <= 30) AS reviews_30d,
        COUNT(*) FILTER (WHERE days_ago <= 90) AS reviews_90d,
        COUNT(*) AS reviews_total,

        -- Ratings
        AVG(rating) AS avg_rating,
        STDDEV_POP(rating) AS rating_std,
        AVG(CASE WHEN verified THEN 1.0 ELSE 0.0 END) AS verified_rate,

        -- Customer diversity
        COUNT(DISTINCT customer_id) AS unique_reviewers,
        COUNT(DISTINCT customer_id) FILTER (WHERE days_ago <= 365) AS unique_reviewers_1y,

        -- Revenue proxy (sum of product price for each review in last year)
        SUM(price) FILTER (WHERE days_ago <= 365) AS estimated_revenue_1y,

        -- Item age
        MAX(days_ago) AS item_age_days,

        -- Trend: 30d vs prev 30d (finer than 90d)
        COUNT(*) FILTER (WHERE days_ago <= 30)
            - COUNT(*) FILTER (WHERE days_ago > 30 AND days_ago <= 60) AS review_trend_30d,

        -- Trend: 90d vs prev 90d
        COUNT(*) FILTER (WHERE days_ago <= 90)
            - COUNT(*) FILTER (WHERE days_ago > 90 AND days_ago <= 180) AS trend_90d

    FROM item_reviews_with_price
    GROUP BY task_ts, product_id
)
SELECT
    t.timestamp,
    t.product_id,

    -- 1. Recency (exponential decay, half-life ~60 days)
    EXP(-COALESCE(ia.days_since_last_review, 9999) / 87.0)
                                                        AS review_recency,

    -- 2. Recent review count (log-scaled, cap at ~30)
    LEAST(LN(1 + COALESCE(ia.reviews_90d, 0)) / LN(31), 1.0)
                                                        AS log_reviews_90d_norm,

    -- 3. Total reviews (log-scaled, cap at ~1000)
    LEAST(LN(1 + COALESCE(ia.reviews_total, 0)) / LN(1001), 1.0)
                                                        AS log_reviews_total_norm,

    -- 4. Average rating (normalized to [0,1] from 1-5 scale)
    (COALESCE(ia.avg_rating, 3.0) - 1.0) / 4.0
                                                        AS avg_rating_norm,

    -- 5. Rating standard deviation (normalized, max stddev ~2.0)
    LEAST(COALESCE(ia.rating_std, 0.0) / 2.0, 1.0)
                                                        AS rating_std_norm,

    -- 6. Verified review rate (already [0,1])
    COALESCE(ia.verified_rate, 0.5)                    AS verified_rate,

    -- 7. Customer diversity - all time (log-scaled, cap at ~500)
    LEAST(LN(1 + COALESCE(ia.unique_reviewers, 0)) / LN(501), 1.0)
                                                        AS log_reviewers_norm,

    -- 8. Customer diversity - last year (log-scaled, cap at ~200)
    LEAST(LN(1 + COALESCE(ia.unique_reviewers_1y, 0)) / LN(201), 1.0)
                                                        AS log_reviewers_1y_norm,

    -- 9. Product price (log-scaled, cap at ~500)
    LEAST(LN(1 + COALESCE(pi.price, 0)) / LN(501), 1.0)
                                                        AS log_price_norm,

    -- 10. Estimated revenue in last year (log-scaled, cap at ~50000)
    LEAST(LN(1 + COALESCE(ia.estimated_revenue_1y, 0)) / LN(50001), 1.0)
                                                        AS log_est_revenue_1y_norm,

    -- 11. Item age (log-scaled, cap at ~2000 days)
    LEAST(LN(1 + COALESCE(ia.item_age_days, 0)) / LN(2001), 1.0)
                                                        AS log_item_age_norm,

    -- 12. Review trend 30d (30d vs prev 30d, shifted to [0,1])
    LEAST(GREATEST(
        0.5 + COALESCE(ia.review_trend_30d, 0) / 10.0,
        0.0), 1.0)                                     AS review_trend_30d_norm,

    -- 13. Activity trend 90d (90d vs prev 90d, shifted to [0,1])
    LEAST(GREATEST(
        0.5 + COALESCE(ia.trend_90d, 0) / 20.0,
        0.0), 1.0)                                     AS trend_90d_norm,

    -- 14. Is new product (no review history, binary)
    CASE WHEN ia.product_id IS NULL THEN 1.0 ELSE 0.0 END
                                                        AS is_new_product

FROM task t
LEFT JOIN product_info pi ON pi.product_id = t.product_id
LEFT JOIN item_agg ia
    ON ia.product_id = t.product_id AND ia.task_ts = t.timestamp
"""

ITEM_CHURN_SQL = ITEM_FEATURES_SQL
ITEM_LTV_SQL = ITEM_FEATURES_SQL
