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

        MIN(days_ago) AS days_since_last_review,

        COUNT(*) FILTER (WHERE days_ago <= 30) AS reviews_30d,
        COUNT(*) FILTER (WHERE days_ago <= 90) AS reviews_90d,
        COUNT(*) AS reviews_total,

        AVG(rating) AS avg_rating,
        STDDEV_POP(rating) AS rating_std,
        COUNT(*) FILTER (WHERE rating <= 2)::DOUBLE
            / NULLIF(COUNT(*), 0) AS low_rating_ratio,
        AVG(CASE WHEN verified THEN 1.0 ELSE 0.0 END) AS verified_rate,

        AVG(price) FILTER (WHERE price > 0) AS avg_price,
        SUM(price) FILTER (WHERE price > 0 AND days_ago <= 365) AS spend_1y,
        SUM(price) FILTER (WHERE price > 0) AS spend_total,

        COUNT(DISTINCT product_id) AS unique_products,
        COUNT(DISTINCT category) AS unique_categories,

        COUNT(*) FILTER (WHERE days_ago <= 90)
            - COUNT(*) FILTER (WHERE days_ago > 90 AND days_ago <= 180) AS trend_90d

    FROM user_reviews_with_product
    GROUP BY task_ts, customer_id
)
SELECT
    t.timestamp,
    t.customer_id,

    EXP(-COALESCE(ua.days_since_last_review, 9999) / 87.0)
                                                        AS review_recency,

    LEAST(LN(1 + COALESCE(ua.reviews_30d, 0)) / LN(21), 1.0)
                                                        AS log_reviews_30d_norm,

    LEAST(LN(1 + COALESCE(ua.reviews_90d, 0)) / LN(31), 1.0)
                                                        AS log_reviews_90d_norm,

    LEAST(LN(1 + COALESCE(ua.reviews_total, 0)) / LN(501), 1.0)
                                                        AS log_reviews_total_norm,

    (COALESCE(ua.avg_rating, 3.0) - 1.0) / 4.0
                                                        AS avg_rating_norm,

    LEAST(COALESCE(ua.rating_std, 0.0) / 2.0, 1.0)
                                                        AS rating_std_norm,

    COALESCE(ua.low_rating_ratio, 0.0)                 AS low_rating_ratio,

    COALESCE(ua.verified_rate, 0.5)                    AS verified_rate,

    LEAST(LN(1 + COALESCE(ua.avg_price, 0)) / LN(501), 1.0)
                                                        AS log_avg_price_norm,

    LEAST(LN(1 + COALESCE(ua.spend_1y, 0)) / LN(5001), 1.0)
                                                        AS log_spend_1y_norm,

    LEAST(LN(1 + COALESCE(ua.spend_total, 0)) / LN(10001), 1.0)
                                                        AS log_spend_total_norm,

    LEAST(LN(1 + COALESCE(ua.unique_products, 0)) / LN(201), 1.0)
                                                        AS log_unique_products_norm,

    LEAST(LN(1 + COALESCE(ua.unique_categories, 0)) / LN(51), 1.0)
                                                        AS log_categories_norm,

    LEAST(GREATEST(
        0.5 + COALESCE(ua.trend_90d, 0) / 20.0,
        0.0), 1.0)                                     AS trend_90d_norm,

    CASE WHEN ua.customer_id IS NULL THEN 1.0 ELSE 0.0 END
                                                        AS is_new_user

FROM task t
LEFT JOIN user_agg ua
    ON ua.customer_id = t.customer_id AND ua.task_ts = t.timestamp
"""

USER_CHURN_SQL = USER_FEATURES_SQL
USER_LTV_SQL = USER_FEATURES_SQL


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

        MIN(days_ago) AS days_since_last_review,

        COUNT(*) FILTER (WHERE days_ago <= 30) AS reviews_30d,
        COUNT(*) FILTER (WHERE days_ago <= 90) AS reviews_90d,
        COUNT(*) AS reviews_total,

        AVG(rating) AS avg_rating,
        STDDEV_POP(rating) AS rating_std,
        AVG(CASE WHEN verified THEN 1.0 ELSE 0.0 END) AS verified_rate,

        COUNT(DISTINCT customer_id) AS unique_reviewers,
        COUNT(DISTINCT customer_id) FILTER (WHERE days_ago <= 365) AS unique_reviewers_1y,

        SUM(price) FILTER (WHERE days_ago <= 365) AS estimated_revenue_1y,

        MAX(days_ago) AS item_age_days,

        COUNT(*) FILTER (WHERE days_ago <= 30)
            - COUNT(*) FILTER (WHERE days_ago > 30 AND days_ago <= 60) AS review_trend_30d,

        COUNT(*) FILTER (WHERE days_ago <= 90)
            - COUNT(*) FILTER (WHERE days_ago > 90 AND days_ago <= 180) AS trend_90d

    FROM item_reviews_with_price
    GROUP BY task_ts, product_id
)
SELECT
    t.timestamp,
    t.product_id,

    EXP(-COALESCE(ia.days_since_last_review, 9999) / 87.0)
                                                        AS review_recency,

    LEAST(LN(1 + COALESCE(ia.reviews_90d, 0)) / LN(31), 1.0)
                                                        AS log_reviews_90d_norm,

    LEAST(LN(1 + COALESCE(ia.reviews_total, 0)) / LN(1001), 1.0)
                                                        AS log_reviews_total_norm,

    (COALESCE(ia.avg_rating, 3.0) - 1.0) / 4.0
                                                        AS avg_rating_norm,

    LEAST(COALESCE(ia.rating_std, 0.0) / 2.0, 1.0)
                                                        AS rating_std_norm,

    COALESCE(ia.verified_rate, 0.5)                    AS verified_rate,

    LEAST(LN(1 + COALESCE(ia.unique_reviewers, 0)) / LN(501), 1.0)
                                                        AS log_reviewers_norm,

    LEAST(LN(1 + COALESCE(ia.unique_reviewers_1y, 0)) / LN(201), 1.0)
                                                        AS log_reviewers_1y_norm,

    LEAST(LN(1 + COALESCE(pi.price, 0)) / LN(501), 1.0)
                                                        AS log_price_norm,

    LEAST(LN(1 + COALESCE(ia.estimated_revenue_1y, 0)) / LN(50001), 1.0)
                                                        AS log_est_revenue_1y_norm,

    LEAST(LN(1 + COALESCE(ia.item_age_days, 0)) / LN(2001), 1.0)
                                                        AS log_item_age_norm,

    LEAST(GREATEST(
        0.5 + COALESCE(ia.review_trend_30d, 0) / 10.0,
        0.0), 1.0)                                     AS review_trend_30d_norm,

    LEAST(GREATEST(
        0.5 + COALESCE(ia.trend_90d, 0) / 20.0,
        0.0), 1.0)                                     AS trend_90d_norm,

    CASE WHEN ia.product_id IS NULL THEN 1.0 ELSE 0.0 END
                                                        AS is_new_product

FROM task t
LEFT JOIN product_info pi ON pi.product_id = t.product_id
LEFT JOIN item_agg ia
    ON ia.product_id = t.product_id AND ia.task_ts = t.timestamp
"""

ITEM_CHURN_SQL = ITEM_FEATURES_SQL
ITEM_LTV_SQL = ITEM_FEATURES_SQL
