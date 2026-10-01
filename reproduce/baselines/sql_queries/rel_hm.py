USER_CHURN_SQL = """
WITH task AS (
    SELECT timestamp, customer_id FROM task_table
),
txn_base AS (
    SELECT
        t.timestamp AS task_ts, t.customer_id,
        txn.t_dat, txn.price, txn.sales_channel_id, txn.article_id,
        DATE_DIFF('day', txn.t_dat, t.timestamp) AS days_ago
    FROM task t
    JOIN transactions txn
        ON txn.customer_id = t.customer_id
        AND txn.t_dat < t.timestamp
),
gap_stats AS (
    SELECT task_ts, customer_id, MAX(gap_days) AS max_gap
    FROM (
        SELECT task_ts, customer_id,
            days_ago - LEAD(days_ago) OVER (
                PARTITION BY task_ts, customer_id ORDER BY t_dat
            ) AS gap_days
        FROM txn_base
    ) g
    WHERE gap_days IS NOT NULL
    GROUP BY task_ts, customer_id
),
cust_agg AS (
    SELECT
        task_ts, customer_id,
        MIN(days_ago) AS days_since_last,
        COUNT(*) FILTER (WHERE days_ago <= 28) AS txn_28d,
        COUNT(*) FILTER (WHERE days_ago <= 365) AS txn_365d,
        COUNT(*) AS txn_total,
        SUM(price) FILTER (WHERE days_ago <= 28) AS spend_28d,
        AVG(price) FILTER (WHERE days_ago <= 365) AS avg_price_365d,
        COUNT(*) FILTER (WHERE days_ago <= 28)
            - COUNT(*) FILTER (WHERE days_ago > 28 AND days_ago <= 56) AS trend_28d,
        AVG(CASE WHEN sales_channel_id = 2 THEN 1.0 ELSE 0.0 END) AS online_ratio,
        COUNT(DISTINCT article_id)::DOUBLE / NULLIF(COUNT(*), 0) AS item_repeat_ratio,
        CASE WHEN COUNT(*) > 1 THEN
            DATE_DIFF('day', MIN(t_dat), MAX(t_dat))::DOUBLE / (COUNT(*) - 1)
        ELSE NULL END AS avg_gap_days
    FROM txn_base
    GROUP BY task_ts, customer_id
),
cust_profile AS (
    SELECT customer_id,
        COALESCE(age, 35) AS age,
        CASE club_member_status WHEN 'ACTIVE' THEN 1.0 ELSE 0.0 END AS is_active_club,
        CASE fashion_news_frequency
            WHEN 'Regularly' THEN 1.0 WHEN 'Monthly' THEN 0.5 ELSE 0.0
        END AS news_engagement
    FROM customer
)
SELECT
    t.timestamp,
    t.customer_id,

    EXP(-COALESCE(ca.days_since_last, 9999) / 30.0)
                                                        AS recency_decay,

    LEAST(LN(1 + COALESCE(ca.txn_28d, 0)) / LN(51), 1.0)
                                                        AS log_txn_28d_norm,

    LEAST(LN(1 + COALESCE(ca.txn_365d, 0)) / LN(201), 1.0)
                                                        AS log_txn_365d_norm,

    LEAST(GREATEST(
        0.5 + COALESCE(ca.trend_28d, 0) / 20.0,
        0.0), 1.0)                                     AS trend_28d_norm,

    LEAST(COALESCE(ca.avg_gap_days, 90.0) / 90.0, 1.0)
                                                        AS avg_gap_norm,

    LEAST(GREATEST(
        COALESCE(ca.days_since_last, 9999)::DOUBLE
            / GREATEST(COALESCE(gs.max_gap, COALESCE(ca.days_since_last, 9999)), 1),
        0.0), 3.0) / 3.0                              AS gap_ratio_norm,

    COALESCE(ca.online_ratio, 0.5)                     AS online_ratio,

    LEAST(LN(1 + COALESCE(ca.spend_28d, 0)) / LN(6), 1.0)
                                                        AS log_spend_28d_norm,

    COALESCE(ca.item_repeat_ratio, 1.0)                AS item_diversity,

    LEAST(COALESCE(cp.age, 35) / 80.0, 1.0)           AS age_norm,

    COALESCE(cp.is_active_club, 0.0)                   AS is_active_club,

    CASE WHEN ca.customer_id IS NULL THEN 1.0 ELSE 0.0 END
                                                        AS is_new_customer

FROM task t
LEFT JOIN cust_agg ca
    ON ca.customer_id = t.customer_id AND ca.task_ts = t.timestamp
LEFT JOIN gap_stats gs
    ON gs.customer_id = t.customer_id AND gs.task_ts = t.timestamp
LEFT JOIN cust_profile cp
    ON cp.customer_id = t.customer_id
"""


ITEM_SALES_SQL = """
WITH task AS (
    SELECT timestamp, article_id FROM task_table
),
txn_filtered AS (
    SELECT
        t.timestamp AS task_ts, t.article_id,
        txn.t_dat, txn.price, txn.customer_id,
        DATE_DIFF('day', txn.t_dat, t.timestamp) AS days_ago
    FROM task t
    JOIN transactions txn
        ON txn.article_id = t.article_id
        AND txn.t_dat < t.timestamp
),
item_agg AS (
    SELECT
        task_ts, article_id,
        MIN(days_ago)                                              AS days_since_last_sale,
        SUM(price) FILTER (WHERE days_ago <= 7)                   AS sales_7d,
        SUM(price) FILTER (WHERE days_ago <= 28)                  AS sales_28d,
        SUM(price) FILTER (WHERE days_ago <= 365)                 AS sales_365d,
        COUNT(*) FILTER (WHERE days_ago <= 7)                     AS txn_count_7d,
        COUNT(*) FILTER (WHERE days_ago <= 28)                    AS txn_count_28d,
        SUM(price) FILTER (WHERE days_ago <= 7)
            - SUM(price) FILTER (WHERE days_ago > 7 AND days_ago <= 14)
                                                                   AS sales_trend_7d,
        COUNT(DISTINCT customer_id) FILTER (WHERE days_ago <= 28) AS distinct_customers_28d,
        MAX(days_ago)                                              AS item_age_days,
        SUM(price) FILTER (WHERE days_ago <= 28)                  AS spend_28d
    FROM txn_filtered
    GROUP BY task_ts, article_id
),
cat_agg AS (
    SELECT
        t.timestamp AS task_ts,
        a.product_group_name,
        SUM(txn.price) FILTER (
            WHERE DATE_DIFF('day', txn.t_dat, t.timestamp) <= 7
        ) AS cat_sales_7d,
        COUNT(DISTINCT txn.article_id) FILTER (
            WHERE DATE_DIFF('day', txn.t_dat, t.timestamp) <= 28
        ) AS cat_active_items_28d
    FROM (SELECT DISTINCT timestamp FROM task_table) t
    JOIN transactions txn ON txn.t_dat < t.timestamp
    JOIN article a ON a.article_id = txn.article_id
    GROUP BY t.timestamp, a.product_group_name
)
SELECT
    t.timestamp,
    t.article_id,

    LEAST(LN(1 + COALESCE(ia.sales_7d, 0)) / LN(6), 1.0)
                                                        AS log_sales_7d_norm,

    LEAST(LN(1 + COALESCE(ia.sales_28d, 0)) / LN(21), 1.0)
                                                        AS log_sales_28d_norm,

    LEAST(LN(1 + COALESCE(ia.sales_365d, 0)) / LN(101), 1.0)
                                                        AS log_sales_365d_norm,

    LEAST(LN(1 + COALESCE(ia.txn_count_7d, 0)) / LN(201), 1.0)
                                                        AS log_txn_7d_norm,

    LEAST(GREATEST(
        0.5 + COALESCE(ia.sales_trend_7d, 0) / 2.0,
        0.0), 1.0)                                     AS sales_trend_7d_norm,

    EXP(-COALESCE(ia.days_since_last_sale, 9999) / 10.0)
                                                        AS sale_recency,

    LEAST(LN(1 + COALESCE(ia.distinct_customers_28d, 0)) / LN(501), 1.0)
                                                        AS log_customers_28d_norm,

    LEAST(LN(1 + COALESCE(ia.item_age_days, 0)) / LN(1001), 1.0)
                                                        AS log_item_age_norm,

    LEAST(LN(1 + COALESCE(ca.cat_sales_7d, 0)) / LN(501), 1.0)
                                                        AS log_cat_sales_7d_norm,

    CASE WHEN COALESCE(ca.cat_active_items_28d, 0) > 0
         THEN LEAST(
             COALESCE(ia.txn_count_28d, 0)::DOUBLE / ca.cat_active_items_28d,
             1.0)
         ELSE 0.0 END                                  AS item_share_norm,

    LEAST(LN(1 + COALESCE(ia.spend_28d, 0)) / LN(21), 1.0)
                                                        AS log_spend_28d_norm,

    CASE WHEN COALESCE(ia.item_age_days, 0) < 30 THEN 1.0 ELSE 0.0 END
                                                        AS is_new_item,

    CASE WHEN ia.article_id IS NULL THEN 1.0 ELSE 0.0 END
                                                        AS never_sold

FROM task t
LEFT JOIN item_agg ia
    ON ia.article_id = t.article_id AND ia.task_ts = t.timestamp
LEFT JOIN article af
    ON af.article_id = t.article_id
LEFT JOIN cat_agg ca
    ON ca.product_group_name = af.product_group_name
    AND ca.task_ts = t.timestamp
"""
