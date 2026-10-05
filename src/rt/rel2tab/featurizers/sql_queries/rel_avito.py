"""SQL feature queries for rel-avito (Avito classifieds) tasks.

Each query produces 8-12 pre-normalized features designed for
few-shot linear prediction in the rel2tab pipeline.
"""

AD_CTR_SQL = """
WITH task AS (
    SELECT timestamp, "AdID" FROM task_table
),
ad_info AS (
    SELECT a."AdID", a."CategoryID", a."Price", a."IsContext"
    FROM "AdsInfo" a
),
ad_search_hist AS (
    SELECT
        t.timestamp AS task_ts, t."AdID",
        COUNT(*) AS impressions_total,
        COUNT(*) FILTER (WHERE DATE_DIFF('day', ss."SearchDate", t.timestamp) <= 3) AS impressions_3d,
        SUM(ss."IsClick") AS clicks_total,
        SUM(ss."IsClick") FILTER (WHERE DATE_DIFF('day', ss."SearchDate", t.timestamp) <= 3) AS clicks_3d,
        AVG(ss."Position") AS avg_position,
        AVG(ss."HistCTR") AS avg_hist_ctr
    FROM task t
    JOIN "SearchStream" ss
        ON ss."AdID" = t."AdID"
        AND ss."SearchDate" < t.timestamp
    GROUP BY t.timestamp, t."AdID"
),
ad_visit_hist AS (
    SELECT
        t.timestamp AS task_ts, t."AdID",
        COUNT(*) AS visits_total
    FROM task t
    JOIN "VisitStream" vs
        ON vs."AdID" = t."AdID"
        AND vs."ViewDate" < t.timestamp
    GROUP BY t.timestamp, t."AdID"
),
cat_ctr AS (
    SELECT
        t.timestamp AS task_ts,
        ai."CategoryID",
        AVG(ss."IsClick") AS cat_avg_ctr
    FROM (SELECT DISTINCT timestamp FROM task_table) t
    JOIN "SearchStream" ss ON ss."SearchDate" < t.timestamp
    JOIN "AdsInfo" ai ON ai."AdID" = ss."AdID"
    GROUP BY t.timestamp, ai."CategoryID"
),
cat_price AS (
    SELECT "CategoryID",
        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY "Price") AS cat_median_price
    FROM "AdsInfo"
    WHERE "Price" > 0
    GROUP BY "CategoryID"
)
SELECT
    t.timestamp,
    t."AdID",

    -- 1. Historical CTR (ratio, [0,1])
    CASE WHEN COALESCE(ash.impressions_total, 0) > 0
         THEN LEAST(ash.clicks_total::DOUBLE / ash.impressions_total, 1.0)
         ELSE 0.02 END                                 AS hist_ctr,

    -- 2. Platform historical CTR (avg_hist_ctr, already ~[0,1])
    LEAST(GREATEST(COALESCE(ash.avg_hist_ctr, 0.02), 0.0), 1.0)
                                                        AS avg_hist_ctr,

    -- 3. Category CTR (already [0,1])
    LEAST(GREATEST(COALESCE(cc.cat_avg_ctr, 0.02), 0.0), 1.0)
                                                        AS cat_avg_ctr,

    -- 4. Log price (normalized, cap at ~1M → LN(1M)≈13.8)
    LEAST(LN(1 + COALESCE(ai."Price", 0)) / 13.8, 1.0)
                                                        AS log_price_norm,

    -- 5. Relative price (ad vs category median, capped at 5x)
    CASE WHEN COALESCE(cp.cat_median_price, 0) > 0
         THEN LEAST(COALESCE(ai."Price", 0) / cp.cat_median_price / 5.0, 1.0)
         ELSE 0.2 END                                  AS relative_price_norm,

    -- 6. Log impressions (capped at ~10K → LN(10001)≈9.2)
    LEAST(LN(1 + COALESCE(ash.impressions_total, 0)) / 9.2, 1.0)
                                                        AS log_impressions_norm,

    -- 7. Average position (normalized, lower=better, cap at 20)
    LEAST(COALESCE(ash.avg_position, 10.0) / 20.0, 1.0)
                                                        AS avg_position_norm,

    -- 8. Visit rate (visits / impressions, [0,1])
    CASE WHEN COALESCE(ash.impressions_total, 0) > 0
         THEN LEAST(COALESCE(avh.visits_total, 0)::DOUBLE / ash.impressions_total, 1.0)
         ELSE 0.0 END                                  AS visit_rate,

    -- 9. CTR trend (recent 3d vs all-time, shifted to [0,1] with 0.5=neutral)
    CASE WHEN COALESCE(ash.impressions_total, 0) > 5 AND COALESCE(ash.impressions_3d, 0) > 2
         THEN LEAST(GREATEST(
             0.5 + (ash.clicks_3d::DOUBLE / ash.impressions_3d
                    - ash.clicks_total::DOUBLE / ash.impressions_total),
             0.0), 1.0)
         ELSE 0.5 END                                  AS ctr_trend_norm,

    -- 10. Is new ad (no search history, binary)
    CASE WHEN ash."AdID" IS NULL THEN 1.0 ELSE 0.0 END AS is_new_ad

FROM task t
LEFT JOIN ad_info ai ON ai."AdID" = t."AdID"
LEFT JOIN ad_search_hist ash ON ash."AdID" = t."AdID" AND ash.task_ts = t.timestamp
LEFT JOIN ad_visit_hist avh ON avh."AdID" = t."AdID" AND avh.task_ts = t.timestamp
LEFT JOIN cat_ctr cc ON cc."CategoryID" = ai."CategoryID" AND cc.task_ts = t.timestamp
LEFT JOIN cat_price cp ON cp."CategoryID" = ai."CategoryID"
"""

USER_VISITS_SQL = """
WITH task AS (
    SELECT timestamp, "UserID" FROM task_table
),
user_search AS (
    SELECT
        t.timestamp AS task_ts, t."UserID",
        COUNT(*) AS searches_total,
        COUNT(*) FILTER (WHERE DATE_DIFF('day', si."SearchDate", t.timestamp) <= 3) AS searches_3d,
        MIN(DATE_DIFF('day', si."SearchDate", t.timestamp)) AS days_since_last_search,
        AVG(si."IsUserLoggedOn") AS avg_logged_on
    FROM task t
    JOIN "SearchInfo" si
        ON si."UserID" = t."UserID"
        AND si."SearchDate" < t.timestamp
    GROUP BY t.timestamp, t."UserID"
),
user_visits AS (
    SELECT
        t.timestamp AS task_ts, t."UserID",
        COUNT(*) AS visits_total,
        COUNT(*) FILTER (WHERE DATE_DIFF('day', vs."ViewDate", t.timestamp) <= 3) AS visits_3d,
        MIN(DATE_DIFF('day', vs."ViewDate", t.timestamp)) AS days_since_last_visit,
        COUNT(DISTINCT vs."AdID") AS unique_ads_visited
    FROM task t
    JOIN "VisitStream" vs
        ON vs."UserID" = t."UserID"
        AND vs."ViewDate" < t.timestamp
    GROUP BY t.timestamp, t."UserID"
),
user_clicks AS (
    SELECT
        t.timestamp AS task_ts, t."UserID",
        SUM(ss."IsClick") AS clicks_total,
        COUNT(*) AS stream_impressions
    FROM task t
    JOIN "SearchInfo" si
        ON si."UserID" = t."UserID"
        AND si."SearchDate" < t.timestamp
    JOIN "SearchStream" ss
        ON ss."SearchID" = si."SearchID"
        AND ss."SearchDate" < t.timestamp
    GROUP BY t.timestamp, t."UserID"
)
SELECT
    t.timestamp,
    t."UserID",

    -- 1. Log visit count (capped at ~5K → LN(5001)≈8.5)
    LEAST(LN(1 + COALESCE(uv.visits_total, 0)) / 8.5, 1.0)
                                                        AS log_visits_norm,

    -- 2. Recent visit intensity (visits in 3d, log-scaled, cap ~100)
    LEAST(LN(1 + COALESCE(uv.visits_3d, 0)) / 4.6, 1.0)
                                                        AS log_visits_3d_norm,

    -- 3. Visit recency (exponential decay, 3-day half-life)
    EXP(-COALESCE(uv.days_since_last_visit, 9999) / 4.3)
                                                        AS visit_recency,

    -- 4. Unique ads visited (diversity, log-scaled, cap ~2K)
    LEAST(LN(1 + COALESCE(uv.unique_ads_visited, 0)) / 7.6, 1.0)
                                                        AS log_unique_ads_norm,

    -- 5. Log search count (cap ~10K)
    LEAST(LN(1 + COALESCE(us.searches_total, 0)) / 9.2, 1.0)
                                                        AS log_searches_norm,

    -- 6. Search recency (exponential decay, 3-day half-life)
    EXP(-COALESCE(us.days_since_last_search, 9999) / 4.3)
                                                        AS search_recency,

    -- 7. User CTR (clicks / impressions, [0,1])
    CASE WHEN COALESCE(uc.stream_impressions, 0) > 0
         THEN LEAST(uc.clicks_total::DOUBLE / uc.stream_impressions, 1.0)
         ELSE 0.0 END                                  AS user_ctr,

    -- 8. Log stream impressions (cap ~50K → LN(50001)≈10.8)
    LEAST(LN(1 + COALESCE(uc.stream_impressions, 0)) / 10.8, 1.0)
                                                        AS log_impressions_norm,

    -- 9. Avg logged on (already [0,1])
    COALESCE(us.avg_logged_on, 0.0)                    AS avg_logged_on,

    -- 10. Is new user (binary)
    CASE WHEN us."UserID" IS NULL AND uv."UserID" IS NULL
         THEN 1.0 ELSE 0.0 END                        AS is_new_user

FROM task t
LEFT JOIN user_search us ON us."UserID" = t."UserID" AND us.task_ts = t.timestamp
LEFT JOIN user_visits uv ON uv."UserID" = t."UserID" AND uv.task_ts = t.timestamp
LEFT JOIN user_clicks uc ON uc."UserID" = t."UserID" AND uc.task_ts = t.timestamp
"""

USER_CLICKS_SQL = USER_VISITS_SQL
