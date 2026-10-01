USER_FEATURES_SQL = """
WITH task AS (
    SELECT timestamp, "user" FROM task_table
),
attendance_history AS (
    SELECT
        t.timestamp AS task_ts, t."user",
        ea.status,
        DATE_DIFF('day', ea.start_time, t.timestamp) AS days_ago
    FROM task t
    JOIN event_attendees ea
        ON ea.user_id = t."user"
        AND ea.start_time < t.timestamp
),
attend_agg AS (
    SELECT
        task_ts, "user",

        MIN(days_ago) FILTER (WHERE status IN ('yes','maybe')) AS days_since_last_attend,
        MIN(days_ago) FILTER (WHERE status = 'invited')        AS days_since_last_invite,

        COUNT(*) FILTER (WHERE status IN ('yes','maybe') AND days_ago <= 7)  AS attend_7d,
        COUNT(*) FILTER (WHERE status IN ('yes','maybe') AND days_ago <= 14) AS attend_14d,
        COUNT(*) FILTER (WHERE status IN ('yes','maybe') AND days_ago <= 90) AS attend_90d,

        COUNT(*) FILTER (WHERE status IN ('yes','maybe') AND days_ago <= 7)
            - COUNT(*) FILTER (WHERE status IN ('yes','maybe')
                               AND days_ago > 7 AND days_ago <= 14) AS attend_trend_7d,

        COUNT(*) FILTER (WHERE status = 'invited' AND days_ago <= 7)  AS invites_7d,
        COUNT(*) FILTER (WHERE status = 'invited' AND days_ago <= 90) AS invites_90d,

        COUNT(*) FILTER (WHERE status = 'yes')::DOUBLE
            / NULLIF(COUNT(*) FILTER (WHERE status IN ('yes','maybe','no')), 0)
            AS yes_ratio
    FROM attendance_history
    GROUP BY task_ts, "user"
),
interest_history AS (
    SELECT
        t.timestamp AS task_ts, t."user",
        ei.invited, ei.interested, ei.not_interested
    FROM task t
    JOIN event_interest ei
        ON ei."user" = t."user"
        AND ei.timestamp < t.timestamp
),
interest_agg AS (
    SELECT
        task_ts, "user",
        AVG(CASE WHEN invited = 1 AND interested = 1     THEN 1.0
                 WHEN invited = 1                        THEN 0.0
                 ELSE NULL END)                         AS interest_rate,
        AVG(CASE WHEN invited = 1 AND not_interested = 1 THEN 1.0
                 WHEN invited = 1                        THEN 0.0
                 ELSE NULL END)                         AS ignore_rate
    FROM interest_history
    GROUP BY task_ts, "user"
),
friend_count AS (
    SELECT "user", COUNT(*) AS num_friends
    FROM user_friends
    GROUP BY "user"
),
user_profile AS (
    SELECT
        user_id,
        "joinedAt" AS joined_at
    FROM users
)
SELECT
    t.timestamp,
    t."user",

    EXP(-COALESCE(aa.days_since_last_attend, 9999) / 20.0)
                                                        AS attend_recency,

    EXP(-COALESCE(aa.days_since_last_invite, 9999) / 20.0)
                                                        AS invite_recency,

    LEAST(LN(1 + COALESCE(aa.attend_7d, 0)) / LN(31), 1.0)
                                                        AS log_attend_7d_norm,

    LEAST(LN(1 + COALESCE(aa.attend_14d, 0)) / LN(51), 1.0)
                                                        AS log_attend_14d_norm,

    LEAST(LN(1 + COALESCE(aa.attend_90d, 0)) / LN(201), 1.0)
                                                        AS log_attend_90d_norm,

    LEAST(GREATEST(
        0.5 + COALESCE(aa.attend_trend_7d, 0) / 10.0,
        0.0), 1.0)                                     AS attend_trend_7d_norm,

    COALESCE(aa.yes_ratio, 0.5)                         AS yes_ratio,

    LEAST(LN(1 + COALESCE(aa.invites_7d, 0)) / LN(31), 1.0)
                                                        AS log_invites_7d_norm,

    LEAST(LN(1 + COALESCE(aa.invites_90d, 0)) / LN(201), 1.0)
                                                        AS log_invites_90d_norm,

    COALESCE(ia.interest_rate, 0.3)                     AS interest_rate,

    COALESCE(ia.ignore_rate, 0.1)                       AS ignore_rate,

    LEAST(LN(1 + COALESCE(fc.num_friends, 0)) / LN(5001), 1.0)
                                                        AS log_friends_norm,

    LEAST(LN(1 + GREATEST(0, COALESCE(
            DATE_DIFF('day', up.joined_at, t.timestamp), 0))) / LN(3651), 1.0)
                                                        AS log_account_age_norm,

    CASE WHEN aa."user" IS NULL AND ia."user" IS NULL
         THEN 1.0 ELSE 0.0 END                         AS is_new_user

FROM task t
LEFT JOIN attend_agg aa
    ON aa."user" = t."user" AND aa.task_ts = t.timestamp
LEFT JOIN interest_agg ia
    ON ia."user" = t."user" AND ia.task_ts = t.timestamp
LEFT JOIN friend_count fc
    ON fc."user" = t."user"
LEFT JOIN user_profile up
    ON up.user_id = t."user"
"""

USER_ATTENDANCE_SQL = USER_FEATURES_SQL
USER_REPEAT_SQL = USER_FEATURES_SQL
USER_IGNORE_SQL = USER_FEATURES_SQL
