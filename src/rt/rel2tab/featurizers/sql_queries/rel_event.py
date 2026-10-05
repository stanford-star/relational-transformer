"""SQL feature queries for rel-event (event recommendation) tasks.

Each query produces ~14 pre-normalized features designed for
few-shot linear prediction in the rel2tab pipeline.

Database tables:
  - users:           user_id, locale, birthyear, gender, joinedAt, location, timezone
  - events:          event_id, user_id (creator), start_time, city, state, zip,
                     country, lat, lng, c_1..c_100, c_other
  - event_attendees: event, status (yes/no/maybe/invited), user_id, start_time
  - event_interest:  user, event, invited, timestamp, interested, not_interested
  - user_friends:    user, friend  (no time column, treated as static)

Tasks (all entity=user, all share the same feature set since they all
predict different facets of next-7-day behavior):
  - user-attendance: timestamp -> regression  (count of events attended next 7d)
  - user-repeat:     timestamp -> binary      (will attend next 7d | attended last 14d)
  - user-ignore:     timestamp -> binary      (will ignore >2 invites next 7d)
"""

# ---------------------------------------------------------------------------
# User-level features (shared by user-attendance, user-repeat, user-ignore)
# 14 features: attend_recency, invite_recency, log_attend_7d/14d/90d,
#   attend_trend_7d, yes_ratio, log_invites_7d/90d, interest_rate,
#   ignore_rate, log_friends, log_account_age, is_new_user
# ---------------------------------------------------------------------------

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

        -- Recency (yes/maybe attendance, separately for invitations)
        MIN(days_ago) FILTER (WHERE status IN ('yes','maybe')) AS days_since_last_attend,
        MIN(days_ago) FILTER (WHERE status = 'invited')        AS days_since_last_invite,

        -- Yes/maybe attendance volume
        COUNT(*) FILTER (WHERE status IN ('yes','maybe') AND days_ago <= 7)  AS attend_7d,
        COUNT(*) FILTER (WHERE status IN ('yes','maybe') AND days_ago <= 14) AS attend_14d,
        COUNT(*) FILTER (WHERE status IN ('yes','maybe') AND days_ago <= 90) AS attend_90d,

        -- Trend: yes/maybe in 7d minus prev 7d
        COUNT(*) FILTER (WHERE status IN ('yes','maybe') AND days_ago <= 7)
            - COUNT(*) FILTER (WHERE status IN ('yes','maybe')
                               AND days_ago > 7 AND days_ago <= 14) AS attend_trend_7d,

        -- Invitation volume
        COUNT(*) FILTER (WHERE status = 'invited' AND days_ago <= 7)  AS invites_7d,
        COUNT(*) FILTER (WHERE status = 'invited' AND days_ago <= 90) AS invites_90d,

        -- Yes ratio among non-invited responses (yes/maybe/no)
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

    -- 1. Recency of last yes/maybe attendance (exp decay, half-life ~14d)
    EXP(-COALESCE(aa.days_since_last_attend, 9999) / 20.0)
                                                        AS attend_recency,

    -- 2. Recency of last invitation (exp decay, half-life ~14d)
    EXP(-COALESCE(aa.days_since_last_invite, 9999) / 20.0)
                                                        AS invite_recency,

    -- 3. Yes/maybe attendance in last 7d (log-scaled, cap at ~30)
    LEAST(LN(1 + COALESCE(aa.attend_7d, 0)) / LN(31), 1.0)
                                                        AS log_attend_7d_norm,

    -- 4. Yes/maybe attendance in last 14d (log-scaled, cap at ~50)
    --    This is the conditioning window for user-repeat.
    LEAST(LN(1 + COALESCE(aa.attend_14d, 0)) / LN(51), 1.0)
                                                        AS log_attend_14d_norm,

    -- 5. Yes/maybe attendance in last 90d (log-scaled, cap at ~200)
    LEAST(LN(1 + COALESCE(aa.attend_90d, 0)) / LN(201), 1.0)
                                                        AS log_attend_90d_norm,

    -- 6. Attendance trend (7d vs prev 7d, shifted to [0,1])
    LEAST(GREATEST(
        0.5 + COALESCE(aa.attend_trend_7d, 0) / 10.0,
        0.0), 1.0)                                     AS attend_trend_7d_norm,

    -- 7. Yes ratio among yes/maybe/no responses (already [0,1])
    COALESCE(aa.yes_ratio, 0.5)                         AS yes_ratio,

    -- 8. Invitations in last 7d (log-scaled, cap at ~30)
    --    Direct prior for user-ignore (target window is invites in next 7d).
    LEAST(LN(1 + COALESCE(aa.invites_7d, 0)) / LN(31), 1.0)
                                                        AS log_invites_7d_norm,

    -- 9. Invitations in last 90d (log-scaled, cap at ~200)
    LEAST(LN(1 + COALESCE(aa.invites_90d, 0)) / LN(201), 1.0)
                                                        AS log_invites_90d_norm,

    -- 10. Interest rate (fraction of invitations marked interested, [0,1])
    COALESCE(ia.interest_rate, 0.3)                     AS interest_rate,

    -- 11. Ignore rate (fraction of invitations marked not_interested, [0,1])
    --     Strong prior for user-ignore.
    COALESCE(ia.ignore_rate, 0.1)                       AS ignore_rate,

    -- 12. Friend count (log-scaled, cap at ~5000)
    LEAST(LN(1 + COALESCE(fc.num_friends, 0)) / LN(5001), 1.0)
                                                        AS log_friends_norm,

    -- 13. Account age (log-scaled, cap at ~3650 days / 10 yrs)
    LEAST(LN(1 + GREATEST(0, COALESCE(
            DATE_DIFF('day', up.joined_at, t.timestamp), 0))) / LN(3651), 1.0)
                                                        AS log_account_age_norm,

    -- 14. Is new user (no attendance and no invitation history, binary)
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
