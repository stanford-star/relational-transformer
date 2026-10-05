"""SQL feature queries for rel-stack (Stack Exchange) tasks.

Each query produces ~12 pre-normalized features designed for
few-shot linear prediction in the rel2tab pipeline.

Database tables:
  - users:       Id, AccountId, DisplayName, Location, CreationDate
  - posts:       Id, OwnerUserId, PostTypeId, ParentId, Title, Tags, Body, CreationDate
  - comments:    Id, PostId, UserId, Text, CreationDate
  - votes:       Id, UserId, PostId, VoteTypeId, CreationDate
  - badges:      Id, UserId, Class, Name, TagBased, Date
  - postLinks:   Id, PostId, RelatedPostId, LinkTypeId, CreationDate
  - postHistory: Id, PostId, UserId, PostHistoryTypeId, CreationDate

Tasks:
  - user-engagement: (timestamp, OwnerUserId) -> binary
  - post-votes:      (timestamp, PostId)      -> regression
  - user-badge:      (timestamp, UserId)      -> binary
"""

# ---------------------------------------------------------------------------
# user-engagement: will user contribute in next ~90 days?
# Entity: OwnerUserId (maps to users.Id and posts.OwnerUserId)
# 12 features: activity_recency, post_recency, contributions_90d, posts_90d,
#   recent_post_burst, total_posts, answer_ratio, badges, account_age,
#   log_edits_90d, received_upvotes, log_votes_90d
# ---------------------------------------------------------------------------

USER_ENGAGEMENT_SQL = """
WITH task AS (
    SELECT timestamp, "OwnerUserId" FROM task_table
),
last_post AS (
    SELECT
        t.timestamp AS task_ts, t."OwnerUserId",
        MIN(DATE_DIFF('day', p."CreationDate", t.timestamp)) AS days_since_last_post,
        COUNT(*) FILTER (WHERE DATE_DIFF('day', p."CreationDate", t.timestamp) <= 90) AS posts_90d,
        COUNT(*) FILTER (WHERE DATE_DIFF('day', p."CreationDate", t.timestamp) <= 30) AS posts_30d,
        COUNT(*) AS posts_total,
        COUNT(*) FILTER (WHERE p."PostTypeId" = 2)::DOUBLE
            / NULLIF(COUNT(*), 0) AS answer_ratio
    FROM task t
    JOIN posts p
        ON p."OwnerUserId" = t."OwnerUserId"
        AND p."CreationDate" < t.timestamp
    GROUP BY t.timestamp, t."OwnerUserId"
),
last_comment AS (
    SELECT
        t.timestamp AS task_ts, t."OwnerUserId",
        MIN(DATE_DIFF('day', c."CreationDate", t.timestamp)) AS days_since_last_comment,
        COUNT(*) FILTER (WHERE DATE_DIFF('day', c."CreationDate", t.timestamp) <= 90) AS comments_90d
    FROM task t
    JOIN comments c
        ON c."UserId" = t."OwnerUserId"
        AND c."CreationDate" < t.timestamp
    GROUP BY t.timestamp, t."OwnerUserId"
),
last_vote AS (
    SELECT
        t.timestamp AS task_ts, t."OwnerUserId",
        MIN(DATE_DIFF('day', v."CreationDate", t.timestamp)) AS days_since_last_vote,
        COUNT(*) FILTER (WHERE DATE_DIFF('day', v."CreationDate", t.timestamp) <= 90) AS votes_90d
    FROM task t
    JOIN votes v
        ON v."UserId" = t."OwnerUserId"
        AND v."CreationDate" < t.timestamp
    GROUP BY t.timestamp, t."OwnerUserId"
),
edit_agg AS (
    SELECT
        t.timestamp AS task_ts, t."OwnerUserId",
        COUNT(*) FILTER (WHERE DATE_DIFF('day', ph."CreationDate", t.timestamp) <= 90) AS edits_90d
    FROM task t
    JOIN "postHistory" ph
        ON ph."UserId" = t."OwnerUserId"
        AND ph."CreationDate" < t.timestamp
    GROUP BY t.timestamp, t."OwnerUserId"
),
received_votes AS (
    SELECT
        t.timestamp AS task_ts, t."OwnerUserId",
        COUNT(*) FILTER (WHERE v."VoteTypeId" = 2) AS received_upvotes
    FROM task t
    JOIN posts p
        ON p."OwnerUserId" = t."OwnerUserId"
        AND p."CreationDate" < t.timestamp
    JOIN votes v
        ON v."PostId" = p."Id"
        AND v."CreationDate" < t.timestamp
    GROUP BY t.timestamp, t."OwnerUserId"
),
badge_agg AS (
    SELECT
        t.timestamp AS task_ts, t."OwnerUserId",
        COUNT(*) AS badges_total
    FROM task t
    JOIN badges b
        ON b."UserId" = t."OwnerUserId"
        AND b."Date" < t.timestamp
    GROUP BY t.timestamp, t."OwnerUserId"
)
SELECT
    t.timestamp,
    t."OwnerUserId",

    -- 1. Recency: min days since any activity (exp decay, half-life ~45d)
    EXP(-LEAST(
        COALESCE(lp.days_since_last_post, 9999),
        COALESCE(lc.days_since_last_comment, 9999),
        COALESCE(lv.days_since_last_vote, 9999)
    ) / 65.0)                                          AS activity_recency,

    -- 2. Post recency (exp decay, half-life ~60d)
    EXP(-COALESCE(lp.days_since_last_post, 9999) / 87.0)
                                                        AS post_recency,

    -- 3. Recent contributions (log-scaled, cap at ~50)
    LEAST(LN(1 + COALESCE(lp.posts_90d, 0)
               + COALESCE(lc.comments_90d, 0)
               + COALESCE(lv.votes_90d, 0)) / LN(51), 1.0)
                                                        AS log_contributions_90d_norm,

    -- 4. Post frequency in 90d (log-scaled, cap at ~30)
    LEAST(LN(1 + COALESCE(lp.posts_90d, 0)) / LN(31), 1.0)
                                                        AS log_posts_90d_norm,

    -- 5. Had recent post burst (posted in last 30d, binary)
    CASE WHEN COALESCE(lp.posts_30d, 0) > 0
         THEN 1.0 ELSE 0.0 END                        AS has_recent_post,

    -- 6. Total posts (log-scaled, cap at ~500)
    LEAST(LN(1 + COALESCE(lp.posts_total, 0)) / LN(501), 1.0)
                                                        AS log_posts_total_norm,

    -- 7. Answer ratio (fraction of posts that are answers, [0,1])
    COALESCE(lp.answer_ratio, 0.0)                     AS answer_ratio,

    -- 8. Badge count (log-scaled, cap at ~200)
    LEAST(LN(1 + COALESCE(ba.badges_total, 0)) / LN(201), 1.0)
                                                        AS log_badges_norm,

    -- 9. Account age (log-scaled, cap at ~3650 days / 10 yrs)
    LEAST(LN(1 + GREATEST(0, DATE_DIFF('day', u."CreationDate", t.timestamp))) / LN(3651), 1.0)
                                                        AS log_account_age_norm,

    -- 10. Edits in last 90d (log-scaled, cap at ~50)
    LEAST(LN(1 + COALESCE(ea.edits_90d, 0)) / LN(51), 1.0)
                                                        AS log_edits_90d_norm,

    -- 11. Received upvotes (log-scaled, reputation proxy, cap at ~5000)
    LEAST(LN(1 + COALESCE(rv.received_upvotes, 0)) / LN(5001), 1.0)
                                                        AS log_received_upvotes_norm,

    -- 12. Votes cast in last 90d (log-scaled, cap at ~200)
    LEAST(LN(1 + COALESCE(lv.votes_90d, 0)) / LN(201), 1.0)
                                                        AS log_votes_90d_norm

FROM task t
LEFT JOIN last_post lp
    ON lp."OwnerUserId" = t."OwnerUserId" AND lp.task_ts = t.timestamp
LEFT JOIN last_comment lc
    ON lc."OwnerUserId" = t."OwnerUserId" AND lc.task_ts = t.timestamp
LEFT JOIN last_vote lv
    ON lv."OwnerUserId" = t."OwnerUserId" AND lv.task_ts = t.timestamp
LEFT JOIN edit_agg ea
    ON ea."OwnerUserId" = t."OwnerUserId" AND ea.task_ts = t.timestamp
LEFT JOIN received_votes rv
    ON rv."OwnerUserId" = t."OwnerUserId" AND rv.task_ts = t.timestamp
LEFT JOIN badge_agg ba
    ON ba."OwnerUserId" = t."OwnerUserId" AND ba.task_ts = t.timestamp
LEFT JOIN users u
    ON u."Id" = t."OwnerUserId"
"""


# ---------------------------------------------------------------------------
# post-votes: predict upvote count in next ~90 days
# Entity: PostId
# 12 features: log_upvotes_90d, log_upvotes_total, upvote_velocity,
#   upvote_trend, upvote_ratio, age_freshness, log_comments,
#   is_answer, post_age_days, owner_reputation, has_accepted_answer,
#   log_edits
# ---------------------------------------------------------------------------

POST_VOTES_SQL = """
WITH task AS (
    SELECT timestamp, "PostId" FROM task_table
),
post_info AS (
    SELECT
        t.timestamp AS task_ts, t."PostId",
        p."PostTypeId",
        p."OwnerUserId",
        p."ParentId",
        DATE_DIFF('day', p."CreationDate", t.timestamp) AS post_age_days
    FROM task t
    JOIN posts p
        ON p."Id" = t."PostId"
        AND p."CreationDate" < t.timestamp
),
vote_agg AS (
    SELECT
        t.timestamp AS task_ts, t."PostId",
        COUNT(*) FILTER (WHERE v."VoteTypeId" = 2 AND DATE_DIFF('day', v."CreationDate", t.timestamp) <= 90)  AS upvotes_90d,
        COUNT(*) FILTER (WHERE v."VoteTypeId" = 2)                     AS upvotes_total,
        COUNT(*) FILTER (WHERE v."VoteTypeId" = 3)                     AS downvotes_total,
        -- Trend: upvotes in 30d minus prev 30d
        COUNT(*) FILTER (WHERE v."VoteTypeId" = 2 AND DATE_DIFF('day', v."CreationDate", t.timestamp) <= 30)
            - COUNT(*) FILTER (WHERE v."VoteTypeId" = 2
                               AND DATE_DIFF('day', v."CreationDate", t.timestamp) > 30
                               AND DATE_DIFF('day', v."CreationDate", t.timestamp) <= 60) AS upvote_trend_30d
    FROM task t
    JOIN votes v
        ON v."PostId" = t."PostId"
        AND v."CreationDate" < t.timestamp
    GROUP BY t.timestamp, t."PostId"
),
post_comment_agg AS (
    SELECT
        t.timestamp AS task_ts, t."PostId",
        COUNT(*) AS comments_total
    FROM task t
    JOIN comments c
        ON c."PostId" = t."PostId"
        AND c."CreationDate" < t.timestamp
    GROUP BY t.timestamp, t."PostId"
),
owner_reputation AS (
    SELECT
        pi.task_ts, pi."PostId",
        COUNT(*) AS owner_upvotes
    FROM post_info pi
    JOIN posts op
        ON op."OwnerUserId" = pi."OwnerUserId"
        AND op."CreationDate" < pi.task_ts
    JOIN votes ov
        ON ov."PostId" = op."Id"
        AND ov."VoteTypeId" = 2
        AND ov."CreationDate" < pi.task_ts
    GROUP BY pi.task_ts, pi."PostId"
),
accepted_answer AS (
    -- For questions: check if any child answer was accepted (VoteTypeId=1 = AcceptedByOriginator)
    SELECT
        pi.task_ts, pi."PostId",
        MAX(CASE WHEN v."VoteTypeId" = 1 THEN 1 ELSE 0 END) AS has_accepted
    FROM post_info pi
    JOIN posts child
        ON child."ParentId" = pi."PostId"
        AND child."CreationDate" < pi.task_ts
    JOIN votes v
        ON v."PostId" = child."Id"
        AND v."CreationDate" < pi.task_ts
    WHERE pi."PostTypeId" = 1
    GROUP BY pi.task_ts, pi."PostId"
),
edit_agg AS (
    SELECT
        t.timestamp AS task_ts, t."PostId",
        COUNT(*) AS edits_total
    FROM task t
    JOIN "postHistory" ph
        ON ph."PostId" = t."PostId"
        AND ph."CreationDate" < t.timestamp
    GROUP BY t.timestamp, t."PostId"
)
SELECT
    t.timestamp,
    t."PostId",

    -- 1. Log upvotes in last 90d (cap at ~50)
    LEAST(LN(1 + COALESCE(va.upvotes_90d, 0)) / LN(51), 1.0)
                                                        AS log_upvotes_90d_norm,

    -- 2. Log upvotes total (cap at ~500)
    LEAST(LN(1 + COALESCE(va.upvotes_total, 0)) / LN(501), 1.0)
                                                        AS log_upvotes_total_norm,

    -- 3. Upvote velocity (upvotes per day of age, cap at 1.0)
    CASE WHEN COALESCE(pi.post_age_days, 0) > 0
         THEN LEAST(COALESCE(va.upvotes_total, 0)::DOUBLE / pi.post_age_days, 1.0)
         ELSE 0.0 END                                 AS upvote_velocity,

    -- 4. Upvote trend (30d vs prev 30d, shifted to [0,1])
    LEAST(GREATEST(
        0.5 + COALESCE(va.upvote_trend_30d, 0) / 10.0,
        0.0), 1.0)                                     AS upvote_trend_norm,

    -- 5. Upvote ratio (upvotes / (upvotes + downvotes), [0,1])
    CASE WHEN COALESCE(va.upvotes_total, 0) + COALESCE(va.downvotes_total, 0) > 0
         THEN va.upvotes_total::DOUBLE / (va.upvotes_total + va.downvotes_total)
         ELSE 0.5 END                                  AS upvote_ratio,

    -- 6. Post age (inverse: newer posts get more votes; exp decay, half-life ~180d)
    EXP(-COALESCE(pi.post_age_days, 0) / 260.0)
                                                        AS age_freshness,

    -- 7. Comments on post (log-scaled, cap at ~50)
    LEAST(LN(1 + COALESCE(pca.comments_total, 0)) / LN(51), 1.0)
                                                        AS log_comments_norm,

    -- 8. Is answer (binary, answers and questions get votes differently)
    CASE WHEN COALESCE(pi."PostTypeId", 0) = 2 THEN 1.0 ELSE 0.0 END
                                                        AS is_answer,

    -- 9. Post age days (log-scaled, cap at ~3650 days)
    LEAST(LN(1 + GREATEST(COALESCE(pi.post_age_days, 0), 0)) / LN(3651), 1.0)
                                                        AS log_post_age_norm,

    -- 10. Owner reputation (log upvotes received by post author, cap at ~10000)
    LEAST(LN(1 + COALESCE(orep.owner_upvotes, 0)) / LN(10001), 1.0)
                                                        AS log_owner_reputation_norm,

    -- 11. Has accepted answer (for questions; 0 for answers/unknown)
    COALESCE(aa.has_accepted, 0)::DOUBLE               AS has_accepted_answer,

    -- 12. Edits on post (log-scaled, cap at ~30)
    LEAST(LN(1 + COALESCE(eda.edits_total, 0)) / LN(31), 1.0)
                                                        AS log_edits_norm

FROM task t
LEFT JOIN post_info pi
    ON pi."PostId" = t."PostId" AND pi.task_ts = t.timestamp
LEFT JOIN vote_agg va
    ON va."PostId" = t."PostId" AND va.task_ts = t.timestamp
LEFT JOIN post_comment_agg pca
    ON pca."PostId" = t."PostId" AND pca.task_ts = t.timestamp
LEFT JOIN owner_reputation orep
    ON orep."PostId" = t."PostId" AND orep.task_ts = t.timestamp
LEFT JOIN accepted_answer aa
    ON aa."PostId" = t."PostId" AND aa.task_ts = t.timestamp
LEFT JOIN edit_agg eda
    ON eda."PostId" = t."PostId" AND eda.task_ts = t.timestamp
"""


# ---------------------------------------------------------------------------
# user-badge: will user receive a badge in next ~90 days?
# Entity: UserId (different from OwnerUserId in user-engagement)
# 12 features: activity_recency, contributions_90d, recent_post,
#   badge_momentum, total_badges, gold_badge_flag, received_upvotes,
#   total_posts, account_age, log_edits_90d, log_votes_90d,
#   log_comments_90d (bonus from contributions)
# ---------------------------------------------------------------------------

USER_BADGE_SQL = """
WITH task AS (
    SELECT timestamp, "UserId" FROM task_table
),
last_post AS (
    SELECT
        t.timestamp AS task_ts, t."UserId",
        MIN(DATE_DIFF('day', p."CreationDate", t.timestamp)) AS days_since_last_post,
        COUNT(*) FILTER (WHERE DATE_DIFF('day', p."CreationDate", t.timestamp) <= 90) AS posts_90d,
        COUNT(*) FILTER (WHERE DATE_DIFF('day', p."CreationDate", t.timestamp) <= 30) AS posts_30d,
        COUNT(*) AS posts_total
    FROM task t
    JOIN posts p
        ON p."OwnerUserId" = t."UserId"
        AND p."CreationDate" < t.timestamp
    GROUP BY t.timestamp, t."UserId"
),
last_comment AS (
    SELECT
        t.timestamp AS task_ts, t."UserId",
        MIN(DATE_DIFF('day', c."CreationDate", t.timestamp)) AS days_since_last_comment,
        COUNT(*) FILTER (WHERE DATE_DIFF('day', c."CreationDate", t.timestamp) <= 90) AS comments_90d
    FROM task t
    JOIN comments c
        ON c."UserId" = t."UserId"
        AND c."CreationDate" < t.timestamp
    GROUP BY t.timestamp, t."UserId"
),
last_vote AS (
    SELECT
        t.timestamp AS task_ts, t."UserId",
        COUNT(*) FILTER (WHERE DATE_DIFF('day', v."CreationDate", t.timestamp) <= 90) AS votes_90d
    FROM task t
    JOIN votes v
        ON v."UserId" = t."UserId"
        AND v."CreationDate" < t.timestamp
    GROUP BY t.timestamp, t."UserId"
),
edit_agg AS (
    SELECT
        t.timestamp AS task_ts, t."UserId",
        COUNT(*) FILTER (WHERE DATE_DIFF('day', ph."CreationDate", t.timestamp) <= 90) AS edits_90d
    FROM task t
    JOIN "postHistory" ph
        ON ph."UserId" = t."UserId"
        AND ph."CreationDate" < t.timestamp
    GROUP BY t.timestamp, t."UserId"
),
badge_agg AS (
    SELECT
        t.timestamp AS task_ts, t."UserId",
        COUNT(*) AS badges_total,
        COUNT(*) FILTER (WHERE DATE_DIFF('day', b."Date", t.timestamp) <= 90) AS badges_90d,
        COUNT(*) FILTER (WHERE b."Class" = 1) AS gold_badges
    FROM task t
    JOIN badges b
        ON b."UserId" = t."UserId"
        AND b."Date" < t.timestamp
    GROUP BY t.timestamp, t."UserId"
),
received_votes AS (
    SELECT
        t.timestamp AS task_ts, t."UserId",
        COUNT(*) FILTER (WHERE v."VoteTypeId" = 2) AS received_upvotes
    FROM task t
    JOIN posts p
        ON p."OwnerUserId" = t."UserId"
        AND p."CreationDate" < t.timestamp
    JOIN votes v
        ON v."PostId" = p."Id"
        AND v."CreationDate" < t.timestamp
    GROUP BY t.timestamp, t."UserId"
)
SELECT
    t.timestamp,
    t."UserId",

    -- 1. Activity recency (exp decay, half-life ~45d)
    EXP(-LEAST(
        COALESCE(lp.days_since_last_post, 9999),
        COALESCE(lc.days_since_last_comment, 9999)
    ) / 65.0)                                          AS activity_recency,

    -- 2. Recent contributions (log-scaled, cap at ~50)
    LEAST(LN(1 + COALESCE(lp.posts_90d, 0)
               + COALESCE(lc.comments_90d, 0)) / LN(51), 1.0)
                                                        AS log_contributions_90d_norm,

    -- 3. Had recent post (posted in last 30d, binary)
    CASE WHEN COALESCE(lp.posts_30d, 0) > 0
         THEN 1.0 ELSE 0.0 END                        AS has_recent_post,

    -- 4. Badge momentum (recent badges in 90d, capped at 5)
    LEAST(COALESCE(ba.badges_90d, 0) / 5.0, 1.0)
                                                        AS badge_momentum,

    -- 5. Total badges (log-scaled, cap at ~200)
    LEAST(LN(1 + COALESCE(ba.badges_total, 0)) / LN(201), 1.0)
                                                        AS log_badges_total_norm,

    -- 6. Has gold badge (binary, strong predictor of badge earning)
    CASE WHEN COALESCE(ba.gold_badges, 0) > 0
         THEN 1.0 ELSE 0.0 END                        AS gold_badge_flag,

    -- 7. Received upvotes (log-scaled, reputation proxy, cap at ~5000)
    LEAST(LN(1 + COALESCE(rv.received_upvotes, 0)) / LN(5001), 1.0)
                                                        AS log_received_upvotes_norm,

    -- 8. Total posts (log-scaled, cap at ~500)
    LEAST(LN(1 + COALESCE(lp.posts_total, 0)) / LN(501), 1.0)
                                                        AS log_posts_total_norm,

    -- 9. Account age (log-scaled, cap at ~3650 days / 10 yrs)
    LEAST(LN(1 + GREATEST(0, DATE_DIFF('day', u."CreationDate", t.timestamp))) / LN(3651), 1.0)
                                                        AS log_account_age_norm,

    -- 10. Edits in last 90d (log-scaled, cap at ~50)
    LEAST(LN(1 + COALESCE(ea.edits_90d, 0)) / LN(51), 1.0)
                                                        AS log_edits_90d_norm,

    -- 11. Votes cast in last 90d (log-scaled, cap at ~200)
    LEAST(LN(1 + COALESCE(lv.votes_90d, 0)) / LN(201), 1.0)
                                                        AS log_votes_90d_norm,

    -- 12. Comments in last 90d (log-scaled, cap at ~50)
    LEAST(LN(1 + COALESCE(lc.comments_90d, 0)) / LN(51), 1.0)
                                                        AS log_comments_90d_norm

FROM task t
LEFT JOIN last_post lp
    ON lp."UserId" = t."UserId" AND lp.task_ts = t.timestamp
LEFT JOIN last_comment lc
    ON lc."UserId" = t."UserId" AND lc.task_ts = t.timestamp
LEFT JOIN last_vote lv
    ON lv."UserId" = t."UserId" AND lv.task_ts = t.timestamp
LEFT JOIN edit_agg ea
    ON ea."UserId" = t."UserId" AND ea.task_ts = t.timestamp
LEFT JOIN badge_agg ba
    ON ba."UserId" = t."UserId" AND ba.task_ts = t.timestamp
LEFT JOIN received_votes rv
    ON rv."UserId" = t."UserId" AND rv.task_ts = t.timestamp
LEFT JOIN users u
    ON u."Id" = t."UserId"
"""
