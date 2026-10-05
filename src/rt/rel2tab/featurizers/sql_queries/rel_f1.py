"""SQL feature queries for rel-f1 (Formula 1) tasks.

Each query produces 8-12 pre-normalized features designed for
few-shot linear prediction in the rel2tab pipeline.
"""

DRIVER_POSITION_SQL = """
WITH task AS (
    SELECT date AS timestamp, "driverId" FROM task_table
),
driver_results AS (
    SELECT
        t.timestamp AS task_ts, t."driverId",
        r."positionOrder", r.grid, r.points,
        r."statusId", r."constructorId",
        DATE_DIFF('day', r.date, t.timestamp) AS days_ago
    FROM task t
    JOIN results r
        ON r."driverId" = t."driverId"
        AND r.date < t.timestamp
),
race_agg AS (
    SELECT
        task_ts, "driverId",

        COUNT(*) AS races_total,
        COUNT(*) FILTER (WHERE days_ago <= 365) AS races_1y,

        AVG("positionOrder") FILTER (WHERE days_ago <= 365) AS avg_finish_order_1y,
        AVG(grid) FILTER (WHERE grid > 0 AND days_ago <= 365) AS avg_grid_1y,

        COUNT(*) FILTER (WHERE "positionOrder" <= 3) AS podiums_total,
        COUNT(*) FILTER (WHERE "positionOrder" <= 3 AND days_ago <= 365) AS podiums_1y,
        COUNT(*) FILTER (WHERE "statusId" != 1) AS dnf_total,
        COUNT(*) FILTER (WHERE "statusId" != 1 AND days_ago <= 365) AS dnf_1y,

        FIRST("constructorId" ORDER BY days_ago) AS latest_constructor_id,
        FIRST("positionOrder" ORDER BY days_ago) AS last_race_position,
        FIRST(CASE WHEN "statusId" != 1 THEN 1 ELSE 0 END ORDER BY days_ago) AS last_race_dnf

    FROM driver_results
    GROUP BY task_ts, "driverId"
),
qual_agg AS (
    SELECT
        t.timestamp AS task_ts, t."driverId",
        AVG(q."position") FILTER (WHERE DATE_DIFF('day', q.date, t.timestamp) <= 365)
            AS avg_qual_position_1y
    FROM task t
    JOIN qualifying q
        ON q."driverId" = t."driverId"
        AND q.date < t.timestamp
    GROUP BY t.timestamp, t."driverId"
),
constructor_strength AS (
    SELECT
        t.timestamp AS task_ts, ra."driverId",
        FIRST(cs."position" ORDER BY DATE_DIFF('day', cs.date, t.timestamp))
            AS constructor_standing
    FROM task t
    JOIN race_agg ra ON ra."driverId" = t."driverId" AND ra.task_ts = t.timestamp
    JOIN constructor_standings cs
        ON cs."constructorId" = ra.latest_constructor_id
        AND cs.date < t.timestamp
    GROUP BY t.timestamp, ra."driverId"
),
constructor_perf AS (
    SELECT
        t.timestamp AS task_ts, ra."driverId",
        AVG(r2."positionOrder") FILTER (WHERE DATE_DIFF('day', r2.date, t.timestamp) <= 365)
            AS team_avg_position_1y
    FROM task t
    JOIN race_agg ra ON ra."driverId" = t."driverId" AND ra.task_ts = t.timestamp
    JOIN results r2
        ON r2."constructorId" = ra.latest_constructor_id
        AND r2.date < t.timestamp
    GROUP BY t.timestamp, ra."driverId"
)
SELECT
    t.timestamp AS date,
    t."driverId",

    -- 1. Avg finish position last year (normalized to [0,1], 1st=best → 0.05, 20th → 1.0)
    LEAST(COALESCE(ra.avg_finish_order_1y, 13.0) / 20.0, 1.0)
                                                        AS avg_finish_1y_norm,

    -- 2. Avg grid position last year (normalized)
    LEAST(COALESCE(ra.avg_grid_1y, 13.0) / 20.0, 1.0)
                                                        AS avg_grid_1y_norm,

    -- 3. Podium rate (all-time, [0,1])
    CASE WHEN COALESCE(ra.races_total, 0) > 0
         THEN ra.podiums_total::DOUBLE / ra.races_total
         ELSE 0.0 END                                  AS podium_rate,

    -- 4. Podium rate last year ([0,1])
    CASE WHEN COALESCE(ra.races_1y, 0) > 0
         THEN ra.podiums_1y::DOUBLE / ra.races_1y
         ELSE 0.0 END                                  AS podium_rate_1y,

    -- 5. DNF rate (all-time, [0,1])
    CASE WHEN COALESCE(ra.races_total, 0) > 0
         THEN ra.dnf_total::DOUBLE / ra.races_total
         ELSE 0.5 END                                  AS dnf_rate,

    -- 6. DNF rate last year ([0,1])
    CASE WHEN COALESCE(ra.races_1y, 0) > 0
         THEN ra.dnf_1y::DOUBLE / ra.races_1y
         ELSE 0.5 END                                  AS dnf_rate_1y,

    -- 7. Last race position (normalized)
    LEAST(COALESCE(ra.last_race_position, 13.0) / 20.0, 1.0)
                                                        AS last_race_pos_norm,

    -- 8. Last race DNF (binary)
    COALESCE(ra.last_race_dnf, 0)::DOUBLE              AS last_race_dnf,

    -- 9. Avg qualifying position last year (normalized)
    LEAST(COALESCE(qa.avg_qual_position_1y, 13.0) / 20.0, 1.0)
                                                        AS avg_qual_1y_norm,

    -- 10. Constructor standing (normalized, 1=best → 0.1, 10=worst → 1.0)
    LEAST(COALESCE(cst.constructor_standing, 6.0) / 10.0, 1.0)
                                                        AS constructor_standing_norm,

    -- 11. Team avg position last year (normalized)
    LEAST(COALESCE(cp.team_avg_position_1y, 13.0) / 20.0, 1.0)
                                                        AS team_avg_pos_1y_norm,

    -- 12. Driver vs team (relative skill, shifted to [0,1], 0.5=neutral)
    LEAST(GREATEST(
        0.5 + (COALESCE(ra.avg_finish_order_1y, 13) - COALESCE(cp.team_avg_position_1y, 13)) / 20.0,
        0.0), 1.0)                                     AS driver_vs_team_norm

FROM task t
LEFT JOIN race_agg ra ON ra."driverId" = t."driverId" AND ra.task_ts = t.timestamp
LEFT JOIN qual_agg qa ON qa."driverId" = t."driverId" AND qa.task_ts = t.timestamp
LEFT JOIN constructor_strength cst ON cst."driverId" = t."driverId" AND cst.task_ts = t.timestamp
LEFT JOIN constructor_perf cp ON cp."driverId" = t."driverId" AND cp.task_ts = t.timestamp
"""

DRIVER_DNF_SQL = DRIVER_POSITION_SQL
DRIVER_TOP3_SQL = DRIVER_POSITION_SQL
