"""SQL feature queries for rel-trial (Clinical Trials) tasks.

Each query produces 8-12 pre-normalized features designed for
few-shot linear prediction in the rel2tab pipeline.
"""

STUDY_OUTCOME_SQL = """
WITH task AS (
    SELECT timestamp, nct_id FROM task_table
),
study_info AS (
    SELECT
        t.timestamp AS task_ts,
        t.nct_id,
        s.phase,
        s.enrollment,
        s.has_dmc,
        s.source_class,
        DATE_DIFF('day', s.start_date, t.timestamp) AS study_age_days
    FROM task t
    JOIN studies s ON s.nct_id = t.nct_id
),
design_info AS (
    SELECT t.timestamp AS task_ts, t.nct_id, d.masking
    FROM task t
    JOIN designs d ON d.nct_id = t.nct_id AND d.date < t.timestamp
),
site_count AS (
    SELECT t.timestamp AS task_ts, t.nct_id,
        COUNT(DISTINCT fs.facility_id) AS num_sites
    FROM task t
    JOIN facilities_studies fs ON fs.nct_id = t.nct_id AND fs.date < t.timestamp
    GROUP BY t.timestamp, t.nct_id
),
sponsor_history AS (
    SELECT
        t.timestamp AS task_ts, t.nct_id,
        AVG(prev_oa.p_value) AS sponsor_avg_p_value,
        COUNT(DISTINCT prev_ss.nct_id) AS sponsor_num_studies
    FROM task t
    JOIN sponsors_studies ss ON ss.nct_id = t.nct_id AND ss.date < t.timestamp
    JOIN sponsors_studies prev_ss
        ON prev_ss.sponsor_id = ss.sponsor_id
        AND prev_ss.nct_id != t.nct_id
        AND prev_ss.date < t.timestamp
    LEFT JOIN outcome_analyses prev_oa
        ON prev_oa.nct_id = prev_ss.nct_id
        AND prev_oa.date < t.timestamp
    GROUP BY t.timestamp, t.nct_id
),
condition_history AS (
    SELECT
        t.timestamp AS task_ts, t.nct_id,
        AVG(prev_oa.p_value) AS condition_avg_p_value,
        COUNT(DISTINCT prev_cs.nct_id) AS condition_num_studies,
        COUNT(*) FILTER (WHERE prev_oa.p_value < 0.05) AS condition_num_significant
    FROM task t
    JOIN conditions_studies cs ON cs.nct_id = t.nct_id AND cs.date < t.timestamp
    JOIN conditions_studies prev_cs
        ON prev_cs.condition_id = cs.condition_id
        AND prev_cs.nct_id != t.nct_id
        AND prev_cs.date < t.timestamp
    LEFT JOIN outcome_analyses prev_oa
        ON prev_oa.nct_id = prev_cs.nct_id
        AND prev_oa.date < t.timestamp
    GROUP BY t.timestamp, t.nct_id
)
SELECT
    t.timestamp,
    t.nct_id,

    -- 1. Sponsor track record (p-value already in [0,1])
    COALESCE(sh.sponsor_avg_p_value, 0.5)              AS sponsor_avg_p,

    -- 2. Sponsor experience (log-scaled, capped)
    LEAST(LN(1 + COALESCE(sh.sponsor_num_studies, 0)) / LN(500), 1.0)
                                                        AS sponsor_experience,

    -- 3. Condition area difficulty (p-value in [0,1])
    COALESCE(ch.condition_avg_p_value, 0.5)            AS condition_avg_p,

    -- 4. Condition historical success rate
    CASE WHEN COALESCE(ch.condition_num_studies, 0) > 0
         THEN LEAST(ch.condition_num_significant::DOUBLE
                     / ch.condition_num_studies, 1.0)
         ELSE 0.3 END                                  AS condition_sig_rate,

    -- 5. Log enrollment (capped at ~50K → LN(50001)≈10.8)
    LEAST(LN(1 + COALESCE(si.enrollment, 0)) / 10.8, 1.0)
                                                        AS log_enrollment_norm,

    -- 6. Study age (recency, exponential decay with 2-year half-life)
    EXP(-COALESCE(si.study_age_days, 0) / 730.0)      AS study_recency,

    -- 7. Late phase indicator (Phase 3/4 → 1, else 0)
    CASE WHEN si.phase IN ('Phase 3', 'Phase 4', 'Phase 2/Phase 3')
         THEN 1.0 ELSE 0.0 END                        AS is_late_phase,

    -- 8. Has DMC (binary)
    CASE WHEN si.has_dmc = 'Yes' THEN 1.0 ELSE 0.0 END AS has_dmc,

    -- 9. Masking level (normalized 0-4 → 0-1)
    CASE di.masking
        WHEN 'None (Open Label)' THEN 0.0
        WHEN 'Single' THEN 0.25
        WHEN 'Double' THEN 0.5
        WHEN 'Triple' THEN 0.75
        WHEN 'Quadruple' THEN 1.0
        ELSE 0.0 END                                  AS masking_norm,

    -- 10. Scale: log num_sites (capped at ~1000 → LN(1001)≈6.9)
    LEAST(LN(1 + COALESCE(sc.num_sites, 0)) / 6.9, 1.0)
                                                        AS log_sites_norm

FROM task t
LEFT JOIN study_info si ON si.nct_id = t.nct_id AND si.task_ts = t.timestamp
LEFT JOIN design_info di ON di.nct_id = t.nct_id AND di.task_ts = t.timestamp
LEFT JOIN site_count sc ON sc.nct_id = t.nct_id AND sc.task_ts = t.timestamp
LEFT JOIN sponsor_history sh ON sh.nct_id = t.nct_id AND sh.task_ts = t.timestamp
LEFT JOIN condition_history ch ON ch.nct_id = t.nct_id AND ch.task_ts = t.timestamp
"""

SITE_SUCCESS_SQL = """
WITH task AS (
    SELECT timestamp, facility_id FROM task_table
),
facility_studies AS (
    SELECT
        t.timestamp AS task_ts, t.facility_id,
        fs.nct_id,
        DATE_DIFF('day', fs.date, t.timestamp) AS days_ago
    FROM task t
    JOIN facilities_studies fs
        ON fs.facility_id = t.facility_id
        AND fs.date < t.timestamp
),
study_counts AS (
    SELECT task_ts, facility_id,
        COUNT(DISTINCT nct_id) AS total_studies,
        COUNT(DISTINCT nct_id) FILTER (WHERE days_ago <= 365) AS studies_1y
    FROM facility_studies
    GROUP BY task_ts, facility_id
),
study_attrs AS (
    SELECT
        fst.task_ts, fst.facility_id,
        AVG(CASE WHEN s.phase IN ('Phase 3', 'Phase 4') THEN 1.0 ELSE 0.0 END) AS frac_late_phase,
        AVG(CASE WHEN s.study_type = 'Interventional' THEN 1.0 ELSE 0.0 END) AS frac_interventional
    FROM facility_studies fst
    JOIN studies s ON s.nct_id = fst.nct_id
    GROUP BY fst.task_ts, fst.facility_id
),
facility_outcomes AS (
    SELECT
        fst.task_ts, fst.facility_id,
        COUNT(*) FILTER (WHERE oa.p_value < 0.05) AS num_significant,
        COUNT(*) AS total_analyses
    FROM facility_studies fst
    JOIN outcome_analyses oa
        ON oa.nct_id = fst.nct_id
        AND oa.date < fst.task_ts
    GROUP BY fst.task_ts, fst.facility_id
),
facility_sponsors AS (
    SELECT
        fst.task_ts, fst.facility_id,
        MAX(CASE WHEN sp.agency_class = 'Industry' THEN 1 ELSE 0 END) AS has_industry_sponsor
    FROM facility_studies fst
    JOIN sponsors_studies ss ON ss.nct_id = fst.nct_id AND ss.date < fst.task_ts
    JOIN sponsors sp ON sp.sponsor_id = ss.sponsor_id
    GROUP BY fst.task_ts, fst.facility_id
)
SELECT
    t.timestamp,
    t.facility_id,

    -- 1. Historical significance rate (ratio in [0,1])
    CASE WHEN COALESCE(fo.total_analyses, 0) > 0
         THEN fo.num_significant::DOUBLE / fo.total_analyses
         ELSE 0.44 END                                 AS significance_rate,

    -- 2. Study volume (log-scaled, capped at ~500)
    LEAST(LN(1 + COALESCE(stc.total_studies, 0)) / 6.2, 1.0)
                                                        AS log_studies_norm,

    -- 3. Recent activity (studies in last year / total)
    CASE WHEN COALESCE(stc.total_studies, 0) > 0
         THEN stc.studies_1y::DOUBLE / stc.total_studies
         ELSE 0.0 END                                  AS recent_study_frac,

    -- 4. Fraction late-phase studies (already [0,1])
    COALESCE(sa.frac_late_phase, 0.0)                  AS frac_late_phase,

    -- 5. Fraction interventional (already [0,1])
    COALESCE(sa.frac_interventional, 0.0)              AS frac_interventional,

    -- 6. Has industry sponsor (binary)
    COALESCE(fsp.has_industry_sponsor, 0)::DOUBLE      AS has_industry_sponsor,

    -- 7. Has any outcome data (binary)
    CASE WHEN fo.facility_id IS NOT NULL THEN 1.0 ELSE 0.0 END
                                                        AS has_outcome_data,

    -- 8. Is new facility (no prior studies)
    CASE WHEN stc.facility_id IS NULL THEN 1.0 ELSE 0.0 END
                                                        AS is_new_facility

FROM task t
LEFT JOIN study_counts stc ON stc.facility_id = t.facility_id AND stc.task_ts = t.timestamp
LEFT JOIN study_attrs sa ON sa.facility_id = t.facility_id AND sa.task_ts = t.timestamp
LEFT JOIN facility_outcomes fo ON fo.facility_id = t.facility_id AND fo.task_ts = t.timestamp
LEFT JOIN facility_sponsors fsp ON fsp.facility_id = t.facility_id AND fsp.task_ts = t.timestamp
"""

STUDY_ADVERSE_SQL = STUDY_OUTCOME_SQL
