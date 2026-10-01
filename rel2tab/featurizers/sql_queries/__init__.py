"""SQL feature queries for relbench tasks.

Each per-dataset module exports SQL constants that produce 8-12
pre-normalized features per task, designed for few-shot linear prediction.

Registry maps (dataset_short, task_name) -> {
    "sql": SQL string for feature extraction,
    "entity_col": entity column name,
    "time_col": time column name,
}
"""

from rel2tab.featurizers.sql_queries.rel_amazon import (
    ITEM_CHURN_SQL,
    ITEM_LTV_SQL,
    USER_LTV_SQL,
)
from rel2tab.featurizers.sql_queries.rel_amazon import (
    USER_CHURN_SQL as AMAZON_USER_CHURN_SQL,
)
from rel2tab.featurizers.sql_queries.rel_avito import (
    AD_CTR_SQL,
    USER_CLICKS_SQL,
    USER_VISITS_SQL,
)
from rel2tab.featurizers.sql_queries.rel_event import (
    USER_ATTENDANCE_SQL,
    USER_IGNORE_SQL,
    USER_REPEAT_SQL,
)
from rel2tab.featurizers.sql_queries.rel_f1 import (
    DRIVER_DNF_SQL,
    DRIVER_POSITION_SQL,
    DRIVER_TOP3_SQL,
)
from rel2tab.featurizers.sql_queries.rel_hm import (
    ITEM_SALES_SQL,
)
from rel2tab.featurizers.sql_queries.rel_hm import (
    USER_CHURN_SQL as HM_USER_CHURN_SQL,
)
from rel2tab.featurizers.sql_queries.rel_stack import (
    POST_VOTES_SQL,
    USER_BADGE_SQL,
    USER_ENGAGEMENT_SQL,
)
from rel2tab.featurizers.sql_queries.rel_trial import (
    SITE_SUCCESS_SQL,
    STUDY_ADVERSE_SQL,
    STUDY_OUTCOME_SQL,
)

SQL_REGISTRY: dict[tuple[str, str], dict] = {
    # rel-trial
    ("rel-trial", "study-outcome"): {
        "sql": STUDY_OUTCOME_SQL,
        "entity_col": "nct_id",
        "time_col": "timestamp",
    },
    ("rel-trial", "study-adverse"): {
        "sql": STUDY_ADVERSE_SQL,
        "entity_col": "nct_id",
        "time_col": "timestamp",
    },
    ("rel-trial", "site-success"): {
        "sql": SITE_SUCCESS_SQL,
        "entity_col": "facility_id",
        "time_col": "timestamp",
    },
    # rel-avito
    ("rel-avito", "user-visits"): {
        "sql": USER_VISITS_SQL,
        "entity_col": "UserID",
        "time_col": "timestamp",
    },
    ("rel-avito", "user-clicks"): {
        "sql": USER_CLICKS_SQL,
        "entity_col": "UserID",
        "time_col": "timestamp",
    },
    ("rel-avito", "ad-ctr"): {
        "sql": AD_CTR_SQL,
        "entity_col": "AdID",
        "time_col": "timestamp",
    },
    # rel-f1
    ("rel-f1", "driver-position"): {
        "sql": DRIVER_POSITION_SQL,
        "entity_col": "driverId",
        "time_col": "date",
    },
    ("rel-f1", "driver-dnf"): {
        "sql": DRIVER_DNF_SQL,
        "entity_col": "driverId",
        "time_col": "date",
    },
    ("rel-f1", "driver-top3"): {
        "sql": DRIVER_TOP3_SQL,
        "entity_col": "driverId",
        "time_col": "date",
    },
    # rel-hm
    ("rel-hm", "user-churn"): {
        "sql": HM_USER_CHURN_SQL,
        "entity_col": "customer_id",
        "time_col": "timestamp",
    },
    ("rel-hm", "item-sales"): {
        "sql": ITEM_SALES_SQL,
        "entity_col": "article_id",
        "time_col": "timestamp",
    },
    # rel-stack
    ("rel-stack", "user-engagement"): {
        "sql": USER_ENGAGEMENT_SQL,
        "entity_col": "OwnerUserId",
        "time_col": "timestamp",
    },
    ("rel-stack", "post-votes"): {
        "sql": POST_VOTES_SQL,
        "entity_col": "PostId",
        "time_col": "timestamp",
    },
    ("rel-stack", "user-badge"): {
        "sql": USER_BADGE_SQL,
        "entity_col": "UserId",
        "time_col": "timestamp",
    },
    # rel-amazon
    ("rel-amazon", "user-churn"): {
        "sql": AMAZON_USER_CHURN_SQL,
        "entity_col": "customer_id",
        "time_col": "timestamp",
    },
    ("rel-amazon", "user-ltv"): {
        "sql": USER_LTV_SQL,
        "entity_col": "customer_id",
        "time_col": "timestamp",
    },
    ("rel-amazon", "item-churn"): {
        "sql": ITEM_CHURN_SQL,
        "entity_col": "product_id",
        "time_col": "timestamp",
    },
    ("rel-amazon", "item-ltv"): {
        "sql": ITEM_LTV_SQL,
        "entity_col": "product_id",
        "time_col": "timestamp",
    },
    # rel-event
    ("rel-event", "user-attendance"): {
        "sql": USER_ATTENDANCE_SQL,
        "entity_col": "user",
        "time_col": "timestamp",
    },
    ("rel-event", "user-repeat"): {
        "sql": USER_REPEAT_SQL,
        "entity_col": "user",
        "time_col": "timestamp",
    },
    ("rel-event", "user-ignore"): {
        "sql": USER_IGNORE_SQL,
        "entity_col": "user",
        "time_col": "timestamp",
    },
}
