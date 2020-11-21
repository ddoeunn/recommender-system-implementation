
# Default column names
DEFAULT_USER_COL = "userID"
DEFAULT_ITEM_COL = "itemID"
DEFAULT_RATING_COL = "rating"
DEFAULT_LABEL_COL = "label"
DEFAULT_TIMESTAMP_COL = "timestamp"
DEFAULT_PREDICTION_COL = "prediction"
DEFAULT_HEADER = (
    DEFAULT_USER_COL,
    DEFAULT_ITEM_COL,
    DEFAULT_RATING_COL,
    DEFAULT_TIMESTAMP_COL,
)

COL_DICT = {
    "col_user": DEFAULT_USER_COL,
    "col_item": DEFAULT_ITEM_COL,
    "col_rating": DEFAULT_RATING_COL,
    "col_prediction": DEFAULT_PREDICTION_COL,
}

# Filtering variables
DEFAULT_TOP_K = 10
DEFAULT_THRESHOLD = 10

# Other
SEED = 42
