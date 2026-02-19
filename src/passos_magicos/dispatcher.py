from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

MODEL_DISPATCHER = {
    "random_forest": RandomForestClassifier,
    "xgboost": XGBClassifier,
    "lightgbm": LGBMClassifier,
}
