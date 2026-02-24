import numpy as np
from src.utils.logger import logger


def get_feature_importances(model):
    """
    Extracts feature importances from an arbitrary trained machine learning model.
    Handles tree-based models, linear models, and XGBoost specifically.

    Args:
        model: Trained model artifact.

    Returns:
        tuple[list, list]: A tuple containing (feature_names, importances).
                           Returns (None, None) if extraction is not supported.
    """
    try:
        # Scikit-learn Tree-based models (e.g., RandomForest, GradientBoosting)
        if hasattr(model, "feature_importances_") and hasattr(
            model, "feature_names_in_"
        ):
            logger.info("Extracting feature importances from tree-based model.")
            return list(model.feature_names_in_), list(model.feature_importances_)

        # Scikit-learn Linear models (e.g., Ridge, Lasso, ElasticNet)
        elif hasattr(model, "coef_") and hasattr(model, "feature_names_in_"):
            logger.info(
                "Extracting feature importances from linear model coefficients."
            )
            # For linear models, the absolute magnitude of coefficients determines importance
            return list(model.feature_names_in_), list(np.abs(model.coef_))

        # Direct XGBoost Booster object extraction
        elif hasattr(model, "get_booster"):
            logger.info("Extracting feature importances via XGBoost booster gain.")
            booster = model.get_booster()
            importance_map = booster.get_score(importance_type="gain")
            return list(importance_map.keys()), list(importance_map.values())

        else:
            logger.warning(
                f"Feature importance extraction not supported for {type(model).__name__}."
            )
            return None, None

    except Exception as e:
        logger.error(f"Error occurred while extracting feature importances: {e}")
        return None, None
