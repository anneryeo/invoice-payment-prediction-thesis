import pandas as pd
from sklearn.utils.validation import check_X_y
from imblearn.utils import check_target_type
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from imblearn.base import BaseSampler


class HybridBalance(BaseSampler):
    """Hybrid undersample + oversample sampler."""

    # Required for sklearn >=1.3
    _parameter_constraints: dict = {
        "undersample_threshold": [float, int],
        "random_state": [int, None],
    }

    # Required by imbalanced-learn
    _sampling_type = "over-sampling"

    def __init__(self, undersample_threshold=0.5, random_state=42):
        super().__init__()
        self.undersample_threshold = undersample_threshold
        self.random_state = random_state
        # imbalanced-learn expects this attribute
        self.sampling_strategy = "auto"

    def _fit_resample(self, X, y):
        """Resample dataset with hybrid strategy."""
        X, y = check_X_y(X, y)
        y = check_target_type(y)

        # Compute class sizes
        class_counts = pd.Series(y).value_counts()
        min_size = class_counts.min()
        target_majority_size = int(min_size / self.undersample_threshold)

        # Undersampling
        under_strategy = {
            cls: min(count, target_majority_size)
            for cls, count in class_counts.items()
        }
        under_sampler = RandomUnderSampler(
            sampling_strategy=under_strategy, random_state=self.random_state
        )
        X_under, y_under = under_sampler.fit_resample(X, y)

        # Oversampling
        over_strategy = {cls: target_majority_size for cls in pd.Series(y_under).unique()}
        over_sampler = RandomOverSampler(
            sampling_strategy=over_strategy, random_state=self.random_state
        )
        X_final, y_final = over_sampler.fit_resample(X_under, y_under)

        return X_final, y_final