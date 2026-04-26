import os
import joblib
import hashlib
import pandas as pd
from pathlib import Path

class CacheManager:
    """Manages disk-based caching for prepared datasets and models."""
    
    def __init__(self, cache_root="data/cache"):
        self.cache_root = Path(cache_root)
        self.dataset_dir = self.cache_root / "datasets"
        self.model_dir = self.cache_root / "models"
        
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)

    def _generate_key(self, **kwargs):
        """Generates a deterministic MD5 hash for a set of parameters."""
        # Sort keys to ensure consistent hashing
        items = sorted(kwargs.items())
        key_string = str(items)
        return hashlib.md5(key_string.encode()).hexdigest()

    def get_dataset(self, balance_strategy, threshold, observation_end, test_size):
        """Retrieves a prepared dataset from cache if it exists."""
        key = self._generate_key(
            strategy=balance_strategy,
            threshold=threshold,
            obs_end=str(observation_end),
            test_size=test_size
        )
        cache_path = self.dataset_dir / f"dataset_{key}.joblib"
        
        if cache_path.exists():
            return joblib.load(cache_path)
        return None

    def save_dataset(self, dataset, balance_strategy, threshold, observation_end, test_size):
        """Saves a prepared dataset to cache."""
        key = self._generate_key(
            strategy=balance_strategy,
            threshold=threshold,
            obs_end=str(observation_end),
            test_size=test_size
        )
        cache_path = self.dataset_dir / f"dataset_{key}.joblib"
        joblib.dump(dataset, cache_path)
        return cache_path

    def clear_cache(self):
        """Removes all files in the cache directory."""
        import shutil
        if self.cache_root.exists():
            shutil.rmtree(self.cache_root)
            self.cache_root.mkdir(parents=True, exist_ok=True)
            self.dataset_dir.mkdir(parents=True, exist_ok=True)
            self.model_dir.mkdir(parents=True, exist_ok=True)
