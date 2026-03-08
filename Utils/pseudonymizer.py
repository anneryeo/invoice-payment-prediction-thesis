import random
import string
import json
import os

class Pseudonymizer:
    def __init__(self, cache_file=r"Database\pseudonym_cache.json"):
        """
        Initialize the pseudonymizer with an optional cache file.
        """
        self.cache_file = cache_file
        self.cache = {}
        self._load_cache()

    def _generate_random_id(self, length=8):
        """Generate a random alphanumeric pseudonym ID."""
        return ''.join(random.choices(string.ascii_uppercase + string.digits, k=length))

    def _load_cache(self):
        """Load cache from JSON file if it exists."""
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "r") as f:
                self.cache = json.load(f)

    def _save_cache(self):
        """Save cache to JSON file."""
        with open(self.cache_file, "w") as f:
            json.dump(self.cache, f, indent=4)

    def pseudonymize(self, df_r, id_col='ID No.'):
        """
        Pseudonymize the IDs in the given DataFrame column.
        
        Parameters:
        - df_r: pandas DataFrame
        - id_col: column name containing IDs to pseudonymize
        
        Returns:
        - DataFrame with pseudonymized IDs
        """
        pseudonymized_ids = []
        
        for original_id in df_r[id_col]:
            if str(original_id) in self.cache:
                # Use cached pseudonym
                pseudonymized_ids.append(self.cache[str(original_id)])
            else:
                # Generate new pseudonym and add to cache
                new_id = self._generate_random_id()
                while new_id in self.cache.values():
                    new_id = self._generate_random_id()
                self.cache[str(original_id)] = new_id
                pseudonymized_ids.append(new_id)
        
        # Save updated cache
        self._save_cache()

        df_r = df_r.copy()
        df_r[id_col] = pseudonymized_ids
        return df_r