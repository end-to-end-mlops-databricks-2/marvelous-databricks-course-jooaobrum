import unittest
from dataclasses import dataclass

import numpy as np
import pandas as pd

from components.data_split import DataSplitter


@dataclass
class MockConfig:
    test_size: float = 0.2
    val_size: float = 0.0
    random_state: int = 42
    target: str = None


class TestDataSplitter(unittest.TestCase):
    def set_data(self):
        """Create sample data for testing."""
        # Create a simple dataset with numeric and categorical features
        np.random.seed(42)
        n_samples = 100

        self.data = pd.DataFrame(
            {
                "feature1": np.random.randn(n_samples),
                "feature2": np.random.randint(0, 100, n_samples),
                "category": np.random.choice(["A", "B", "C"], n_samples),
                "target": np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),  # Imbalanced target
            }
        )

        # Create a separate dataset with groups
        group_ids = np.repeat(np.arange(20), 5)  # 20 groups with 5 samples each
        np.random.shuffle(group_ids)

        self.grouped_data = self.data.copy()
        self.grouped_data["group_id"] = group_ids

        # Create config
        self.config = MockConfig()

    def test_random_split(self):
        """Test random splitting functionality."""
        splitter = DataSplitter(self.config)

        # Test with data passed to method
        train, test = splitter.random_split(self.data)

        # Check that the splits have the expected sizes
        expected_test_size = int(len(self.data) * self.config.test_size)
        expected_train_size = len(self.data) - expected_test_size

        self.assertEqual(len(train), expected_train_size)
        self.assertEqual(len(test), expected_test_size)

        # Check that train and test don't overlap
        train_indices = set(train.index)
        test_indices = set(test.index)
        self.assertEqual(len(train_indices.intersection(test_indices)), 0)

        # Check that all data is accounted for
        self.assertEqual(len(train) + len(test), len(self.data))

        # Check that the internal state is updated
        self.assertIsNotNone(splitter.train)
        self.assertIsNotNone(splitter.test)

    def test_stratified_random_split(self):
        """Test stratified splitting functionality."""
        # Set the target in config
        self.config.target = "target"
        splitter = DataSplitter(self.config)

        # Test with data passed to method and target in config
        train, test = splitter.stratified_random_split(data=self.data)

        # Check that the splits have the expected sizes
        expected_test_size = int(len(self.data) * self.config.test_size)
        expected_train_size = len(self.data) - expected_test_size

        self.assertEqual(len(train), expected_train_size)
        self.assertEqual(len(test), expected_test_size)

        # Check stratification - distribution should be similar in train and test
        train_target_dist = train["target"].value_counts(normalize=True)
        test_target_dist = test["target"].value_counts(normalize=True)

        # Check that the class proportions are similar in train and test
        for class_val in self.data["target"].unique():
            train_prop = train_target_dist.get(class_val, 0)
            test_prop = test_target_dist.get(class_val, 0)
            self.assertAlmostEqual(train_prop, test_prop, delta=0.1)
