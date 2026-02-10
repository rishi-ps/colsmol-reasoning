"""
ViDoRe dataset loading utilities.

Unified interface for loading ViDoRe v1, v2, and v3 benchmarks
for both training and evaluation.
"""

from typing import Optional
from datasets import load_dataset


# ViDoRe v1 benchmark datasets
VIDORE_V1_DATASETS = [
    "vidore/syntheticDocQA_artificial_intelligence_test",
    "vidore/syntheticDocQA_energy_test",
    "vidore/syntheticDocQA_government_reports_test",
    "vidore/syntheticDocQA_healthcare_industry_test",
    "vidore/docvqa_test_subsampled",
    "vidore/arxivqa_test_subsampled",
    "vidore/infographicsvqa_test_subsampled",
    "vidore/tabfquad_test_subsampled",
    "vidore/tatdqa_test",
    "vidore/shiftproject_test",
]

# ViDoRe v2 benchmark datasets (English-only subset)
VIDORE_V2_ENGLISH_DATASETS = [
    "vidore/Vidore2BioMedicalLecturesRetrieval",
]

# ViDoRe v2 all datasets
VIDORE_V2_ALL_DATASETS = [
    "vidore/Vidore2BioMedicalLecturesRetrieval",
    "vidore/Vidore2ESGReportsHLRetrieval",
    "vidore/Vidore2ESGReportsRetrieval",
    "vidore/Vidore2EconomicsReportsRetrieval",
]

# ViDoRe training dataset
VIDORE_TRAIN_DATASETS = [
    "vidore/colpali_train_set",
]


def load_vidore_dataset(
    version: str = "v1",
    subset: Optional[str] = None,
    split: str = "test",
    streaming: bool = False,
):
    """Load ViDoRe benchmark dataset(s).

    Args:
        version: One of 'v1', 'v2', 'v2_english', 'v3', 'train'
        subset: Specific dataset name to load (if None, returns all for version)
        split: Dataset split to use ('test' for benchmarks, 'train' for training set)
        streaming: Whether to stream the dataset (useful for large training sets)

    Returns:
        Dataset or dict of datasets
    """
    dataset_map = {
        "v1": VIDORE_V1_DATASETS,
        "v2": VIDORE_V2_ALL_DATASETS,
        "v2_english": VIDORE_V2_ENGLISH_DATASETS,
        "train": VIDORE_TRAIN_DATASETS,
    }

    if version not in dataset_map:
        raise ValueError(f"Unknown version '{version}'. Choose from: {list(dataset_map.keys())}")

    datasets_to_load = dataset_map[version]

    if subset:
        matching = [d for d in datasets_to_load if subset in d]
        if not matching:
            raise ValueError(f"No dataset matching '{subset}' in {version}")
        datasets_to_load = matching

    loaded = {}
    for dataset_name in datasets_to_load:
        short_name = dataset_name.split("/")[-1]
        print(f"Loading {short_name} (streaming={streaming})...")
        try:
            loaded[short_name] = load_dataset(dataset_name, split=split, streaming=streaming)
        except Exception as e:
            print(f"  Warning: Failed to load {short_name}: {e}")

    if len(loaded) == 1:
        return next(iter(loaded.values()))

    return loaded
