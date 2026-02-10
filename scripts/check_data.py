from datasets import load_dataset_builder, get_dataset_config_names
from huggingface_hub import list_datasets

print("Checking for ViDoRe training datasets...")

# Check if 'test' datasets have a 'train' split
dataset_name = "vidore/syntheticDocQA_artificial_intelligence_test"
try:
    ds_builder = load_dataset_builder(dataset_name)
    print(f"Splits for {dataset_name}: {list(ds_builder.info.splits.keys())}")
except Exception as e:
    print(f"Error checking splits for {dataset_name}: {e}")

# Search for other vidore datasets that might be for training
print("\nSearching for 'vidore' datasets with 'train' in name...")
try:
    datasets = list_datasets(author="vidore", limit=20, sort="downloads", direction=-1)
    for ds in datasets:
        if "train" in ds.id.lower():
            print(f"  {ds.id}")
except Exception as e:
    print(f"Error listing datasets: {e}")
