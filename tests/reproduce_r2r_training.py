
import sys
import os
import torch
from torch.utils.data import DataLoader
from PIL import Image

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.colsmol import ColSmolWrapper, ColSmolConfig
from src.reasoning.augmented_retriever import ReasoningAugmentedRetriever, R2RConfig
from src.reasoning.trainer import R2RTrainer, R2RTrainerConfig

def get_dummy_data():
    """Create dummy data mimicking R2RDataset format."""
    # Create a small white image
    img = Image.new('RGB', (224, 224), color='white')
    
    return [
        {
            'query': "What is the capital of France?",
            'trace': "Look for a map or text mentioning Paris.",
            'image': img
        },
        {
            'query': "Show me Q3 revenue.",
            'trace': "Look for a table with Q3 column and Revenue row.",
            'image': img
        }
    ] * 4 # 8 examples total

def collate_fn(batch):
    return {
        'query': [x['query'] for x in batch],
        'trace': [x['trace'] for x in batch],
        'image': [x['image'] for x in batch],
    }

def main():
    print("Initializing ColSmol (256M)...")
    # Config matching what we think works (float16 + gradient checkpointing)
    retriever_cfg = ColSmolConfig(
        model_name="vidore/colSmol-256M",
        use_lora=True,
        lora_rank=16,
        device="cuda" if torch.cuda.is_available() else "cpu",
        dtype=torch.float16 
    )
    retriever = ColSmolWrapper(retriever_cfg).load()
    
    # R2R Wrapper
    r2r = ReasoningAugmentedRetriever(retriever)
    
    # Trainer Config
    trainer_cfg = R2RTrainerConfig(
        learning_rate=1e-5,
        batch_size=2,
        gradient_accumulation_steps=1,
        num_epochs=1,
        output_dir="checkpoints/debug_r2r",
        log_every_n_steps=1
    )
    
    trainer = R2RTrainer(r2r, trainer_cfg)
    
    # Data
    print("Creating dummy dataset...")
    data = get_dummy_data()
    loader = DataLoader(data, batch_size=2, collate_fn=collate_fn)
    
    # Train
    print("Starting training loop (5 steps max)...")
    trainer.setup()
    trainer.retriever.retriever.model.train()
    
    for i, batch in enumerate(loader):
        print(f"Step {i+1}")
        loss = trainer.train_step(batch)
        print(f"  Loss: {loss['loss']}")
        
        if i >= 5:
            break

if __name__ == "__main__":
    main()
