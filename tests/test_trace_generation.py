
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.reasoning.trace_generator import TraceGenerator, TraceConfig

def test_generation():
    print("Initializing TraceGenerator...")
    config = TraceConfig(
        model_name="Qwen/Qwen2.5-0.5B-Instruct",
        device="cuda",
        max_new_tokens=50,
        batch_size=2
    )
    generator = TraceGenerator(config).load()
    
    queries = [
        "What is the total revenue in Q3?",
        "Find the chart showing user growth."
    ]
    
    print(f"Generating traces for {len(queries)} queries...")
    traces = generator.generate_traces_batch(queries)
    
    for q, t in zip(queries, traces):
        print(f"\nQuery: {q}")
        print(f"Trace: {t}")
        assert len(t) > 0, "Trace should not be empty"
        
    print("\nTest passed!")

if __name__ == "__main__":
    test_generation()
