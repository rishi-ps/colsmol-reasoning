"""
Memory-efficient ViDoRe evaluation for ColSmol models.
Uses CPU-based chunked similarity computation to avoid GPU OOM.
"""
import torch
import mteb
from colpali_engine.models import ColIdefics3, ColIdefics3Processor
from mteb.models.model_implementations.colpali_models import ColPaliEngineWrapper
import gc


class ColSmolMemoryEfficientWrapper(ColPaliEngineWrapper):
    """ColSmol wrapper with CPU-based similarity computation."""
    
    def __init__(
        self,
        model_name: str = "vidore/colSmol-256M",
        revision: str | None = None,
        device: str | None = None,
        **kwargs,
    ):
        # Call parent init
        super().__init__(
            model_name=model_name,
            model_class=ColIdefics3,
            processor_class=ColIdefics3Processor,
            revision=revision,
            device=device,
            **kwargs,
        )
    
    def similarity(self, a, b):
        """Compute MaxSim on CPU in chunks to avoid OOM."""
        # Move to CPU and convert to float32
        a_cpu = a.cpu().float() if a.is_cuda else a.float()
        b_cpu = b.cpu().float() if b.is_cuda else b.float()
        
        # Clear GPU memory
        torch.cuda.empty_cache()
        gc.collect()
        
        num_queries = a_cpu.shape[0]
        num_docs = b_cpu.shape[0]
        
        # Process in small chunks to avoid CPU memory issues too
        chunk_size = 32
        all_scores = []
        
        for i in range(0, num_queries, chunk_size):
            q_chunk = a_cpu[i:i + chunk_size]
            row_scores = []
            
            for j in range(0, num_docs, chunk_size):
                d_chunk = b_cpu[j:j + chunk_size]
                # MaxSim: einsum("bnd,csd->bcns") -> max over s, sum over n
                sim = torch.einsum("qnd,psd->qpns", q_chunk, d_chunk)
                scores = sim.max(dim=-1)[0].sum(dim=-1)
                row_scores.append(scores)
            
            all_scores.append(torch.cat(row_scores, dim=1))
        
        return torch.cat(all_scores, dim=0)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate ColSmol on ViDoRe with memory-efficient scoring")
    parser.add_argument("--model", default="vidore/colSmol-256M", help="Model to evaluate")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size for encoding")
    parser.add_argument("--output-folder", default="results", help="Output folder for results")
    parser.add_argument("--benchmark", default="ViDoRe(v1)", help="Benchmark name")
    args = parser.parse_args()
    
    print(f"Loading model: {args.model}")
    model = ColSmolMemoryEfficientWrapper(
        model_name=args.model,
        torch_dtype=torch.bfloat16,
    )
    
    print(f"Running benchmark: {args.benchmark}")
    tasks = mteb.get_benchmark(args.benchmark)
    
    # Create cache for results
    cache = mteb.ResultCache(cache_path=args.output_folder)
    
    # Run one task at a time and collect results
    all_results = []
    for task in tasks:
        print(f"\nEvaluating: {task.metadata.name}")
        result = mteb.evaluate(
            model,
            tasks=task,
            cache=cache,
            encode_kwargs={"batch_size": args.batch_size}
        )
        all_results.append(result)
    
    print("\n" + "="*50)
    print("RESULTS")
    print("="*50)
    for result in all_results:
        print(f"\n{result.task_name}:")
        for split, scores in result.scores.items():
            if isinstance(scores, list) and len(scores) > 0:
                # Get the main ndcg_at_5 score
                if isinstance(scores[0], dict) and 'ndcg_at_5' in scores[0]:
                    print(f"  {split}: NDCG@5 = {scores[0]['ndcg_at_5']:.4f}")
                else:
                    print(f"  {split}: {scores}")
            else:
                print(f"  {split}: {scores}")


if __name__ == "__main__":
    main()
