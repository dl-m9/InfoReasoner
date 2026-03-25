import torch
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from IG.calculate import gen_answers_batch, calculate_uncertainty_soft_batch, create_collate_fn, process_item_for_ig
from IG.models.huggingface_models import HuggingfaceModel
from IG.uncertainty_measures.semantic_entropy import EntailmentDeberta

# Import IG client for service testing
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from verl.utils.reward_score.IG_client import IGClient
    IG_CLIENT_AVAILABLE = True
except ImportError:
    IG_CLIENT_AVAILABLE = False
    print("[WARNING] IG client not available, will only test local computation")



model_path = 'Qwen/Qwen2.5-3B'
num_generations = 10 # 10 is good for most cases
sub_batch_size = 10
temperature = 1.0
max_new_tokens = 128
max_context_words = 4096
computation_chunk_size = 8 # adjust to balance speeds and gpu memory cost
prompt_type = 'default'
device = 'cuda:7'

# Build generator
generator = HuggingfaceModel(
    model_path,
    stop_sequences='default',
    max_new_tokens=max_new_tokens,
    torch_dtype=torch.float16,
    device=device,
)
generator.model.eval()

# Build entailment model
entailment_model = EntailmentDeberta(device=device)
entailment_model.model.eval()



example={
    'question': 'when is the last time the philadelphia won the superbowl?',
    'context': """Doc 1(Title: "Philadelphia Eagles") Redskins during the 2017 season for the first time since the 2013 season. The Eagles and Pittsburgh Steelers are both located in Pennsylvania and began play in 1933. From that season, through 1966, this was a major rivalry for both teams as both were part of the same division. In 1967, they were placed in separate divisions but remained in the same conference for three years. In 1970, the Steelers (along with the Cleveland Browns and Baltimore Colts) moved to the American Football Conference while the Eagles stayed with the rest of the old-line NFL teams in the National Football
Doc 2(Title: "Philadelphia Eagles") Philadelphia Eagles The Philadelphia Eagles are a professional American football franchise based in Philadelphia, Pennsylvania. The Eagles compete in the National Football League (NFL) as a member club of the league's National Football Conference (NFC) East division. They are Super Bowl champions, having won Super Bowl LII; their first Super Bowl in franchise history, and their fourth NFL title overall, after winning the Championship Game in 1948, 1949, and 1960. The franchise was established in 1933 as a replacement for the bankrupt Frankford Yellow Jackets, when a group led by Bert Bell secured the rights to an NFL franchise in
Doc 3(Title: "Philadelphia Eagles") starring Danny DeVito makes several references to the Philadelphia Eagles, most notably Season 3, Episode 2 – ""The Gang Gets Invincible,"" the title being a reference to the Wahlberg film. Philadelphia Eagles The Philadelphia Eagles are a professional American football franchise based in Philadelphia, Pennsylvania. The Eagles compete in the National Football League (NFL) as a member club of the league's National Football Conference (NFC) East division. They are Super Bowl champions, having won Super Bowl LII; their first Super Bowl in franchise history, and their fourth NFL title overall, after winning the Championship Game in 1948, 1949, and 1960.""",
    'answers': ['Super Bowl LII', '2017'], # this is the provided ground-truth answers
}
result = gen_answers_batch(example, 
                           generator, 
                           temperature, 
                           num_generations, 
                           sub_batch_size, 
                           max_new_tokens, 
                           prompt_type, 
                           device,
                           max_context_words)

# Baseline example with empty context
example_baseline = example.copy()
example_baseline['context'] = ''

result_baseline = gen_answers_batch(
    example_baseline,
    generator,
    temperature,
    num_generations,
    sub_batch_size,
    max_new_tokens,
    prompt_type,
    device,
    max_context_words
)
print(result)
print("--------------------------------")
print(result_baseline)



#### Calculate IG
keys = ['question', 'response_text', 'answers', 'likelihood', 'context_label', 'log_liks_agg', 'context']
ig_collate_fn = create_collate_fn(keys)

# calculate ig
with torch.no_grad():
    # Convert for IG
    r = process_item_for_ig(result)
    rb = process_item_for_ig(result_baseline)
    ig_input = ig_collate_fn([r, rb])
    ig, ig_baseline = calculate_uncertainty_soft_batch(
        ig_input, entailment_model, computation_chunk_size
    )
    d_ig = ig - ig_baseline


print(f"IG: {ig}")
print(f"IG baseline: {ig_baseline}")
print(f"ΔIG: {d_ig}")

print("\n" + "="*80)
print("Testing IG Service API")
print("="*80)

# Test IG service
if IG_CLIENT_AVAILABLE:
    service_url = "http://0.0.0.0:310"
    print(f"\nConnecting to IG service at: {service_url}")
    
    try:
        client = IGClient(service_url=service_url, timeout=120.0)
        
        # Health check
        print("\n[1] Health check...")
        if client.health_check():
            print("✓ Service is healthy")
        else:
            print("✗ Service is not healthy")
            raise ConnectionError("Service is not healthy")
        
        # Test single item
        print("\n[2] Computing info gain via service...")
        question = example['question']
        context = example['context']
        answers = example['answers']
        
        import time
        start_time = time.time()
        info_gain_service = client.compute_info_gain(
            question=question,
            context=context,
            answers=answers
        )
        end_time = time.time()
        print(f"Info gain service call runtime: {end_time - start_time:.4f} seconds")
        
        print(f"✓ Info gain from service: {info_gain_service}")
        print(f"  Local computation: {d_ig}")
        print(f"  Difference: {abs(info_gain_service - d_ig):.6f}")
        
        # Test batch
        print("\n[3] Testing batch API...")
        batch_items = [
            {
                "question": example['question'],
                "context": example['context'],
                "answers": example['answers']
            },
            {
                "question": example['question'],
                "context": "",  # Baseline
                "answers": example['answers']
            }
        ]
        start_time = time.time()
        batch_scores = client.compute_info_gain_batch(batch_items)
        end_time = time.time()
        print(f"Batch scores service call runtime: {end_time - start_time:.4f} seconds")
        print(f"✓ Batch scores: {batch_scores}")
        
        print("\n" + "="*80)
        print("Service test completed successfully!")
        print("="*80)
        
    except Exception as e:
        print(f"\n✗ Service test failed: {e}")
        import traceback
        traceback.print_exc()
else:
    print("\n[SKIP] IG client not available, skipping service test")