mkdir llm_attention && cd llm_attention
# Extract the zip → puts src/, Makefile, download_weights.py in place
unzip llm_attention_code.zip
mkdir weights


# 2 — Download GPT-2 Weights (~500 MB, once)

python3 -m venv venv && source venv/bin/activate
pip install transformers torch numpy
python3 download_weights.py
 Produces: weights/gpt2.bin  (~546 MB)

 make
// Uses: gcc -O3 -march=native -fopenmp


# Serial + Parallel + Benchmark (default)
./llm_attn "Hi how are you?"

# Generate 30 tokens
./llm_attn -g 30 "The meaning of life is"

# Force 4 threads, 10 benchmark runs
OMP_NUM_THREADS=4 ./llm_attn -r 10 "Hi how are you?"

# Thread sweep (1,2,4,8)
make sweep

