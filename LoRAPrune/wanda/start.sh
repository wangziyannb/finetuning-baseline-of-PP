CUDA_VISIBLE_DEVICES=3 python main.py \
    --model '/data8T/zmy/LLM/llama-7b-hf/' \
    --prune_method "wanda" \
    --sparsity_ratio 0.5 \
    --sparsity_type "unstructured" \
