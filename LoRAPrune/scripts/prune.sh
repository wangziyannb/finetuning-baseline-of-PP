#!/bin/bash
#BASE_MODEL='baffo32/decapoda-research-llama-7B-hf'
BASE_MODEL='/home/zwang53/data/models/llama2-13b'
prune0='0_loraprune_llama-2-13b_0.2'
prune1='0_loraprune_llama-2-13b_0.4'
prune2='0_loraprune_llama-2-13b_0.6'
prune3='1_loraprune_llama-2-13b_0.2'
prune4='1_loraprune_llama-2-13b_0.4'
prune5='1_loraprune_llama-2-13b_0.6'
prune6='2_loraprune_llama-2-13b_0.2'
prune7='2_loraprune_llama-2-13b_0.4'
prune8='2_loraprune_llama-2-13b_0.6'

BASE_MODEL_7B='/home/zwang53/data/models/llama2-7b'
prune0_7b='0_loraprune_llama-2-7b_0.2'
prune1_7b='0_loraprune_llama-2-7b_0.4'
prune2_7b='0_loraprune_llama-2-7b_0.6'
prune3_7b='1_loraprune_llama-2-7b_0.2'
prune4_7b='1_loraprune_llama-2-7b_0.4'
prune5_7b='1_loraprune_llama-2-7b_0.6'
prune6_7b='2_loraprune_llama-2-7b_0.2'
prune7_7b='2_loraprune_llama-2-7b_0.4'
prune8_7b='2_loraprune_llama-2-7b_0.6'

data(){
  python datas.py --base_model $1 --seed $2 --seq_len $3 --size $4
}

task(){
  CUDA_VISIBLE_DEVICES=$5 python prune.py --seed $4 --base_model $1 --ratio $2 --output_dir outputs_dir/"$3"
}
task_oneshot(){
  CUDA_VISIBLE_DEVICES=$5 python prune.py --seed $4 --base_model $1 --ratio $2 --init_ratio $2 --warmup_iters 0 --cooldown_iters 0 --output_dir outputs_dir/one_shot_pruning/"$3" --batch_size 10 --num_epochs 1 --nsamples 10 --val_set_size 0 --prune_freq 1
}

data "$BASE_MODEL_7B" 0 512 20000 &
#data "$BASE_MODEL_7B" 1 512 20000 &
#data "$BASE_MODEL_7B" 2 512 20000
wait

#task_oneshot "$BASE_MODEL" 0.22 "$prune0" 0 0 &
#task_oneshot "$BASE_MODEL" 0.44 "$prune1" 0 1 &
#task_oneshot "$BASE_MODEL" 0.66 "$prune2" 0 2 &
#task_oneshot "$BASE_MODEL" 0.22 "$prune3" 1 3 &
#task_oneshot "$BASE_MODEL" 0.44 "$prune4" 1 4 &
#task_oneshot "$BASE_MODEL" 0.66 "$prune5" 1 5 &
#task_oneshot "$BASE_MODEL" 0.22 "$prune6" 2 6 &
#task_oneshot "$BASE_MODEL" 0.44 "$prune7" 2 7 &
#task_oneshot "$BASE_MODEL" 0.66 "$prune8" 2 7
wait
#task_oneshot "$BASE_MODEL_7B" 0.22 "$prune0_7b" 0 0 &
#task_oneshot "$BASE_MODEL_7B" 0.44 "$prune1_7b" 0 1 &
#task_oneshot "$BASE_MODEL_7B" 0.66 "$prune2_7b" 0 2 &
#task_oneshot "$BASE_MODEL_7B" 0.22 "$prune3_7b" 1 3 &
#task_oneshot "$BASE_MODEL_7B" 0.44 "$prune4_7b" 1 4 &
#task_oneshot "$BASE_MODEL_7B" 0.66 "$prune5_7b" 1 5 &
#task_oneshot "$BASE_MODEL_7B" 0.22 "$prune6_7b" 2 6 &
#task_oneshot "$BASE_MODEL_7B" 0.44 "$prune7_7b" 2 7 &
#task_oneshot "$BASE_MODEL_7B" 0.66 "$prune8_7b" 2 7
wait

#task "$BASE_MODEL" 0.22 "$prune0" 0 0 &
#task "$BASE_MODEL" 0.44 "$prune1" 0 1 &
#task "$BASE_MODEL" 0.66 "$prune2" 0 2 &
#task "$BASE_MODEL" 0.22 "$prune3" 1 3 &
#task "$BASE_MODEL" 0.44 "$prune4" 1 4 &
#task "$BASE_MODEL" 0.66 "$prune5" 1 5 &
#task "$BASE_MODEL" 0.22 "$prune6" 2 6 &
#task "$BASE_MODEL" 0.44 "$prune7" 2 7
##task "$BASE_MODEL" 0.66 "$prune8" 2 8
wait
#
task "$BASE_MODEL_7B" 0.22 "$prune0_7b" 0 1 &
#task "$BASE_MODEL_7B" 0.44 "$prune1_7b" 0 1 &
#task "$BASE_MODEL_7B" 0.66 "$prune2_7b" 0 2 &
#task "$BASE_MODEL_7B" 0.22 "$prune3_7b" 1 3 &
#task "$BASE_MODEL_7B" 0.44 "$prune4_7b" 1 4 &
#task "$BASE_MODEL_7B" 0.66 "$prune5_7b" 1 5 &
#task "$BASE_MODEL_7B" 0.22 "$prune6_7b" 2 6 &
#task "$BASE_MODEL_7B" 0.44 "$prune7_7b" 2 7
#task "$BASE_MODEL_7B" 0.66 "$prune8_7b" 2 8
wait



echo 'All task finished'
