FINETUNE_NAME=$1
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

CKPT=$FINETUNE_NAME
CONFIG=$SCRIPT_DIR/../../fusion/eval/MMMU/eval/configs/llava1.5.yaml

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python $SCRIPT_DIR/../../fusion/eval/MMMU/eval/run_fusion.py \
        --data_path $SCRIPT_DIR/../../playground/eval/MMMU/images \
        --config_path $CONFIG \
        --model_path $SCRIPT_DIR/../../playground/training/training_dirs/$FINETUNE_NAME \
        --answers-file $SCRIPT_DIR/../../playground/eval/MMMU/answers/$CKPT/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --split "validation" \
        --conv-mode phi_3 & # or llama_3
done

wait

output_file=$SCRIPT_DIR/../../playground/eval/MMMU/answers/$CKPT/merge_val.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat $SCRIPT_DIR/../../playground/eval/MMMU/answers/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

python $SCRIPT_DIR/../../fusion/eval/MMMU/eval/eval.py --result_file $output_file --output_path $SCRIPT_DIR/../../playground/eval/MMMU/$CKPT/val.json