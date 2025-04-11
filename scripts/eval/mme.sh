gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"
CHUNKS=${#GPULIST[@]}

FINETUNE_NAME=$1
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m fusion.eval.model_vqa_loader \
        --model-path $SCRIPT_DIR/../../playground/training/training_dirs/$FINETUNE_NAME \
        --question-file $SCRIPT_DIR/../../playground/eval/MME/llava_mme.jsonl \
        --image-folder $SCRIPT_DIR/../../playground/eval/MME/MME_Benchmark_release_version \
        --answers-file $SCRIPT_DIR/../../playground/eval/MME/answers/${FINETUNE_NAME}/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode phi_3 & # or llama_3

done

wait

output_file=$SCRIPT_DIR/../../playground/eval/MME/answers/${FINETUNE_NAME}.jsonl \

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat $SCRIPT_DIR/../../playground/eval/MME/answers/${FINETUNE_NAME}/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

cd $SCRIPT_DIR/../../playground/eval/MME

python convert_answer_to_mme.py --experiment $FINETUNE_NAME

cd eval_tool

python calculation.py --results_dir answers/$FINETUNE_NAME
