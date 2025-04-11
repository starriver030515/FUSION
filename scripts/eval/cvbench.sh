gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"
CHUNKS=${#GPULIST[@]}

FINETUNE_NAME=$1
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m fusion.eval.model_vqa_cvbench \
        --model-path $SCRIPT_DIR/../../playground/training/training_dirs/$FINETUNE_NAME \
        --question-file $SCRIPT_DIR/../../playground/eval/CVBench/test.jsonl \
        --image-folder $SCRIPT_DIR/../../playground/eval/CVBench/img \
        --answers-file $SCRIPT_DIR/../../playground/eval/CVBench/answers/${FINETUNE_NAME}/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --single-pred-prompt \
        --temperature 0 \
        --conv-mode phi_3 & # or llama_3

done

wait

output_file=$SCRIPT_DIR/../../playground/eval/CVBench/answers/${FINETUNE_NAME}/output.jsonl \

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat $SCRIPT_DIR/../../playground/eval/CVBench/answers/${FINETUNE_NAME}/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done


python $SCRIPT_DIR/../../fusion/eval/eval_cvbench_qa.py \
    --base-dir $SCRIPT_DIR/../../playground/eval/cvbench \
    --output-file $output_file
