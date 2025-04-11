gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"
CHUNKS=${#GPULIST[@]}

FINETUNE_NAME=$1
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
SPLIT="MMBench_TEST_CN_legacy"

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m fusion.eval.model_vqa_mmbench \
        --model-path $SCRIPT_DIR/../../playground/training/training_dirs/$FINETUNE_NAME \
        --question-file $SCRIPT_DIR/../../playground/eval/mmbench_cn/$SPLIT.tsv \
        --answers-file $SCRIPT_DIR/../../playground/eval/mmbench_cn/answers/$SPLIT/${FINETUNE_NAME}/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --lang cn \
        --single-pred-prompt \
        --temperature 0 \
        --conv-mode phi_3 & # or llama_3

done

wait

output_file=$SCRIPT_DIR/../../playground/eval/mmbench_cn/answers/$SPLIT/${FINETUNE_NAME}/${FINETUNE_NAME}.jsonl \

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat $SCRIPT_DIR/../../playground/eval/mmbench_cn/answers/$SPLIT/${FINETUNE_NAME}/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

mkdir -p $SCRIPT_DIR/../../playground/eval/mmbench_cn/answers_upload/$FINETUNE_NAME

python $SCRIPT_DIR/../convert_mmbench_for_submission.py \
    --annotation-file $SCRIPT_DIR/../../playground/eval/mmbench_cn/$SPLIT.tsv \
    --result-dir $SCRIPT_DIR/../../playground/eval/mmbench_cn/answers/$SPLIT/${FINETUNE_NAME} \
    --upload-dir $SCRIPT_DIR/../../playground/eval/mmbench_cn/answers_upload/$FINETUNE_NAME \
    --experiment $FINETUNE_NAME
