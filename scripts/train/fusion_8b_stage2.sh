export OMP_NUM_THREADS=8
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=eth0
export NCCL_DEBUG=INFO
export GPUS_PER_NODE=8
export NNODES=4
export MASTER_PORT=29705
export CPUS_PER_TASK=16

export SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

export IMAGE_PATH=$SCRIPT_DIR/../../playground/train/images

export PRETRAIN_NAME=FUSION-LLaMA3.1-8B-Stage1
export MEDIUM_NAME=FUSION-LLaMA3.1-8B-Stage1.5
export FINETUNE_NAME=FUSION-LLaMA3.1-8B

export NUM_TRAIN_EPOCHS=1

mkdir -p $SCRIPT_DIR/../../playground/training/training_dirs/$FINETUNE_NAME

srun -p Your Partion \
    --nodes=$NNODES \
    --ntasks-per-node=1 \
    --gres=gpu:$GPUS_PER_NODE \
    --cpus-per-task=$CPUS_PER_TASK \
    --kill-on-bad-exit=1 \
    bash -c 'ACCELERATE_CPU_AFFINITY=1 torchrun --nnodes $NNODES --nproc_per_node $GPUS_PER_NODE --node_rank $SLURM_NODEID --master_addr $(scontrol show hostname $SLURM_NODELIST | head -n1) --master_port ${MASTER_PORT} $SCRIPT_DIR/../../fusion/train/train_mem.py \
    --deepspeed $SCRIPT_DIR/../zero2.json \
    --model_name_or_path $SCRIPT_DIR/../../playground/training/training_dirs/$MEDIUM_NAME \
    --version llama_3 \
    --data_path $SCRIPT_DIR/yaml/stage2.yaml \
    --image_folder $IMAGE_PATH \
    --vision_tower google/siglip-so400m-patch14-384 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end True \
    --mm_use_im_patch_token False \
    --unfreeze_mm_vision_tower True \
    --window_size 3 \
    --query_len "4,16,36,64,144" \
    --num_of_vision_sampler_layers 10 \
    --start_of_vision_sampler_layers 0 \
    --stride_of_vision_sampler_layers 3 \
    --image_aspect_ratio static_hd \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir $SCRIPT_DIR/../../playground/training/training_dirs/$FINETUNE_NAME \
    --num_train_epochs $NUM_TRAIN_EPOCHS \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 1 \
    --learning_rate 1e-5 \
    --mm_vision_tower_lr 1e-6 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to none \
    2>&1 | tee $SCRIPT_DIR/../../playground/training/training_dirs/$FINETUNE_NAME/output.log'