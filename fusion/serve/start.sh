proxy_off

python -m fusion.serve.controller --host 10.140.24.117 --port 10000

python -m fusion.serve.gradio_web_server --controller http://10.140.24.117:10000 --model-list-mode reload --host 10.140.24.117

python -m fusion.serve.model_worker --host 10.140.24.117 --controller http://10.140.24.117:10000 --port 10010 --worker http://10.140.24.117:10010 --model-path /mnt/petrelfs/liuzheng/TVLM/playground/training/training_dirs/FUSION-Phi3.5-3B-Finetune-2