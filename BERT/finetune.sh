CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.launch --nproc_per_node 4 train.py --data_root ../food_data/ --dataset food --model_type ViT-B_16 --pretrained_model scratch/scratch_run_checkpoint.bin  --name finetune --train_batch_size 2 --eval_every 10000 --eval_batch_size 2 --num_steps 100000 --output_dir ./finetune --freeze 1 --balanced 1