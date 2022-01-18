# $1 $2 $3 $4 data_root pretrained_model output_dir name

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.launch --nproc_per_node=4  train.py --data_root $1 --pretrained_model $2 --output_dir $3 --name $4 --train_batch_size 4 --eval_every 10000 --eval_batch_size 4 --num_steps 100000 --balanced 1 --freeze 1