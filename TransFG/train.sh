#CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.launch train.py \
#                                     --data_root ../food_data/ \
#                                     --model_type ViT-B_16 \
#                                     --pretrained_dir ./imagenet21k_ViT-B_16.npz  \
#                                     --name sample_run \
#                                     --dataset food \
#                                     --train_batch_size 4 \
#                                     --eval_every 10000 \
#                                     --eval_batch_size 4 \
#                                     --num_steps 100000 \
#                                     --output_dir ./output

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.launch train.py \
                                     --data_root $1 \
                                     --model_type ViT-B_16 \
                                     --pretrained_dir ./ViT-B_16.npz  \
                                     --name sample_run \
                                     --dataset food \
                                     --train_batch_size 4 \
                                     --eval_every 10000 \
                                     --eval_batch_size 4 \
                                     --num_steps 100000 \
                                     --output_dir $2