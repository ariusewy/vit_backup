python ./cifar10_finetune.py --val_begin --gpu 0  --save_path ../save/pretrain_cifar100 --nohup 20 --val_per 5 \
  --max_epoch 15 --lr 1e-3  --optimizer sgd
  # >Pretrain.txt 2>&1 & 