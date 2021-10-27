nohup python  ./cifar_ft_test.py \
--batch_size 8 \
--gpu 1 \
--lr 1e-3 \
--updata_per 30 \
--shift-depth 1000 --shift-type PS --optimizer radam > 30_head_score.txt

# nohup python  ./cifar_ft_test.py \
# --batch_size 8 \
# --gpu 1 \
# --lr 1e-3 \
# --updata_per 10 \
# --save_path ./save/addQ_dim1_average \
# --shift-depth 1000 --shift-type PS --optimizer radam > 1023_addQ_dim1_average_from2.txt 2>&1 &
