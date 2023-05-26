cd ..

seeds="20 30 40"

for seed in $seeds
do

CUDA_VISIBLE_DEVICES=0 \
python cal_dist.py \
--input1 #path to the first model, PLM for adapter \
--input2 #path to the second model, PLM for adapter \
--method # adapter or finetune \
--save_path # path to save \

# optional, for adapter
# --adapter1 path to the first adapter module \
# --adapter2 path to the second adapter module \

done
