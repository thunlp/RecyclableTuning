cd ..
task=chemprot
seeds="20 30 40"
modes="finetune"
model1_name=wb_chemprot
model2_name=wb_chemprot
model1_step=2610
model2_step=2610
plm1_path=roberta_base_wb
plm2_path=roberta_base_wb_bio
name=wb2bio_chemprot

for mode in $modes
do
for seed in $seeds
do

    sed -e "s/#seed/$seed/g" \
    -e "s/#task/$task/g" \
    -e "s/#model1_name/$model1_name/g" \
    -e "s/#model2_name/$model2_name/g" \
    -e "s/#model1_step/$model1_step/g" \
    -e "s/#model2_step/$model2_step/g" \
    -e "s/#plm1_path/$plm2_path/g" \
    -e "s/#plm2_path/$plm2_path/g" \
    -e "s/#name/$name/g" \
    run_configs/${mode}/check_itp/check_itp_template.json \
    > run_configs/${mode}/check_itp/${task}_check_itp.json \
    
    CUDA_VISIBLE_DEVICES=0 \
    python3 ${mode}_itp.py ./run_configs/${mode}/check_itp/${task}_check_itp.json

done
done