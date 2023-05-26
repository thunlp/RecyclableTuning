cd ..

tasks="chemprot"
seeds="20 30 40"
modes="adapter finetune"
plm_path=roberta_base_wb_bio
ori_plm_path=roberta_base_wb
applied_name=wb_chemprot
applied_step=6525
name=wb2bio_chemprot

for mode in $modes
do
for task in $tasks
do
for seed in $seeds
do    
    sed -e "s/#seed/$seed/g" \
    -e "s/#task/$task/g" \
    -e "s/#plm_path/$plm_path/g" \
    -e "s/#ori_plm_path/$ori_plm_path/g" \
    -e "s/#applied_name/$applied_name/g" \
    -e "s/#applied_step/$applied_step/g" \
    -e "s/#name/$name/g" \
    run_configs/${mode}/direct_delta/direct_delta_template.json \
    > run_configs/${mode}/direct_delta/${task}_direct_delta.json

    CUDA_VISIBLE_DEVICES=0 \
    python3 direct_delta_${mode}.py run_configs/${mode}/direct_delta/${task}_direct_delta.json

done
done
done