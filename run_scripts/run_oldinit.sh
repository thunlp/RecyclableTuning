cd ..

task=chemprot
seeds="20 30 40"
mode=finetune
oldinit_path=wb_chemprot
oldinit_step=2610
old_plm_path=roberta_base_wb # used only in adapter mode
plm_path=roberta_base_wb_bio
name=bio_from_wb

for seed in $seeds
do  
    
    sed -e "s/#seed/$seed/g" \
    -e "s/#task/$task/g" \
    -e "s/#oldinit_path/$oldinit_path/g" \
    -e "s/#oldinit_step/$oldinit_step/g" \
    -e "s/#old_plm_path/$old_plm_path/g" \
    -e "s/#plm_path/$plm_path/g" \
    -e "s/#name/$name/g" \
    run_configs/${mode}/base_model/task_from_path_oldinit_template.json \
    > run_configs/${mode}/base_model/${task}_from_path_oldinit.json

    CUDA_VISIBLE_DEVICES=0 \
    python3 ${mode}_run_oldinit.py run_configs/${mode}/base_model/${task}_from_path_oldinit.json

done