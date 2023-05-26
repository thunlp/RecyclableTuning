cd ..

task=chemprot
seeds="20 30 40"
plm_path=roberta_base_wb
name=wb_chemprot
modes="adapter finetune"

for mode in $modes
do
for seed in $seeds
do
    sed -e "s/#seed/$seed/g" \
    -e "s/#task/$task/g" \
    -e "s/#name/$name/g" \
    -e "s/#plm_path/$plm_path/g" \
    run_configs/${mode}/base_model/task_from_path_template.json \
    > run_configs/${mode}/base_model/${task}_from_path.json

    CUDA_VISIBLE_DEVICES=0 \
    python3 ${mode}_run.py run_configs/${mode}/base_model/${task}_from_path.json
done
done