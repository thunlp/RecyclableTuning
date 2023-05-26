task=chemprot
seeds="20 30 40"
mode=finetune
teacher_name=wb_chemprot
teacher_step=2610
old_plm_path=roberta_base_wb
plm_path=roberta_base_wb_bio
fewnum=32 # the student fewshot number
name=bio_from_wb
alphas="0.9 0.7"
betas="0 5 10 50 100"
Ts="5 10 20"

cd ..

for seed in $seeds
do
for alpha in $alphas
do
for beta in $betas
do
for T in $Ts
do  
    python dataset/get_few.py \
    --seed ${seed} \
    --task ${task} \
    --shot ${fewnum}
    
    sed -e "s/#seed/$seed/g" \
    -e "s/#task/$task/g" \
    -e "s/#teacher_name/$teacher_name/g" \
    -e "s/#teacher_step/$teacher_step/g" \
    -e "s/#old_plm_path/$old_plm_path/g" \
    -e "s/#plm_path/$plm_path/g" \
    -e "s/#fewnum/$fewnum/g" \
    -e "s/#name/$name/g" \
    -e "s/#ALPHA/$alpha/g" \
    -e "s/#BETA/$beta/g" \
    -e "s/#T/$T/g" \
    run_configs/${mode}/distill/distill_task_from_path_template.json \
    > run_configs/${mode}/distill/${task}_from_path.json

    CUDA_VISIBLE_DEVICES=0 \
    python3 ${mode}_kd.py run_configs/${mode}/distill/${task}_from_path.json

done
done
done
done