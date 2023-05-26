# RecyclableTuning
We have released the code for the most of the main experiments in paper *Recyclable Tuning for Lifelong Pre-training*, including two recycle methods for adapter-tuning and fine-tuning on $RoBERTa_\texttt{BASE}$.

## Dataset and Checkpoint Preparation

We follow the datasets used in *Dont't-Stop-Pretraining* for domain-related tasks. We also use various other datasets to support our experiment. Please download the dataset you need here and put them under `./dataset`. We have upload task `ChemProt` as an example. 

Pre-trained models of $RoBERTa_\texttt{BASE}$ could be downloaded here. The series of models simulate the lifelong pre-training scenario.  Please put all the pre-trained models you need under `./pretrained_models`.

## Pilot Studies

All the scripts for the experiments are in `./run_scripts`, while all the hyper-parameters we use are listed in files in `./run_configs`. Here we only provide example scripts and example configurations. More experiments can be conducted by changing the scripts and hyper-parameters accordingly.

Also note that we have added some files in the transformers package in order to realize the models' functions. In order to successfully run the experiment, please do not change files in `./transformers`.  It is based on transformers version `4.18.0`.

### Normal Tuning

To fine-tune or adapter-tune a model normally, please use the script `run.sh`. The example script tune the task `ChemProt` using different seeds. This script provides method to tune arbitrary tasks from arbitrary pre-trained models. Please refer to `base_model/task_from_path_template.json` in `run_configs` if you would like to change the default hyper-parameters like learning rate, batch size, etc.

```bash
cd run_scripts
bash run.sh
```

If you would like to perform few-shot tuning, please add the followings to the script first to generate the few-shot dataset:

```bash
python dataset/get_few.py \
--seed ${seed} \
--task ${task} \
--shot ${fewnum}
```

Then, change the dataset paths in the template accordingly before start your tuning.

### Direct Apply Delta

To apply the changes during the original training process to a new pre-trained model, please use the script `direct_apply_delta.sh`. `old_plm_path` indicates the start model of the previous training process, and it is valid only when performing fine-tuning.  `applied_name` indicates the name of previous tuned model. They will be used to calculate the change ("delta") of model. `plm_path` indicates the start model of current process, to which the change ("delta") of model will be applied. For more hyper-parameters please refer to `direct_delta/direct_delta_template.json` in `run_configs`.

```bash
cd run_scripts
bash direct_apply_delta.sh
```

### Interpolation

To interpolate two models and check the evaluation & test scores, please use the scripts `check_itp.sh`. Indexes 1 and 2 refer to the configurations of two models to be interpolated. `plm_path` are valid only when performing adapter-tuning, as we need the parameters of model backbones only in adapter-tuning. Interpolation can be performed after we get at least two base models with same size and model configurations. For more hyper-parameters please refer to `check_itp/check_itp_template.json` in `run_configs`. 

```bash
cd run_scripts
bash check_itp.sh
```

## Main Experiments

### Distillation

To distill the knowledge of a teacher model to students, please use the script `run_scripts.sh`. As all our main experiments use few-shot dataset to tune the student, we use `fewnum` to indicate the few-shot number. The original teacher model is identified by `teacher_name` and `teacher_step`. `alpha` and `beta` are hyper-parameters to regulate the relative proportions of task loss, distill loss and patience loss, while `T` refers to the distill temperature. For more hyper-parameters please refer to `distill/distill_task_from_path_template.json` in `run_configs`. 

```bash
cd run_scripts
bash run_distill.sh
```

### Delta Initialization

To apply the tuned model as current training process's initialization, please use the script `run_oldinit.sh`. `oldinit_path`, `oldinit_step` and `old_plm_path` will help identify the previous tuned model and calculate the change ("delta"), which will be applied to the model from `plm_path` as initialization. For more  hyper-parameters please refer to `base_model/task_from_path_oldinit_template.json` in `run_configs`. 

```bash
cd run_scripts
bash run_oldinit.sh
```

### Combined Method

To combine the "delta" of teacher as initialization and distill knowledge from teacher model, please use the scripts `run_distill_oldinit.sh`. The meanings of the hyper-parameters are same as those in distillation, and the only change is that the teacher model is applied in current training's initialization. Please refer to `distill/distill_task_from_path_oldinit_template.json` for more information about the hyper-parameters.

```bash
cd run_scripts
bash run_distill_oldinit.sh
```

## Auxiliary Scripts

### Calculate L2 Distance

Please use the scripts `cal_dist`.sh if you want to calculate the L2 distance of two models with same configurations. `adapter_model` is optional and only valid when calculating the distance between adapter modules.

## Acknowledgement

The authors would like to thank anonymous reviewers for their valuable feedbacks.





