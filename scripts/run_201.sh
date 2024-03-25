CONFIG_PATH="configs/algo_201.yaml"
declare -a optimizers=("RS" "FLS" "BLS" "SH" "REA" "REA+W")
declare -a zc_metrics=("synflow" "params" "flops")
declare -a datasets=("cifar10" "cifar100" "ImageNet16-120")
for dataset in "${datasets[@]}"
do
    for opt in "${optimizers[@]}"
    do
        python main.py --ss nb201 --optimizer "$opt" --config_file "$CONFIG_PATH" --dataset "$dataset"
    done
    for metric in "${zc_metrics[@]}"
    do
        python scripts/change_config_file.py --algo MF-NAS --config_file "$CONFIG_PATH" --key metric_stage1 --value "$metric"
        python main.py --ss nb201 --optimizer MF-NAS --config_file "$CONFIG_PATH" --dataset "$dataset"
    done
done
