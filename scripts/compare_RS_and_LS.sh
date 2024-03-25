CONFIG_PATH="configs/algo_201.yaml"
declare -a zc_metrics=("plain" "fisher" "zen" "params" "flops" "synflow")
declare -a datasets=("cifar10" "cifar100" "ImageNet16-120")

python scripts/change_config_file.py --algo MF-NAS --config_file "$CONFIG_PATH" --key optimizer_stage1 --value RS
for dataset in "${datasets[@]}"
do
    for metric in "${zc_metrics[@]}"
    do
        python scripts/change_config_file.py --algo MF-NAS --config_file "$CONFIG_PATH" --key metric_stage1 --value "$metric"
        python main.py --ss nb201 --optimizer MF-NAS --config_file "$CONFIG_PATH" --dataset "$dataset"
    done
done
python scripts/change_config_file.py --algo MF-NAS --config_file "$CONFIG_PATH" --key optimizer_stage1 --value FLS
