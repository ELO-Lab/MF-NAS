CONFIG_PATH="configs/algo_201.yaml"
declare -a zc_metrics=("plain" "fisher" "zen" "params" "flops" "synflow")
declare -a datasets=("cifar10" "cifar100" "ImageNet16-120")

python scripts/change_config_file.py --object MF-NAS --config_file "$CONFIG_PATH" --attribute optimizer_stage1 --new_value RS
for dataset in "${datasets[@]}"
do
    for metric in "${zc_metrics[@]}"
    do
        python scripts/change_config_file.py --object MF-NAS --config_file "$CONFIG_PATH" --attribute metric_stage1 --new_value "$metric"
        python main.py --ss nb201 --optimizer MF-NAS --config_file "$CONFIG_PATH" --dataset "$dataset"
    done
done
python scripts/change_config_file.py --object MF-NAS --config_file "$CONFIG_PATH" --attribute optimizer_stage1 --new_value FLS
