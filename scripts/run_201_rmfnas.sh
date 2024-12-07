CONFIG_PATH="configs/algo_201.yaml"
declare -a datasets=("cifar10" "cifar100" "ImageNet16-120")
declare -a zc_metrics=("snip" "grasp" "fisher" "plain" "epe_nas" "jacov" "l2_norm" "zen" "grad_norm" "nwot" "synflow" "flops" "params")
declare -a list_k=(32 48 64 96)
for dataset in "${datasets[@]}"
do
    for metric in "${zc_metrics[@]}"
    do
        python scripts/change_config_file.py --object R-MF-NAS --config_file "$CONFIG_PATH" --attribute metric_stage1 --new_value "$metric"
        for k in "${list_k[@]}"
        do
            python scripts/change_config_file.py --object R-MF-NAS --config_file "$CONFIG_PATH" --attribute n_candidate --new_value $k
            python main.py --ss nb201 --optimizer R-MF-NAS --config_file "$CONFIG_PATH" --dataset "$dataset" --evaluate_after_search --save_path "./exp/R-MF-NAS"
        done
    done
done
python scripts/change_config_file.py --object R-MF-NAS --config_file "$CONFIG_PATH" --attribute metric_stage1 --new_value synflow
python scripts/change_config_file.py --object R-MF-NAS --config_file "$CONFIG_PATH" --attribute n_candidate --new_value 32