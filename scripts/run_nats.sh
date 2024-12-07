CONFIG_PATH="configs/algo_nats.yaml"
declare -a zc_metrics=("synflow" "fisher" "params" "flops" "jacov" "plain" "grasp" "epe_nas" "grad_norm" "snip" "l2_norm" "zen" "nwot" "meco_opt" "zico" "swap")
declare -a datasets=("cifar10" "cifar100" "ImageNet16-120")
declare -a list_k=(32 48 64 96)

for dataset in "${datasets[@]}"
do
    for metric in "${zc_metrics[@]}"
    do
        python scripts/change_config_file.py --object MF-NAS --config_file "$CONFIG_PATH" --attribute metric_stage1 --new_value "$metric"
        for k in "${list_k[@]}"
        do
            python scripts/change_config_file.py --object MF-NAS --config_file "$CONFIG_PATH" --attribute n_candidate --new_value $k
            python main.py --ss nats --optimizer MF-NAS --dataset $dataset --config_file "$CONFIG_PATH" --evaluate_after_search
        done
    done
done
python scripts/change_config_file.py --object MF-NAS --config_file "$CONFIG_PATH" --attribute metric_stage1 --new_value synflow
python scripts/change_config_file.py --object MF-NAS --config_file "$CONFIG_PATH" --attribute n_candidate --new_value 32