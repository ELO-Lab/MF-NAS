CONFIG_PATH="configs/algo_101.yaml"
declare -a zc_metrics=("synflow" "params" "flops" "snip" "grasp" "fisher" "grad_norm" "jacob_cov")
declare -a list_k=(16 32 48 64)

for metric in "${zc_metrics[@]}"
do
    python scripts/change_config_file.py --object R-MF-NAS --config_file "$CONFIG_PATH" --attribute metric_stage1 --new_value "$metric"
    for k in "${list_k[@]}"
    do
        python scripts/change_config_file.py --object R-MF-NAS --config_file "$CONFIG_PATH" --attribute n_candidate --new_value $k
        python main.py --ss nb101 --optimizer R-MF-NAS --config_file "$CONFIG_PATH" --evaluate_after_search --save_path "./exp/R-MF-NAS"
    done
done
python scripts/change_config_file.py --object R-MF-NAS --config_file "$CONFIG_PATH" --attribute metric_stage1 --new_value "synflow"
python scripts/change_config_file.py --object R-MF-NAS --config_file "$CONFIG_PATH" --attribute n_candidate --new_value 16