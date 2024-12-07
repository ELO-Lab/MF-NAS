CONFIG_PATH="configs/algo_asr.yaml"
declare -a optimizers=("RS" "FLS" "BLS" "SH" "REA" "REA+W")
declare -a zc_metrics=("synflow" "params" "flops" "fisher" "grad_norm" "jacob_cov" "l2_norm" "plain" "snip")
declare -a list_k=(16)

for opt in "${optimizers[@]}"
do
    python main.py --ss nbasr --optimizer "$opt" --config_file "$CONFIG_PATH"
done
for metric in "${zc_metrics[@]}"
do
    python scripts/change_config_file.py --object MF-NAS --config_file "$CONFIG_PATH" --attribute metric_stage1 --new_value "$metric"
    for k in "${list_k[@]}"
    do
        python scripts/change_config_file.py --object MF-NAS --config_file "$CONFIG_PATH" --attribute n_candidate --new_value $k
        python main.py --ss nbasr --optimizer MF-NAS --config_file "$CONFIG_PATH" --evaluate_after_search --save_path "./exp/MF-NAS"
    done
done
python scripts/change_config_file.py --object MF-NAS --config_file "$CONFIG_PATH" --attribute metric_stage1 --new_value synflow
python scripts/change_config_file.py --object MF-NAS --config_file "$CONFIG_PATH" --attribute n_candidate --new_value 16