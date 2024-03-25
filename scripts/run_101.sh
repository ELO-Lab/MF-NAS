CONFIG_PATH="configs/algo_101.yaml"
declare -a optimizers=("RS" "FLS" "BLS" "SH" "REA" "REA+W")
declare -a zc_metrics=("synflow" "params" "flops")

for opt in "${optimizers[@]}"
do
    python main.py --ss nb101 --optimizer "$opt" --config_file "$CONFIG_PATH"
done
for metric in "${zc_metrics[@]}"
do
    python scripts/change_config_file.py --algo MF-NAS --config_file "$CONFIG_PATH" --key metric_stage1 --value "$metric"
    python main.py --ss nb101 --optimizer MF-NAS --config_file "$CONFIG_PATH"
done
