CONFIG_PATH="configs/algo_asr.yaml"
declare -a optimizers=("RS" "FLS" "BLS" "SH" "REA" "REA+W")
declare -a zc_metrics=("synflow" "params" "FLOPs")

for opt in "${optimizers[@]}"
do
    python main.py --ss nbasr --optimizer "$opt" --config_file "$CONFIG_PATH"
done
for metric in "${zc_metrics[@]}"
do
    python scripts/change_config_file.py --object MF-NAS  --config_file "$CONFIG_PATH" --attribute metric_stage1 --new_value "$metric"
    python main.py --ss nbasr --optimizer MF-NAS --config_file "$CONFIG_PATH"
done
