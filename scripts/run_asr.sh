CONFIG_PATH="configs/algo_asr.yaml"
declare -a optimizers=("RS" "FLS" "BLS" "SH" "REA" "REA+W")
declare -a zc_metrics=("synflow" "params" "FLOPs")

for opt in "${optimizers[@]}"
do
    python main.py --ss nbasr --optimizer "$opt" --config_file "$CONFIG_PATH"
done
for metric in "${zc_metrics[@]}"
do
    python scripts/change_config_file.py --algo MF-NAS  --config_file "$CONFIG_PATH" --key metric_stage1 --value "$metric"
    python main.py --ss nbasr --optimizer MF-NAS --config_file "$CONFIG_PATH"
done
