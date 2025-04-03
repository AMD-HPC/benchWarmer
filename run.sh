make amd
./benchWarmer-amd | tee output.log
if [ -z "$ROCM_PATH" ]; then
  echo "\$ROCM_PATH not found. Set \$ROCM_PATH to publish results."
  exit
fi
python3 fom.py output.log
