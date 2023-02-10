# export CUDA_VISIBLE_DEVICES=0

SCENE="ship_re"

EXPERIMENT=blender
CHECKPOINT_DIR=./results/"$EXPERIMENT"/$SCENE

python -m train \
  --gin_configs=configs/blender_256.gin \
  --gin_bindings="Config.checkpoint_dir = '${CHECKPOINT_DIR}'" \
  --logtostderr

