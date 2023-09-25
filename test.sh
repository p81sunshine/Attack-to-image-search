#!/bin/bash

# 定义要遍历的文件夹
folder="../../workspace2/imagenet100"

# 使用find寻找所有子文件夹
for sub_folder in $(find ${folder} -mindepth 1 -type d)
do
  # 打印每个子文件夹路径
  modifier="modifier"
  echo $modifier
  subname=${sub_folder:29:38}
  echo $subname
  save="$modifier$subname"
  echo $save
  python3 ../test_attack_black.py --untargeted -a black -d imagenet --reset_adam -n 1 --solver adam -b 2 -p 1 --hash 20 --use_resize --method "tanh" --batch 256 --gpu 0 --lr 0.01 -s $sub_folder:29:38  --start_idx=0 --dist_metrics "pdist" --path $folder --save_ckpts $save 
  
done
