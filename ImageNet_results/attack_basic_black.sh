#!/bin/bash

# 定义要遍历的文件夹
folder="../../../autodl-fs/"

echo $folder
# 使用find寻找所有子文件夹
for sub_folder in $(find ${folder} -mindepth 1 -type d)
do
  # 打印每个子文件夹路径
  modifier="MM"
  subname=${sub_folder:19:27}
  echo $subname
  save="$modifier$subname"
  echo $save
  python3 ../test_attack_black.py --untargeted -a black -d imagenet --reset_adam -n 5 --solver adam -b 2 -p 1 --hash 20 --use_resize --method "tanh" --batch 256 --gpu 0 --lr 0.01 -s $subname  --start_idx=0 --dist_metrics "pdist" --path $sub_folder --save_ckpts $save 
  
done
