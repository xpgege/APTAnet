# APTAnet
毕业设计：基于深度学习的PTI亲和力预测算法

1.准备环境
conda env create -f environment.yaml

2.准备数据
参考https://github.com/PaccMann/TITAN

3.训练数据
 CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run --nnodes=1 --nproc_per_node=2 --node_rank=0  train.py 
 train_affinity_filepath test_affinity_filepath receptor_filepath  ligand_filepath  output_dir training_name  pretrain_weights_path
 
 
