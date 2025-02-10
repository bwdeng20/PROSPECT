## Environments
```
conda create -n prop python=3.10
conda activate prop

pip install torch==2.0.1  --index-url https://download.pytorch.org/whl/cu121
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cu121.html

pip install -r requirements.txt
```

## Usage
Unzip the data `atk_data.zip` into the folder `PROSPECT/atk_data`.  The config files
of part of experiments are in the folder `PROSPECT/exp_configs`.

To run PROSPECT on the Cora dataset attacked with `MetaAttack`, simply use
```
python prospect_main.py --config exp_configs/cora_metattack_0.15_clean=0_itr=0.0++gnn=sage_hid=64.yaml
```
> [!NOTE]
> All experiments are repeated over five different train/val/test splits.

## Citation
If you find this repository useful in your research, please consider giving a star ⭐ and a citation.
```bib
@inproceedings{deng2024prospect,
    author = {Deng, Bowen and Chen, Jialong and Hu, Yanming and Xu, Zhiyong and Chen, Chuan and Zhang, Tao},
    title = {PROSPECT: Learn MLPs on Graphs Robust against Adversarial Structure Attacks},
    year = {2024},
    isbn = {9798400704369},
    publisher = {Association for Computing Machinery},
    address = {New York, NY, USA},
    url = {https://doi.org/10.1145/3627673.3679857},
    doi = {10.1145/3627673.3679857},
    booktitle = {Proceedings of the 33rd ACM International Conference on Information and Knowledge Management},
    pages = {425–435},
    numpages = {11},
    keywords = {adversarial machine learning, graph heterophily, graph knowledge distillation, graph neural network},
    location = {Boise, ID, USA},
    series = {CIKM '24}
}
```
