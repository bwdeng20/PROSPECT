import platform
import logging
from functools import partial
import colorlog
import numpy as np
import pandas as pd
import rootutils
import torch
from jsonargparse import ArgumentParser, ActionConfigFile
from torch_geometric import compile as pyg_compile
from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import (EarlyStopping, LearningRateMonitor,
                                         ModelCheckpoint)
from lightning.pytorch.loggers import TensorBoardLogger

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.data.gnn_node_datamodule import UnifiedAtkNodeData
from src.models.components.gnns import GraphSAGE, GCN
from src.models.components.mlps import MLP, JKMLP
from src.models.gd_module import GDMLPLitModule
from src.models.components.lr_schedulers import CosineAnnealingColdRestart

OsType = platform.system()
stream_handler = logging.StreamHandler()
stream_handler.setLevel("DEBUG")
log_colors = {
    'DEBUG': 'white',
    'INFO': 'green',
    'WARNING': 'yellow',
    'ERROR': 'red',
    'CRITICAL': 'purple'
}
fmt_string = '%(log_color)s[%(asctime)s][%(levelname)s]%(message)s'
fmt = colorlog.ColoredFormatter(fmt_string, log_colors=log_colors)
stream_handler.setFormatter(fmt)

logger = logging.getLogger(__name__)
logger.setLevel("DEBUG")
logger.propagate = 0
logger.addHandler(stream_handler)

logger.info(f"Start on {OsType} system...")
torch.set_float32_matmul_precision('medium')

# ================================ build arg parser ==================================
parser = ArgumentParser(description="Main PROSPECT")
parser.add_argument("--config", action=ActionConfigFile)

group = parser.add_argument_group('trainer')
group.add_argument("--tr.max_epochs", type=int, default=500)
group.add_argument("--tr.device", type=int, default=0)
group.add_argument("--tr.patience", type=int, default=400)

group = parser.add_argument_group('data')
group.add_argument("--dt.root", type=str, default="data", help="data root dir")
group.add_argument("--dt.atk_root", type=str, default="atk_data", help="pre-attacked data root dir")
group.add_argument("--dt.data", "-dt", type=str, default="citeseer")
group.add_argument("--dt.attack_method", type=str, default="metattack")
group.add_argument("--dt.ptb_budget", "-pt", type=float, default=0.2, help="The perturbation budget")
group.add_argument("--dt.setting", type=str, default="nettack", help="data split setting")
group.add_argument("--dt.lamb", type=float, default=0.0, help="The ratio of train set nodes in MetaAttack")
group.add_argument("--dt.load_clean", type=int, default=0, help="Load clean graph data")
group.add_argument("--dt.loader", type=str, default="full", help="Full graph data every step")
group.add_argument("--dt.load2device", type=int, default=0, help="transfer data to device at first if True")
group.add_argument("--dt.inductive_test_rate", "-itr", type=float, default=0., help="ind. test node ratio")
group.add_argument("--dt.transform", type=str, default="", help="Valid PyG transform names joined by `-`")

group = parser.add_argument_group('model')
group.add_argument('--md.lr_s', type=float, default=0.01)
group.add_argument('--md.lr_t', type=float, default=0.01)
group.add_argument('--md.wd_s', type=float, default=1e-5)
group.add_argument('--md.wd_t', type=float, default=1e-5)
group.add_argument('--md.dropout_s', type=float, default=0.01)
group.add_argument('--md.dropout_t', type=float, default=0.01)
group.add_argument('--md.T_s', type=float)
group.add_argument('--md.T_t', type=float)
group.add_argument('--md.eta_min_s', type=float)
group.add_argument('--md.eta_min_t', type=float)
group.add_argument('--md.lambda_aim_s', type=float)
group.add_argument('--md.lambda_aim_t', type=float)
group.add_argument('--md.lambda_kd_logit_s', type=float)
group.add_argument('--md.lambda_kd_logit_t', type=float)
group.add_argument('--md.period', type=int)
group.add_argument('--md.mlp_type', type=str, default='MLP')
group.add_argument('--md.hid', type=int, default=64)
group.add_argument('--md.hid_sX', type=int, default=4)
group.add_argument('--md.num_layer_s', type=int, default=2)
group.add_argument('--md.act_s', type=str, default='relu')
group.add_argument('--md.act_t', type=str, default='relu')
group.add_argument("--md.normalize_x", type=bool, default=False, help="The entries of one X row account to 1")

group = parser.add_argument_group('others')
group.add_argument("--log2disk", type=bool, default=True)
group.add_argument("--ckpt2disk", type=bool, default=True)
group.add_argument("--compile", type=bool, default=False, help="torch_geometric.compile")
group.add_argument("--debug", default=False, type=bool, help="If True, use fast debug training example")
group.add_argument("--seed", nargs='*', type=int, help='<Required> Set flag', default=[15, 16, 17, 18, 19])
args = parser.parse_args()

mdcfg = args.md
dtcfg = args.dt
trcfg = args.tr
# ==========================override with debug setting ========================


if args.debug:
    args.seed = [15, 16]
    args.max_epochs = 20

# =================================== configs ==================================
ProjectDir = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
DatasetDir = ProjectDir / "atk_data"

which_data = f"{dtcfg.data}_{dtcfg.attack_method}_{dtcfg.ptb_budget}_{dtcfg.lamb}_{dtcfg.setting}" \
             f"_clean={dtcfg.load_clean}_itr={dtcfg.inductive_test_rate}"

which_model = f"prospect-gnn=sage,hid={mdcfg.hid}"
LogDir = ProjectDir / "prospect_logs"
DataDescLogDir = LogDir / f"{which_data}"
ModelLogDir = DataDescLogDir / which_model
ModelLogDir.mkdir(parents=True, exist_ok=True)

# ----------------- Some Model args ------------------------
df = pd.read_csv(ProjectDir / "llc_graph_dataset_summary.csv")
row = df.loc[df["Dataset"] == dtcfg.data]
hin = row["#node_features"].item()
hout = row["#classes"].item()

num_layer_t = 2
hid = mdcfg.hid

device = trcfg.device
devices = [device]
if dtcfg.loader == "neighbor":
    device = None

MLPClass = MLP if mdcfg.mlp_type == "MLP" else JKMLP


# ==============================================================

def get_datamodule(seed):
    our_transform = ""
    if mdcfg.normalize_x:
        our_transform = our_transform + "-NormalizeFeatures"
    our_transform = our_transform + "-ToSparseTensor"
    datamodule = UnifiedAtkNodeData(
        name=dtcfg.data,
        data_dir=str(DatasetDir),
        attack_method=dtcfg.attack_method,
        ptb_budget=dtcfg.ptb_budget,
        load_clean=dtcfg.load_clean,
        seed=seed,
        loader=dtcfg.loader,
        inductive_test_rate=dtcfg.inductive_test_rate,
        device=device,
        transform=our_transform,
    )
    return datamodule


def multiple_run():
    acc_arr_s = -np.ones(len(args.seed))
    acc_arr_t = -np.ones(len(args.seed))
    for i, seed in enumerate(args.seed):
        # e.g., D:\projects\DistillRobust\tune\optuna_logs\DataDescLogDir\hid=256
        OneRunLogDir = ModelLogDir / f"{seed}"
        seed_everything(seed)
        datamodule = get_datamodule(seed)

        mlp = MLPClass(
            in_channels=hin,
            hidden_channels=hid * mdcfg.hid_sX,
            num_layers=mdcfg.num_layer_s,
            out_channels=hout,
            act=mdcfg.act_s,
            dropout=mdcfg.dropout_s)

        gnn = GraphSAGE(
            in_channels=hin,
            hidden_channels=hid,
            num_layers=num_layer_t,
            out_channels=hout,
            act=mdcfg.act_t,
            dropout=mdcfg.dropout_t,
        )

        co_module = GDMLPLitModule(  # PROSPECT is just one instantiation of this versatile GD-MLP class
            student=mlp,
            teacher=gnn,
            optimizer_s=partial(torch.optim.Adam, lr=mdcfg.lr_s, weight_decay=mdcfg.wd_s),
            optimizer_t=partial(torch.optim.Adam, lr=mdcfg.lr_t, weight_decay=mdcfg.wd_t),
            scheduler_s=CosineAnnealingColdRestart,
            scheduler_s_kwargs={"T_0": mdcfg.period, "cold_start": False, "eta_min": mdcfg.eta_min_s, "verbose": False},
            scheduler_t=CosineAnnealingColdRestart,
            scheduler_t_kwargs={"T_0": mdcfg.period, "cold_start": True, "eta_min": mdcfg.eta_min_t, "verbose": False},
            logit_distiller="DML",
            # feature_distiller="FeatureMimic",
            lambda_aim=[mdcfg.lambda_aim_s, mdcfg.lambda_aim_t],
            lambda_kd_logit=[mdcfg.lambda_kd_logit_s, mdcfg.lambda_kd_logit_t],
            lambda_kd_fea=0,  # no influence if no feature_distiller is provided
            ld_kwargs={"temperature": mdcfg.T_s, "temperature_t": mdcfg.T_t},
            log_verbose=1,
            log_distill=True,
        )
        if args.compile:
            co_module = pyg_compile(co_module)

        callback_list: list[any] = [LearningRateMonitor(logging_interval="epoch")] if args.log2disk else list()
        checkpoint = ModelCheckpoint(
            save_top_k=1,
            monitor="val_acc/students_0_0",
            mode="max",
            save_last=False,
            auto_insert_metric_name=False,
            filename="best_co",
        )
        early = EarlyStopping(monitor="val_acc/students_0_0", mode="max", patience=trcfg.patience)
        callback_list.append(checkpoint)
        callback_list.append(early)
        callback_list = None if len(callback_list) == 0 else callback_list

        exp_logger = TensorBoardLogger(save_dir=OneRunLogDir, name="log2disk") if args.log2disk else False
        distill_trainer = Trainer(
            accelerator="gpu",
            max_epochs=trcfg.max_epochs,
            callbacks=callback_list,
            devices=[device],
            logger=exp_logger,
            default_root_dir=OneRunLogDir,
            num_sanity_val_steps=0,
        )

        # logger.info(" ==> student is learning !")
        distill_trainer.fit(
            model=co_module,
            train_dataloaders=datamodule.distill_dataloader(),
            val_dataloaders=datamodule.val_dataloader(),
        )

        # logger.info(" ==> student gets score !")
        if 0 < dtcfg.inductive_test_rate < 1:
            dataloader2test = [datamodule.inductive_test_dataloader(), datamodule.test_dataloader()]
        elif dtcfg.inductive_test_rate == 0:
            dataloader2test = datamodule.test_dataloader()
        elif dtcfg.inductive_test_rate == 1:
            dataloader2test = datamodule.inductive_test_dataloader()
        else:
            raise ValueError(f"Inductive test rate should be in [0,1]")

        assert distill_trainer.checkpoint_callback.best_model_path is not None

        distill_trainer.test(
            co_module,
            ckpt_path=distill_trainer.checkpoint_callback.best_model_path,
            dataloaders=dataloader2test
        )

        acc_arr_s[i] = distill_trainer.callback_metrics["test_all_acc/students_0_0"].item()
        acc_arr_t[i] = distill_trainer.callback_metrics["test_all_acc/teachers_0_0"].item()

    mean_acc_s = acc_arr_s.mean()
    mean_acc_t = acc_arr_t.mean()

    mean_acc_s100 = mean_acc_s * 100
    mean_acc_t100 = mean_acc_t * 100
    std_acc_s100 = acc_arr_s.std() * 100
    std_acc_t100 = acc_arr_t.std() * 100

    performance_s = (np.around(mean_acc_s100, decimals=2).astype(str) + "±"
                     + np.around(std_acc_s100, decimals=2).astype(str))
    performance_t = (np.around(mean_acc_t100, decimals=2).astype(str) + "±"
                     + np.around(std_acc_t100, decimals=2).astype(str))
    logger.info(f"\n|Test Acc over {len(args.seed)} different splits\n"
                f"|student:    {performance_s}%"
                f"|teacher:    {performance_t}%")
    return mean_acc_s, mean_acc_t


if __name__ == "__main__":
    multiple_run()
