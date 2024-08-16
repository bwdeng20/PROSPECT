from src import utils
from src.data.gnn_node_datamodule import ModifiedLightningNodeData

log = utils.get_pylogger(__name__)


def test_stage4node_level(cfg, trainer, model, datamodule, task="train"):
    if task == "train":
        if cfg.get("test"):
            do_test = True
        else:
            do_test = False
        ckpt_path = trainer.checkpoint_callback.best_model_path
        if ckpt_path == "":
            log.warning(
                "Best ckpt not found in `trainer.checkpoint_callback.best_model_path`"
                "Using current weights for testing..."
            )
            ckpt_path = None

    elif task == "eval":  # ckpt path must be specified in an eval task
        ckpt_path = cfg.ckpt_path
        do_test = True
    else:
        raise RuntimeError(f"Task `{task}` is not valid")

    if do_test:
        log.info("Starting testing!")
        if isinstance(datamodule, ModifiedLightningNodeData):  # If practical inductive setting
            itr = datamodule.inductive_test_rate
            if 0 < itr < 1:
                log.info(f"Test on {(1 - itr) * 100}% transductive and {itr * 100}% inductive test nodes")
                trainer.test(
                    model=model,
                    ckpt_path=ckpt_path,
                    dataloaders=[
                        datamodule.test_dataloader(),
                        datamodule.inductive_test_dataloader(),
                    ],
                )

            elif itr == 1:  # totally inductive
                log.info(f"Test on {itr * 100}% inductive test nodes")
                trainer.test(
                    model=model,
                    ckpt_path=ckpt_path,
                    dataloaders=datamodule.inductive_test_dataloader(),
                )
            elif itr == 0:  # totally transductive
                log.info(f"Test on {itr * 100}% transductive test nodes")
                trainer.test(
                    model=model,
                    ckpt_path=ckpt_path,
                    dataloaders=datamodule.test_dataloader(),
                )
            else:
                raise ValueError(f"Inductive test rate {itr} should lie in [0,1]")

        else:  # If simple non-mix case of either inductive or transductive setting
            trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)

    else:
        log.info("Skipping the test stage according to the config file")
    log.info(f"Best ckpt path: {ckpt_path}")
