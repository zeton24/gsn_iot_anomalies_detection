from hydra.utils import instantiate
from hydra import main
from omegaconf import DictConfig
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
import wandb
from pytorch_lightning import Trainer


@main(config_path="conf", config_name="config", version_base="1.1")
def test(cfg: DictConfig):
    # Logger i callbacki.
    wandb_logger = WandbLogger(project='dummy', job_type='train')
    ckpt_path = 'checkpoints/'
    ckpt = 'model-{epoch:02d}-{val_loss:.2f}'

    ckpt_callback = ModelCheckpoint(monitor='val_loss', dirpath=ckpt_path, filename=ckpt, save_top_k=3, mode='min')
    early_stop_callback = EarlyStopping(monitor='val_loss', patience=3, verbose=False, mode='min')
    callbacks = [early_stop_callback, ckpt_callback]

    # Inicjalizacja.
    dm = instantiate(cfg.dataset)
    dm.prepare_data()
    dm.setup()

    classifier = instantiate(cfg.classifier)
    trainer = Trainer(accelerator="auto", max_epochs=10, precision=32, logger=wandb_logger, callbacks=callbacks)

    # Trening i test.
    trainer.fit(classifier, dm)
    trainer.test(classifier, dm)

    # Zapisanie modelu.
    wandb.finish()
    run = wandb.init(project='dummy', job_type='producer')

    artifact = wandb.Artifact('model', type='model')
    artifact.add_dir(ckpt_path)

    run.log_artifact(artifact)
    run.join()


test()
