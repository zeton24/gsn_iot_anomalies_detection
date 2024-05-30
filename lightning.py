import os
import gdown
import polars
import torch
import pytorch_lightning as pl
from torchmetrics.functional import accuracy, precision, recall, f1_score
from torchmetrics.classification import BinaryConfusionMatrix
from torch.utils.data import Dataset, DataLoader, random_split
F = torch.nn.functional


class IoTDataset(Dataset):
    def __init__(self, inputs: torch.tensor, labels: torch.tensor, n_classes: int):
        self.inputs = inputs
        self.labels = labels
        self.n_classes = n_classes

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        temp = torch.zeros(self.n_classes)
        temp[self.labels[idx].item()] = 1
        return self.inputs[idx], temp


class IoTDataModule(pl.LightningDataModule):
    def __init__(self, file_id: str, file_name: str, batch_size: int = 64, binary_classification: bool = False):
        super().__init__()
        self.file_id = file_id
        self.file_name = file_name
        self.batch_size = batch_size
        self.binary_classification = binary_classification
        self.n_classes = None
        self.mapping = None
        self.train_dataset = None
        self.test_dataset = None
        self.val_dataset = None

    def prepare_data(self):
        if not os.path.exists(self.file_name):
            gdown.download(id=self.file_id, output=self.file_name)

    def setup(self, stage=None):
        data = torch.reshape(polars.read_csv(self.file_name, columns=range(64)).cast(polars.Float32).to_torch(), (-1, 1, 64))

        if self.binary_classification:
            self.n_classes = 2
            self.mapping = {"Normal": 0, "Anomaly": 1}
            labels = torch.reshape(polars.read_csv(self.file_name, columns=[64])['Label'].replace(self.mapping, return_dtype=polars.UInt8).to_torch(), (-1, 1))
        else:
            labels = polars.read_csv(self.file_name, columns=[65])['Cat']
            classes = labels.unique()
            self.n_classes = len(classes)
            self.mapping = {cls: idx for (idx, cls) in enumerate(classes)}
            labels = labels.replace(self.mapping, return_dtype=polars.UInt8).to_torch()

        dataset = IoTDataset(data, labels, self.n_classes)
        del data, labels

        # Train, test i val.
        length = len(dataset)
        train_and_val_size = int(length * .75)
        dataset, self.test_dataset = random_split(dataset, [train_and_val_size, length - train_and_val_size])

        ll = len(dataset)
        train_size = int(ll * .9)
        self.train_dataset, self.val_dataset = random_split(dataset, [train_size, ll - train_size])
        del dataset

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)


class AnomalyClassifier(pl.LightningModule):

    def __init__(self, lr, model: torch.nn.Module):
        super().__init__()
        self.save_hyperparameters(ignore=['model'])
        self.lr = lr
        self.model = model
        self.num_classes = model.num_classes
        self.current_epoch_training_loss = torch.tensor(0.0)
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.bcm = BinaryConfusionMatrix()

    def forward(self, x):
        return self.model(x)

    def compute_loss(self, x, y):
        return F.cross_entropy(x, y)

    def common_step(self, batch, batch_idx):
        x, y = batch
        outputs = self(x)
        loss = self.compute_loss(outputs, y)
        return loss, outputs, y

    def common_test_valid_step(self, batch, batch_idx):
        loss, outputs, y = self.common_step(batch, batch_idx)
        preds = torch.argmax(outputs, dim=1)
        z = torch.argmax(y, dim=1)
        if self.num_classes == 2:
            acc = accuracy(preds, z, task="binary")
            prec = precision(preds, z, task="binary")
            rec = recall(preds, z, task="binary")
            f1 = f1_score(preds, z, task="binary")
            self.bcm.update(preds, z)
        else:
            acc = accuracy(preds, z, num_classes=self.num_classes, task="multiclass")
            prec = precision(preds, z, num_classes=self.num_classes, task="multiclass", average="macro")
            rec = recall(preds, z, num_classes=self.num_classes, task="multiclass", average="macro")
            f1 = f1_score(preds, z, num_classes=self.num_classes, task="multiclass", average="macro")

        return loss, acc, prec, rec, f1

    def training_step(self, batch, batch_idx):
        loss, outputs, y = self.common_step(batch, batch_idx)
        self.training_step_outputs.append(loss)
        preds = torch.argmax(outputs, dim=1)
        z = torch.argmax(y, dim=1)
        if self.num_classes == 2:
            acc = accuracy(preds, z, task="binary")
        else:
            acc = accuracy(preds, z, num_classes=self.num_classes, task="multiclass")
        self.log_dict({"train_loss": loss, "train_accuracy": acc}, on_step=False, on_epoch=True, prog_bar=True)
        return {'loss': loss}

    def on_train_epoch_end(self):
        outs = torch.stack(self.training_step_outputs)
        self.current_epoch_training_loss = outs.mean()
        self.training_step_outputs.clear()

    def validation_step(self, batch, batch_idx):
        loss, acc, _, _, _ = self.common_test_valid_step(batch, batch_idx)
        self.validation_step_outputs.append(loss)

        self.log_dict({'val_loss': loss, 'val_acc': acc}, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        #return {'val_loss': loss, 'val_acc': acc}

    def on_validation_epoch_end(self):
        outs = torch.stack(self.validation_step_outputs)
        avg_loss = outs.mean()

        self.log('train', self.current_epoch_training_loss.item(), on_epoch=True, logger=True)
        self.log('val', avg_loss.item(), on_epoch=True, logger=True)

        self.validation_step_outputs.clear()

    def test_step(self, batch, batch_idx):
        loss, acc, prec, rec, f1 = self.common_test_valid_step(batch, batch_idx)

        self.log_dict({'test_loss': loss, 'test_acc': acc}, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log_dict({'precision': prec, 'recall': rec, 'f1_score':  f1}, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return {'test_loss': loss, 'test_acc': acc, 'precision': prec, 'recall': rec, 'f1_score':  f1}

    def on_test_epoch_end(self):
        bcm_result = self.bcm.compute()
        self.bcm.plot()
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
        return [optimizer], [lr_scheduler]
