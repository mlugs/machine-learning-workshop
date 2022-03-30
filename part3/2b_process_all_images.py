# This script loads a model checkpoint and generates masks for all images using that model
#
# lots of code duplication because there is no way to import from a jupyter notebook :(

from pathlib import Path

import cv2
import numpy as np
import pytorch_lightning as lightning
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchmetrics import Accuracy
from tqdm import tqdm

from unet_parts import *


class DiceMaskDataset(Dataset):
    def __init__(
        self, images_dir: str, masks_dir: str, scale: float = 1.0, mask_suffix: str = ""
    ):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.mask_suffix = mask_suffix
        assert 0 < scale <= 1, "Scale must be between 0 and 1"
        self.scale = scale

        self.ids = [i.stem for i in self.images_dir.glob("*png")]
        if not self.ids:
            raise RuntimeError(
                f"No input file found in {images_dir}, make sure you put your images there"
            )

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, pil_img, scale, is_mask):

        w, h = pil_img.size

        newW, newH = int(scale * w), int(scale * h)
        assert (
            newW > 0 and newH > 0
        ), "Scale is too small, resized images would have no pixel"
        pil_img = pil_img.resize(
            (newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC
        )
        img_ndarray = np.asarray(pil_img)

        if img_ndarray.ndim == 2 and not is_mask:
            img_ndarray = img_ndarray[np.newaxis, ...]
        elif not is_mask:
            img_ndarray = img_ndarray.transpose((2, 0, 1))

        if not is_mask:
            img_ndarray = img_ndarray / 255

        return img_ndarray

    @classmethod
    def load(cls, filename):
        return Image.open(filename)

    def __getitem__(self, idx):
        name = self.ids[idx]
        mask_file = list(self.masks_dir.glob(name + self.mask_suffix + ".*"))
        img_file = list(self.images_dir.glob(name + ".*"))

        assert (
            len(mask_file) == 1
        ), f"Either no mask or multiple masks found for the ID {name}: {mask_file}"
        assert (
            len(img_file) == 1
        ), f"Either no image or multiple images found for the ID {name}: {img_file}"
        mask = self.load(mask_file[0])
        img = self.load(img_file[0])

        assert (
            img.size == mask.size
        ), "Image and mask {name} should be the same size, but are {img.size} and {mask.size}"

        img = self.preprocess(img, self.scale, is_mask=False)
        mask = self.preprocess(mask, self.scale, is_mask=True)

        return (
            torch.as_tensor(img.copy()).float().contiguous(),
            torch.as_tensor(mask.copy()).long().contiguous(),
        )


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


class DiceMaskModel(lightning.LightningModule):
    def __init__(self, learning_rate=1e-5):
        super().__init__()
        self.save_hyperparameters()

        self.learning_rate = learning_rate
        self.batch_size = 1

        self.model = UNet(n_channels=3, n_classes=2, bilinear=True)

        self.loss = torch.nn.CrossEntropyLoss()

        acc = Accuracy()
        self.train_acc = acc.clone()
        self.valid_acc = acc.clone()

    def forward(self, x):
        return self.model(x)

    def step(self, batch, batch_idx, name):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss(y_hat, y)
        self.log(f"{name}/loss", loss)
        return {"loss": loss}

    def training_step(self, batch, batch_nb):
        return self.step(batch, batch_nb, name="train")

    def validation_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, name="val")

    def test_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, name="test")

    def configure_optimizers(self):
        return torch.optim.RMSprop(self.parameters(), lr=self.learning_rate)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(train_ds, batch_size=self.batch_size)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(val_ds, batch_size=self.batch_size)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(test_ds, batch_size=self.batch_size)


dataset = DiceMaskDataset(
    "/data/dice-mfa-blue-good/train", "RESULT_dice-mfa-blue-good/train/masks"
)
model = DiceMaskModel.load_from_checkpoint(
    "workshop-p3--dice-unet/dw5qurvr/checkpoints/epoch=9-step=7950.ckpt"
)
model.eval()
print("model loaded")

folder_result = "RESULT_dice-mfa-blue-unet-masks"
files = list(Path("/data/dice-mfa-blue-good/train").glob("*png"))

pbar = tqdm(files, total=len(files))
for fn in pbar:
    target = Path(folder_result) / fn.name
    pbar.set_description(target.name)
    if target.exists():
        continue
    
    orig_img = Image.open(str(fn))
    img = dataset.preprocess(orig_img, 1, is_mask=False)
    img = torch.as_tensor(img).float().contiguous()[None, :]
    probs = model(img)
    probs = probs.squeeze(0)
    mask_bools = probs.squeeze().detach().numpy()[1] > 0.5
    mask_img = Image.fromarray((mask_bools * 255).astype(np.uint8))
    mask_img.save(str(target), output="png")
