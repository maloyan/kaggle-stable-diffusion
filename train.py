import os
import unicodedata

import numpy as np
import pandas as pd
import timm
import torch
import wandb
from PIL import Image
from scipy import spatial
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from timm.utils import AverageMeter
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm


class CFG:
    custom_name_prefix = "" #"small_dataset_attention_head"
    model_name = "convnext_xlarge_in22ft1k"  # "eva_giant_patch14_336.clip_ft_in1k" #"convnext_xlarge.fb_in22k_ft_in1k_384" # "vit_huge_patch14_clip_336.laion2b_ft_in12k_in1k"
    input_size = (256, 256)  # (336, 336)
    batch_size = 64
    num_epochs = 3
    lr = 1e-4
    seed = 42


wandb.init(
    project="kaggle_sd_vit",
    name=f"{CFG.custom_name_prefix}{CFG.model_name}",
)


class DiffusionDataset(Dataset):
    def __init__(self, df, transform):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(f"/root/NarekMaloyan/root/kaggle-sd/input/images/{row['image_name']}")
        image = self.transform(image)
        prompt = row["prompt"]
        return image, prompt


# class CustomAttentionHead(nn.Module):
#     def __init__(self, in_features, num_classes, attention_size):
#         super(CustomAttentionHead, self).__init__()
#         self.attention = nn.Sequential(
#             nn.Linear(in_features, attention_size),
#             nn.ReLU(inplace=True),
#             nn.Linear(attention_size, num_classes)
#         )

#     def forward(self, x):
#         return self.attention(x)


class DiffusionCollator:
    def __init__(self):
        self.st_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

    def __call__(self, batch):
        images, prompts = zip(*batch)
        images = torch.stack(images)
        prompt_embeddings = self.st_model.encode(
            prompts, show_progress_bar=False, convert_to_tensor=True
        )
        return images, prompt_embeddings


def get_dataloaders(trn_df, val_df, input_size, batch_size):
    transform = transforms.Compose(
        [
            transforms.Resize(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    trn_dataset = DiffusionDataset(trn_df, transform)
    val_dataset = DiffusionDataset(val_df, transform)
    collator = DiffusionCollator()

    dataloaders = {}
    dataloaders["train"] = DataLoader(
        dataset=trn_dataset,
        shuffle=True,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=8,
        drop_last=True,
        collate_fn=collator,
    )
    dataloaders["val"] = DataLoader(
        dataset=val_dataset,
        shuffle=False,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=8,
        drop_last=False,
        collate_fn=collator,
    )
    return dataloaders


def cosine_similarity(y_trues, y_preds):
    return np.mean(
        [
            1 - spatial.distance.cosine(y_true, y_pred)
            for y_true, y_pred in zip(y_trues, y_preds)
        ]
    )


def is_english_only(string):
    for s in string:
        cat = unicodedata.category(s)
        if not cat in ["Ll", "Lu", "Nd", "Po", "Pd", "Zs"]:
            return False
    return True


def train(trn_df, val_df, model_name, input_size, batch_size, num_epochs, lr):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataloaders = get_dataloaders(trn_df, val_df, input_size, batch_size)

    model = timm.create_model(model_name, pretrained=True, num_classes=384)
    # in_features = model.head.fc.in_features
    # attention_size = 512
    # num_classes = 384

    # custom_attention_head = CustomAttentionHead(
    #     in_features, num_classes, attention_size
    # )

    # # Replace the classifier in the timm model with the custom attention head
    # model.head.fc = custom_attention_head

    model.set_grad_checkpointing()
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    ttl_iters = num_epochs * len(dataloaders["train"])
    scheduler = CosineAnnealingLR(optimizer, T_max=ttl_iters, eta_min=1e-6)
    criterion = nn.CosineEmbeddingLoss()
    #criterion2 = nn.MSELoss()
    best_score = -1.0

    for epoch in range(num_epochs):
        train_meters = {
            "loss": AverageMeter(),
            "cos": AverageMeter(),
        }
        model.train()
        pbar = tqdm(dataloaders["train"], total=len(dataloaders["train"]))
        for X, y in pbar:
            X, y = X.to(device), y.to(device)

            optimizer.zero_grad()
            X_out = model(X)
            target = torch.ones(X.size(0)).to(device)
            loss = criterion(X_out, y, target) #+ 0.1 * criterion2(X_out, y)  # + torch.linalg.norm(X_out - y, 1)
            loss.backward()

            optimizer.step()
            scheduler.step()

            trn_loss = loss.item()
            trn_cos = cosine_similarity(
                X_out.detach().cpu().numpy(), y.detach().cpu().numpy()
            )

            train_meters["loss"].update(trn_loss, n=X.size(0))
            train_meters["cos"].update(trn_cos, n=X.size(0))
            pbar.set_description(
                "Epoch {:d} / trn/loss={:.4f}, trn/cos={:.4f}".format(
                    epoch + 1, train_meters["loss"].avg, train_meters["cos"].avg
                )
            )

        print(
            "Epoch {:d} / trn/loss={:.4f}, trn/cos={:.4f}".format(
                epoch + 1, train_meters["loss"].avg, train_meters["cos"].avg
            )
        )

        val_meters = {
            "loss": AverageMeter(),
            "cos": AverageMeter(),
        }

        model.eval()
        for X, y in tqdm(dataloaders["val"], total=len(dataloaders["val"])):
            X, y = X.to(device), y.to(device)

            with torch.no_grad():
                X_out = model(X)
                target = torch.ones(X.size(0)).to(device)
                loss = criterion(X_out, y, target)  # + torch.linalg.norm(X_out - y, 1)

                val_loss = loss.item()
                val_cos = cosine_similarity(
                    X_out.detach().cpu().numpy(), y.detach().cpu().numpy()
                )

            val_meters["loss"].update(val_loss, n=X.size(0))
            val_meters["cos"].update(val_cos, n=X.size(0))

        print(
            "Epoch {:d} / val/loss={:.4f}, val/cos={:.4f}".format(
                epoch + 1, val_meters["loss"].avg, val_meters["cos"].avg
            )
        )
        wandb.log(
            {
                "trn/loss": train_meters["loss"].avg,
                "trn/cos": train_meters["cos"].avg,
                "val/loss": val_meters["loss"].avg,
                "val/cos": val_meters["cos"].avg,
            }
        )
        if val_meters["cos"].avg > best_score:
            best_score = val_meters["cos"].avg
            torch.save(
                model.state_dict(), f"/root/NarekMaloyan/root/kaggle-sd/checkpoint/{CFG.custom_name_prefix}{model_name}.pth"
            )


df = pd.read_parquet("/root/NarekMaloyan/root/kaggle-sd/input/metadata.parquet")
#df = pd.read_csv("/root/NarekMaloyan/root/kaggle-sd/input/train_meta.csv")
df = df[(df["width"] == 512) & (df["height"] == 512)]
df["prompt"] = df["prompt"].str.strip()
df = df[~df.prompt.isna()]
df = df[df["prompt"].map(lambda x: len(x.split())) >= 5]
df = df[~df["prompt"].str.contains("^(?:\s*|NULL|null|NaN)$", na=True)]
df = df[df["prompt"].apply(is_english_only)]
df["head"] = df["prompt"].str[:15]
df["tail"] = df["prompt"].str[-15:]
df.drop_duplicates(subset="head", inplace=True)
df.drop_duplicates(subset="tail", inplace=True)

img_list = os.listdir("/root/NarekMaloyan/root/kaggle-sd/input/images")
non_existing_imgs = list(set(df.image_name.values).difference(img_list))
df = df[~df.image_name.isin(non_existing_imgs)]
df.reset_index(drop=True, inplace=True)
print(df.shape)
assert not set(df.image_name.values).difference(img_list)

trn_df, val_df = train_test_split(df, test_size=0.1)

train(
    trn_df,
    val_df,
    CFG.model_name,
    CFG.input_size,
    CFG.batch_size,
    CFG.num_epochs,
    CFG.lr,
)
