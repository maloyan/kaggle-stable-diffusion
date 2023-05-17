import numpy as np
import pandas as pd
import torch
import wandb
from datasets import load_dataset
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import AutoProcessor, BlipForConditionalGeneration


class config:
    project = "kaggle-sd"
    image_path = "/root/kaggle-sd/input_generated/images/"
    data_path = "/root/kaggle-sd/input_generated/"
    data_path_original = "/root/kaggle-sd/input/"
    checkpoint_path = "/root/kaggle-sd/checkpoint"
    model_name = "Salesforce/blip-image-captioning-large"
    epochs = 50
    batch_size = 2


wandb.init(
    project=config.project,
    name=f"{config.model_name}",
)


class ImageCaptioningDataset(Dataset):
    def __init__(self, dataset, processor):
        self.dataset = dataset
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset.iloc[idx]
        image = Image.open(f"{config.image_path}/{item['imgId']}.png")
        encoding = self.processor(
            images=image,
            text=item["prompt"],
            padding="max_length",
            return_tensors="pt",
        )
        # remove batch dimension
        encoding = {k: v.squeeze() for k, v in encoding.items()}
        return encoding


dataset = pd.read_csv(f"{config.data_path}/prompts.csv")
# dataset = load_dataset("poloclub/diffusiondb", "large_random_100k")
processor = AutoProcessor.from_pretrained(config.model_name)
processor.save_pretrained(f"{config.checkpoint_path}/processor")

model = BlipForConditionalGeneration.from_pretrained(config.model_name)

train_dataset = ImageCaptioningDataset(dataset, processor)
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=config.batch_size)


optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

model.train()

best_loss = 1000
steps = 0
for epoch in range(config.epochs):
    print("Epoch:", epoch)
    for ind, batch in enumerate(train_dataloader):
        input_ids = batch.pop("input_ids").to(device)
        pixel_values = batch.pop("pixel_values").to(device)

        outputs = model(
            input_ids=input_ids, pixel_values=pixel_values, labels=input_ids
        )

        loss = outputs.loss
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()
        if ind % 10_000 == 0:
            wandb.log({"loss": loss})
            if loss < best_loss:
                best_loss = loss
                steps = 0
                model.save_pretrained(f"{config.checkpoint_path}/model")
                print("Model saved!")
                print(f"Best loss is {best_loss}")
            else:
                steps += 1

            if steps == 5:
                break

# prepare image for the model
df_test = pd.read_csv(f"{config.data_path_original}/prompts.csv")
for _, example in df_test.iterrows():
    image = Image.open(f"{config.image_path}/{example['imgId']}.png")
    inputs = processor(images=image, return_tensors="pt").to(device)
    pixel_values = inputs.pixel_values

    generated_ids = model.generate(pixel_values=pixel_values, max_length=50)
    generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[
        0
    ]
    print(f"{generated_caption} \t {example['prompt']}")


torch.save(model.state_dict(), f"{config.checkpoint_path}/model.pt")
