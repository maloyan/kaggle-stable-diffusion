import os
import sys
from pathlib import Path

import numpy as np
import open_clip
import pandas as pd
import torch
from PIL import Image
from sentence_transformers import SentenceTransformer, models
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import OFAModel, OFATokenizer
from transformers.models.ofa.generate import sequence_generator


class CFG:
    sentence_library = "../input/sentence-transformers-222/sentence-transformers"
    image_to_prompts = "../input/stable-diffusion-image-to-prompts/"
    ckpt_dir = "/kaggle/input/stable-diffusion-data/OFA-large-caption/"
    image_dir = "/kaggle/input/stable-diffusion-image-to-prompts/images/"
    batch_size = 16
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    resolution = 480

    device = "cuda"
    seed = 42
    embedding_length = 384
    sentence_model_path = "/kaggle/input/sentence-transformers-222/all-MiniLM-L6-v2"
    model_name = "coca_ViT-L-14"
    model_checkpoint_path = "/kaggle/input/open-clip-models/mscoco_finetuned_CoCa-ViT-L-14-laion2B-s13B-b90k.bin"


comp_path = Path(CFG.image_to_prompts)


images = os.listdir(comp_path / "images")
imgIds = [i.split(".")[0] for i in images]

eIds = list(range(CFG.embedding_length))

imgId_eId = [
    "_".join(map(str, i))
    for i in zip(
        np.repeat(imgIds, CFG.embedding_length),
        np.tile(range(CFG.embedding_length), len(imgIds)),
    )
]

model = open_clip.create_model(CFG.model_name)
model.to(CFG.device)
open_clip.load_checkpoint(model, CFG.model_checkpoint_path)

transform = open_clip.image_transform(
    model.visual.image_size,
    is_train=False,
    mean=getattr(model.visual, "image_mean", None),
    std=getattr(model.visual, "image_std", None),
)


prompts = []

for image_name in images:
    img = Image.open(CFG.image_dir + image_name).convert("RGB")
    img = transform(img).unsqueeze(0)

    with torch.no_grad(), torch.cuda.amp.autocast():
        generated = model.generate(img.to(device))

    prompts.append(
        open_clip.decode(generated[0])
        .split("<end_of_text>")[0]
        .replace("<start_of_text>", "")
        .rstrip(" .,")
    )


st_model = SentenceTransformer(CFG.sentence_model_path)


class ImageGen(Dataset):
    def __init__(self, root, batch_size=32):
        self.root = root
        self.im_paths = os.listdir(self.root)
        self.batch_size = batch_size
        self.sz = len(self.im_paths)
        self.genlen = self.sz // self.batch_size + int(self.sz % self.batch_size > 0)

    def __getitem__(self, index):
        if index >= self.genlen:
            raise IndexError("Out of bounds")

        l, r = index * self.batch_size, min(self.sz, (index + 1) * self.batch_size)

        f_paths = [os.path.join(self.root, self.im_paths[i]) for i in range(l, r)]
        f_ids = [self.im_paths[i][:-4] for i in range(l, r)]

        ims = [Image.open(f_path) for f_path in f_paths]
        ims = [patch_resize_transform(im).cuda().unsqueeze(0) for im in ims]
        ims = torch.cat(ims)

        return ims, f_ids

    def __len__(self):
        return self.genlen


patch_resize_transform = transforms.Compose(
    [
        lambda image: image.convert("RGB"),
        transforms.Resize(
            (CFG.resolution, CFG.resolution), interpolation=Image.BICUBIC
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=CFG.mean, std=CFG.std),
    ]
)

tokenizer = OFATokenizer.from_pretrained(CFG.ckpt_dir)
model = OFAModel.from_pretrained(CFG.ckpt_dir, use_cache=False).to(CFG.device)

txt = " what does the image describe?"
inputs = tokenizer([txt], return_tensors="pt").input_ids

sub_ids = []
subs = []
imgen = ImageGen(CFG.image_dir, CFG.batch_size)
for b in imgen:
    for j in range(len(b[1])):
        sub_ids.extend([f"{b[1][j]}_{i}" for i in range(384)])

    img_batch = b[0]
    out = model.generate(
        inputs.repeat(len(img_batch), 1).cuda(),
        patch_images=img_batch,
        num_beams=5,
        no_repeat_ngram_size=3,
    )
    out_captions = tokenizer.batch_decode(out, skip_special_tokens=True)
    out_captions = [cap + ", fine details, masterpiece" for cap in out_captions]
    subs.append(out_captions)

prompt_embeddings = st_model.encode(prompts).flatten()
submission_laion_coca_vit = pd.DataFrame(
    index=imgId_eId, data=prompt_embeddings, columns=["val"]
).rename_axis("imgId_eId")


sub_embeds = st_model.encode(subs).flatten()
submission_ofa_transformer = pd.DataFrame({"imgId_eId": sub_ids, "val": sub_embeds})
