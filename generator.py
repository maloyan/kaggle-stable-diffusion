import torch
from diffusers import StableDiffusionPipeline
from datasets import load_dataset
import pandas as pd

dataset = load_dataset("bartman081523/stable-diffusion-discord-prompts")
pipe = StableDiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2",
    torch_dtype=torch.float16,
    safety_checker=None
)
pipe = pipe.to("cuda")

ind = 0
ids = []
prompts = []

for prompt in dataset['train']['text'][:5000]:
    image = pipe(prompt).images[0]
    image.resize((512, 512)).save(f"/root/kaggle-sd/input_generated/images/{ind}.png")
    ids.append(ind)
    prompts.append(prompt)
    ind += 1


df = pd.DataFrame({'imgId': ids,'prompt': prompts})
df.to_csv("/root/kaggle-sd/input_generated/prompts.csv", index=None)
