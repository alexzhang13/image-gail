import glob
import os
import io
from PIL import Image
from torchvision import models, transforms
from tqdm import tqdm
import lmdb
import torch

IMAGE_ROOT = "/n/fs/nlp-murahari/datasets/VIST/images"
splits = ["train", "val", "test"]
OUT_DIR = "/n/fs/nlp-alzhang/image-feats"

def get_default_transform():
    return transforms.Compose(
        [
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
        ]
    )


def process_images(img_root, out_path):
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    img_num = 0
    buff = io.BytesIO()
    transform = get_default_transform()
    env = lmdb.open(out_path, map_size=1099511627776)

    # read all *.png and *.jpg files
    all_paths = glob.glob(os.path.join(img_root, "*.png")) + glob.glob(os.path.join(img_root, "*.jpg"))
    for p in tqdm(all_paths):
        try:
            loaded_image = transform(Image.open(p).convert("RGB"))
            loaded_image = np.array(loaded_image)
        except:
            print("image corrupt: %s"%p)
            continue
        key = os.path.basename(p)
        print("Img Num: ", img_num, end="")
        print("\t Key: ", key, end="\n")
        with env.begin(write=True) as txn:
            txn.put(key.encode(), loaded_image.tobytes())

for split in splits:
    process_images(os.path.join(IMAGE_ROOT, split), os.path.join(OUT_DIR, "image-feats-%s"%split))