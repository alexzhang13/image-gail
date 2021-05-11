import glob
import os
from PIL import Image
from torchvision import models, transforms
from torchvision import models, transforms
from tqdm import tqdm
import lmdb

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
    transform = get_default_transform()
    env = lmdb.open(out_path, map_size=1099511627776)

    # read all *.png and *.jpg files
    all_paths = glob.glob(os.path.join(img_root, "*.png")) + glob.glob(os.path.join(img_root, "*.jpg"))
    for p in tqdm(all_paths):
        try:
            loaded_image = transform(Image.open(p).convert("RGB"))
        except:
            print("image corrupt: %s"%p)
            continue
        key = os.path.basename(p)
        with env.begin(write=True) as txn:
            txn.put(key.encode(), loaded_image)

for split in splits:
    process_images(os.path.join(IMAGE_ROOT, split), os.path.join(OUT_DIR, "image-feats-%s"%split))