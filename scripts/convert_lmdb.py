#!/usr/bin/env python3

import argparse
import os
from os.path import basename
import lmdb
import glob
import logging
from types import SimpleNamespace
from joblib import Parallel, delayed
from torchvision import transforms
from PIL import Image

from datetime import datetime

def iter_filename(root_dir):
    all_paths = glob.glob(os.path.join(root_dir, "*.png")) + glob.glob(os.path.join(root_dir, "*.jpg"))
    for p in all_paths:
        yield p

def get_default_transform():
    return transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ]
    )

transform = get_default_transform()

def convert_single_file(filepath):
    filename = basename(filepath).split('.')[0]

    try:
        loaded_image = transform(Image.open(filepath).convert("RGB"))
        byte_encimg = loaded_image.numpy().tobytes()
        byte_filename = filename.encode()
        success = True
        explain = ''
    except Exception as e:
        byte_encimg = None
        byte_filename = None
        success = False
        explain = str(e.args)

    return SimpleNamespace(
        byte_encimg=byte_encimg,
        byte_filename=byte_filename,
        success=success,
        filename=filename,
        explain=explain)


def main():
    batch_size = 200

    parser = argparse.ArgumentParser(description='Convert lmdb')
    parser.add_argument(
        '--root_dir', '-r', default='', help='Path to input root dir.')
    parser.add_argument(
        '--lmdb', '-l', default='', help='Path to output lmdb file.')
    parser.add_argument(
        '--logging_file',
        '-g',
        default='convert_lmdb.log',
        help='Path to logging file.')
    args = parser.parse_args()
    print(args)

    logger = logging.getLogger('convert_lmdb')
    fh = logging.FileHandler(args.logging_file, 'w')
    logger.setLevel(10)
    logger.addHandler(fh)

    lmdb_env = lmdb.open(args.lmdb, map_size=int(1e12))

    count = {'total': 0, 'success': 0}

    def deal_batch(batch):
        res = Parallel(n_jobs=20)(delayed(convert_single_file)(filename)
                                    for filename in batch)
        for converted in res:
            if converted.success:
                lmdb_txn = lmdb_env.begin(write=True)
                lmdb_txn.put(converted.byte_filename, converted.byte_encimg)
                lmdb_txn.commit()
                count['success'] += 1
            else:
                logger.error('"%s" failed with reason %s' %
                                (converted.filename, converted.explain))

            count['total'] += 1
        now = datetime.now() # current date and time        
        logger.info('Time: %s Progress: %d files tried. %d successed.' %
                    (now.strftime("%m/%d/%Y, %H:%M:%S"), count['total'], count['success']))

    logger.info('start')
    batch = []
    for index, filename in zip(
            range(int(10**10)), iter_filename(args.root_dir)):
        batch.append(filename)
        if len(batch) == batch_size:
            deal_batch(batch)
            batch = []
    if len(batch) > 0:
        deal_batch(batch)
        batch = []
    lmdb_env.close()
    logger.info('finish')
    return


if __name__ == '__main__':
    main()