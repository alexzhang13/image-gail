# GAIL for Images on VIST Dataset

## Setup
Install Anaconda3, PyTorch, and in the main folder, run
```
conda install --file requirements.txt
```
To download the VIST dataset, visit http://visionandlanguage.net/VIST/dataset.html
and make sure to change the necessary params when initializing `VISTDatasetImages(params)` object.


## Training
See the file for more details on arguments. To train the model, run
```
python train.py --freeze_epochs 5 --batch_size 16 --epochs 100 --gen_per_discrim 1 --variance 0.01 --name default_fixed
```
Network checkpoints are only saved for the best validation scores.

## Evaluation
There is a `test.py` file for evaluating the model on the validation/test splits of the VIST dataset. After training, run
```
python test.py --batch_size 16 --name test --path ./saved_models/name/checkpoint_name_epoch_x.t7
```
There is also a file called `scores_verify.py` for ensuring the results dumped into the JSON files correctly match with the
list scores.
