import json
import argparse
import numpy as np

parser = argparse.ArgumentParser(description="Arguments for training")
parser.add_argument('--path', default="./logger/val_schedule_5_1.json")
args = parser.parse_args()

# Opening JSON file
f = open(args.path,)
   
# returns JSON object as 
# a dictionary
data = json.load(f)
   
# Iterating through the json
# list
print(len(data))
accuracy = 0.0
r3_accuracy = 0.0
total = 0

for i in range(len(data)):
    feat_diff = np.array(data[i]['scores'])
    min_indices = np.argmin(feat_diff, axis=1).flatten()
    r3_indices = np.argsort(feat_diff, axis=1)[:,0].flatten()
    shape = (min_indices.shape[0])
    p1 = r3_indices[0].flatten()
    p2 = r3_indices[0].flatten()
    p3 = r3_indices[0].flatten()

    zeros = min_indices == 0
    r3 = p1 + p2 + p3

    correct = zeros.nonzero()[0].shape[0]
    r3_correct = r3.nonzero()[0].shape[0]
    
    accuracy += correct
    r3_accuracy += r3_correct
    total += shape

print("[Val] [Epoch #: X]\t [Accuracy: %f]\t [R3 Accuracy: %f]\n" % (accuracy/(total),r3_accuracy/(total)))
   
# Closing file
f.close()
