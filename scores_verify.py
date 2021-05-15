import json
import argparse

parser = argparse.ArgumentParser(description="Arguments for training")
parser.add_argument('--path', default="./logger/val_schedule_5_1.json")

# Opening JSON file
f = open(args.path,)

args = parser.parse_args()
   
# returns JSON object as 
# a dictionary
data = json.load(f)
   
# Iterating through the json
# list
print(len(data))
for i in range(len(data)):
    print(data[i]['scores'])
   
# Closing file
f.close()