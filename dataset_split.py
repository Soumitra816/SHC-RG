import json
import random

# Read the dataset from the file dataset.json
with open('SHINES.json', 'r') as file:
    dataset = json.load(file)

# Shuffle the dataset
random.shuffle(dataset)

# Split the dataset into 80:20 ratio
split_ratio = 0.8
split_index = int(len(dataset) * split_ratio)

train_set = dataset[:split_index]
test_set = dataset[split_index:]

# Write the split datasets to separate files
with open('train_set.json', 'w') as file:
    json.dump(train_set, file, indent=4)

with open('test_set.json', 'w') as file:
    json.dump(test_set, file, indent=4)

# Output the split datasets
print("Train Set:")
for sample in train_set:
    print(sample)
print("\nTest Set:")
for sample in test_set:
    print(sample)

