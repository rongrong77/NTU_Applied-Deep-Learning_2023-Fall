import json
from sklearn.model_selection import train_test_split

# Load the original data from data.json
with open('./data/train_1.json', 'r',encoding='utf-8') as f:
    data = json.load(f)

# Split the data into training and validation sets
train_data, valid_data = train_test_split(data, test_size=0.2, random_state=42)

# Save the training data to train.json
with open('./data/train_2.json', 'w',encoding='utf-8') as f:
    json.dump(train_data, f, indent=2,ensure_ascii=False)

# Save the validation data to valid.json
with open('./data/valid_2.json', 'w',encoding='utf-8') as f:
    json.dump(valid_data, f, indent=2,ensure_ascii=False)
