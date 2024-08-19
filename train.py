import json
import random
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import LlamaForSequenceClassification, Trainer, TrainingArguments, LlamaTokenizer, DataCollatorWithPadding

# Read the dataset from the file dataset.json
with open('dataset.json', 'r') as file:
    dataset = json.load(file)

# Shuffle the dataset
random.shuffle(dataset)

# Split the dataset into 80:20 ratio
split_ratio = 0.8
split_index = int(len(dataset) * split_ratio)

train_set = dataset[:split_index]
test_set = dataset[split_index:]

# Save the split datasets to separate files
with open('train_set.json', 'w') as file:
    json.dump(train_set, file, indent=4)

with open('test_set.json', 'w') as file:
    json.dump(test_set, file, indent=4)

# Prepare the training data in the format required for fine-tuning
class SelfHarmDataset(Dataset):
    def __init__(self, samples, tokenizer):
        self.samples = samples
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        post_text = sample['post_text']
        label = 1 if sample['label'] == 'self-harm' else 0
        casual_spans = sample.get('casual_mention_spans', [])
        serious_spans = sample.get('serious_intent_spans', [])
        emojis = sample.get('emojis', [])

        # Create the fine-tuning prompt
        prompt = {
            "instruction": "Analyze the following social media post to determine if it is related to self-harm. Identify spans related to casual mention and serious intent regarding self-harm.",
            "input": {
                "post_text": post_text,
                "emojis": emojis
            },
            "output": {
                "classification": "self-harm" if label == 1 else "non-self-harm",
                "casual_mention_spans": casual_spans,
                "serious_intent_spans": serious_spans
            }
        }

        inputs = self.tokenizer(json.dumps(prompt), truncation=True, padding='max_length', max_length=512, return_tensors="pt")
        inputs = {key: val.squeeze() for key, val in inputs.items()}
        inputs['labels'] = torch.tensor(label)

        return inputs

# Initialize tokenizer
tokenizer = LlamaTokenizer.from_pretrained(<MODEL NAME>)

# Create datasets
train_dataset = SelfHarmDataset(train_set, tokenizer)
val_dataset = SelfHarmDataset(test_set, tokenizer)

# Create data collator
data_collator = DataCollatorWithPadding(tokenizer)

# Initialize model
model = LlamaForSequenceClassification.from_pretrained(<MODEL NAME>, num_labels=2)

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="epoch",
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
)

# Fine-tune the model
trainer.train()

# Function to generate rationales
def generate_rationale(post_text, classification, casual_spans, serious_spans, emojis):
    rationale_prompt = {
        "instruction": "Using the provided classification and extracted spans, generate a rationale explaining why the post is classified as self-harm or non-self-harm. Consider the spans and the emoji meanings provided.",
        "input": {
            "post_text": post_text,
            "classification": classification,
            "casual_mention_spans": casual_spans,
            "serious_intent_spans": serious_spans,
            "emojis": emojis
        },
        "output": ""
    }
    # Tokenize and generate rationale
    rationale_input = tokenizer.encode_plus(json.dumps(rationale_prompt), return_tensors='pt')
    rationale_output = model.generate(input_ids=rationale_input['input_ids'], max_length=150)
    rationale = tokenizer.decode(rationale_output[0], skip_special_tokens=True)
    return rationale

# Example usage
def predict_and_generate_rationale(sample):
    post_text = sample['post_text']
    inputs = tokenizer(post_text, return_tensors='pt', truncation=True, padding='max_length', max_length=512)
    outputs = model(**inputs)
    pred = torch.argmax(outputs.logits, dim=1).item()

    classification = "self-harm" if pred == 1 else "non-self-harm"
    casual_spans = sample.get('casual_mention_spans', [])
    serious_spans = sample.get('serious_intent_spans', [])
    emojis = sample.get('emojis', [])

    rationale = generate_rationale(post_text, classification, casual_spans, serious_spans, emojis)

    print(f"Post: {post_text}")
    print(f"Classification: {classification}")
    print(f"Casual Mention Spans: {casual_spans}")
    print(f"Serious Intent Spans: {serious_spans}")
    print(f"Rationale: {rationale}\n")

# Predict and generate rationales for test samples
for sample in test_set:
    predict_and_generate_rationale(sample)

