import pandas as pd

# need sentencepiece

# T5 Stuff
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch

train_df = pd.read_csv("./data/train.csv")
test_df = pd.read_csv("./data/test.csv")

attr_id = "id"
attr_label = "label"
attr_tweet = "tweet"

train_df = train_df.drop(columns=[attr_id])
test_df = test_df.drop(columns=[attr_id])

print(train_df.shape)

# label 0 == no hate
hateful_map = {0: 'appropriate', 1: 'hateful'}
train_df = train_df.replace({attr_label: hateful_map})
print(train_df.head(10))

# Model stuff https://huggingface.co/docs/transformers/model_doc/t5
model = T5ForConditionalGeneration.from_pretrained("t5-small")
tokenizer = T5Tokenizer.from_pretrained("t5-small")

# when generating, we will use the logits of right-most token to predict the next token
# so the padding should be on the left
tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token  # to avoid an error

task_prefix = "twitter sentiment analysis:"


max_source_length = 512
max_target_length = 128


# Encode our data
encodings = tokenizer(
    [task_prefix + sentence for sentence in train_df[attr_tweet]],
    padding="longest",
    max_length=max_source_length,
    truncation=True,
    return_tensors="pt"
)

target_encoding = tokenizer(
    [label for label in train_df[attr_label]],
    padding="longest",
    max_length=max_target_length,
    truncation=True
)

labels = torch.tensor(target_encoding.input_ids)

# This prevents torch optimizer from considering padding in innaccuracies
labels[labels == tokenizer.pad_token_id] = -100

print("about to train")

loss = model(
    input_ids=encodings.input_ids,
    attention_mask=encodings.attention_mask,
    labels=labels
).loss

print('LOSS', loss)


test_inputs = tokenizer(
    [task_prefix + sentence for sentence in test_df[attr_tweet]],
    return_tensors='pt',
    padding=True
)

output_sequences = model.generate(
    input_ids=test_inputs["input_ids"],
    attention_mask=test_inputs["attention_mask"],
    do_sample=False,  # disable sampling to test if batching affects output
)

output = tokenizer.batch_decode(output_sequences, skip_special_tokens=True)

print(output)
with open('./data/predicted.txt', 'w') as f:
    for line in output:
        f.write(line)

print("wrote")








