from typing import List

import pandas as pd
import torch
from tqdm import tqdm

from src.config.config import ModelConfig, load_config_from_json
from src.loader.class_loader import Cluster, Dataset
from src.model.model import Model
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import TrainingArguments, Trainer
from transformers import DataCollatorForSeq2Seq
from sklearn.model_selection import train_test_split

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained("VietAI/vit5-large-vietnews-summarization")
model = AutoModelForSeq2SeqLM.from_pretrained("VietAI/vit5-large-vietnews-summarization")


def convert_examples_to_features(document: str, summary: str):
    input_encodings = tokenizer(document, max_length=1024, truncation=True)

    with tokenizer.as_target_tokenizer():
        target_encodings = tokenizer(summary, max_length=256, truncation=True)

    return {
        "input_ids": input_encodings["input_ids"],
        "attention_mask": input_encodings["attention_mask"],
        "labels": target_encodings["input_ids"]
    }


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, document, summary, encodings, attention_mask, labels):
        self.document = document
        self.summary = summary
        self.encodings = encodings
        self.attention_mask = attention_mask
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def load_data(train_dataset: Dataset):
    print("Loading train data...")
    documents = []
    summaries = []
    for cluster in tqdm(train_dataset.clusters):
        document_cl = ""
        for document in cluster.documents:
            document_cl += document.text_container.raw_str + " "
        documents.append(document_cl)
        summaries.append(cluster.summary.raw_str)

    input_ids = []
    attention_mask = []
    labels = []

    for i in range(0, len(train_dataset.clusters)):
        features = convert_examples_to_features(documents[i], summaries[i])
        input_ids.append(features["input_ids"])
        attention_mask.append(features["attention_mask"])
        labels.append(features["labels"])

    data_set = pd.DataFrame({"document": documents,
                             "summary": summaries,
                             "input_ids": input_ids,
                             "attention_mask": attention_mask,
                             "labels": labels})

    return MyDataset(data_set["input_ids"], data_set["labels"])


class Vit5(Model):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.train_dataset = None
        self.val_dataset = None
        self.config = config

    def training(self, train_dataset: Dataset, val_dataset=None) -> None:
        if val_dataset is None:
            print("No validation dataset")
            return

        self.train_dataset = load_data(train_dataset)
        self.val_dataset = load_data(val_dataset)

        seq2seq_data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

        training_args = TrainingArguments(
            output_dir='vit5-abmusu', num_train_epochs=1, warmup_steps=500,
            per_device_train_batch_size=1, per_device_eval_batch_size=1,
            weight_decay=0.01, logging_steps=10, push_to_hub=False,
            evaluation_strategy='steps', eval_steps=500, save_steps=1e6,
            gradient_accumulation_steps=16
        )

        trainer = Trainer(model=model, args=training_args, tokenizer=tokenizer,
                          data_collator=seq2seq_data_collator,
                          train_dataset=self.train_dataset,
                          eval_dataset=self.val_dataset)

        trainer.train()

    def predict(self, cluster: Cluster) -> (List[str], List[float]):
        pass


if __name__ == '__main__':
    from src.loader.class_loader import Cluster, load_cluster

    SOURCE = 'sent_splitted_token'

    train_dataset = load_cluster(
        "/home/hvn/Documents/dskt/vlsp-final-year/dataset/vlsp_2022_abmusu_train_data_new.jsonl",
        1,
    )

    val_dataset = load_cluster(
        "/home/hvn/Documents/dskt/vlsp-final-year/dataset/vlsp_2022_abmusu_validation_data_new.jsonl",
        1,
    )

    train_dataset.set_source(SOURCE)
    config = load_config_from_json()

    vit5 = Vit5(config.models[0])
    vit5.training(train_dataset, val_dataset)
