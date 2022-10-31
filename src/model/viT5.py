from typing import List

import torch

from src.config.config import ModelConfig, load_config_from_json
from src.loader.class_loader import Cluster
from src.model.model import Model
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("VietAI/vit5-large-vietnews-summarization")
model = AutoModelForSeq2SeqLM.from_pretrained("VietAI/vit5-large-vietnews-summarization")


class ViT5(Model):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

    def predict(self, cluster: Cluster) -> (List[str], List[float]):
        text = ""
        cl_sentences = cluster.get_all_sents()
        for sent in cl_sentences[:-1]:
            text += sent + ". "
        text += cl_sentences[-1]

        encoding = tokenizer(text, return_tensors="pt")
        input_ids, attention_masks = encoding["input_ids"], encoding["attention_mask"]
        outputs = model.generate(
            input_ids=input_ids, attention_mask=attention_masks,
            max_length=256,
            early_stopping=True
        )

        sentences = []

        for output in outputs:
            line = tokenizer.decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            sentences.append(line)

        print(sentences)

        return sentences, []


if __name__ == '__main__':
    from src.loader.class_loader import Cluster, load_cluster

    dataset = load_cluster(
        "/home/hvn/Documents/dskt/vlsp-final-year/dataset/vlsp_2022_abmusu_train_data_new.jsonl",
        1,
    )

    config = load_config_from_json()
    viT5 = ViT5(config.models[0])
    viT5.predict(dataset.clusters[0])
