from typing import List

from src.config.config import ModelConfig, load_config_from_json
from src.loader.class_loader import Cluster
from src.model.mmr import MMR
from src.model.model import Model

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


class MmrFtViT5(Model):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.mmr_model = MMR(config, sentences_count=[config.params[1]])
        self.tokenizer = AutoTokenizer.from_pretrained("VietAI/vit5-large-vietnews-summarization")
        self.viT5_model = AutoModelForSeq2SeqLM.from_pretrained("VietAI/vit5-large-vietnews-summarization")

    def predict(self, cluster: Cluster) -> (List[str], List[float]):
        sentences, scores = self.mmr_model.predict(cluster)
        text = ""
        for sent in sentences[:-1]:
            text += sent + " "
        text += sentences[-1]

        encoding = self.tokenizer(text, return_tensors="pt")
        input_ids, attention_masks = encoding["input_ids"], encoding["attention_mask"]
        outputs = self.viT5_model.generate(
            input_ids=input_ids, attention_mask=attention_masks,
            max_length=1e9,
            early_stopping=False
        )

        t5_sentences = []

        for output in outputs:
            line = self.tokenizer.decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            t5_sentences.append(line)

        print(sentences)
        print(t5_sentences)

        return t5_sentences, scores


if __name__ == '__main__':
    from src.loader.class_loader import Cluster, load_cluster

    dataset = load_cluster(
        "/home/hvn/Documents/dskt/vlsp-final-year/dataset/vlsp_abmusu_test_data.jsonl",

    )

    config = load_config_from_json()
    mmr_x_viT5 = MmrFtViT5(config.models[0])
    mmr_x_viT5.predict(dataset.clusters[299])
