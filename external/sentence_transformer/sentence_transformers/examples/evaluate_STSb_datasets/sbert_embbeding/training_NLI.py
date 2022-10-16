"""
The system trains BERT (or any other transformer model like RoBERTa, DistilBERT etc.) on the SNLI + MultiNLI (AllNLI) dataset
with softmax loss function. At every 1000 training steps, the model is evaluated on the
STS benchmark dataset
Usage:
python training_nli.py
OR
python training_nli.py pretrained_transformer_model_name
"""
from torch.utils.data import DataLoader
import math
from sentence_transformers import models, losses
from sentence_transformers import SentencesDataset, LoggingHandler, SentenceTransformer
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.readers import *
import logging
from datetime import datetime
import sys
import os
import argparse
import numpy as np
np.random.seed(42)
torch.manual_seed(42)

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--batch_size', type=int, default=24)
parser.add_argument('--evaluation_steps', type=int, default= 1000)
parser.add_argument('--ckpt_path', type=str, default = "./output")
parser.add_argument('--num_epochs', type=int, default ="1")
parser.add_argument('--NLIdata_path', type=str, default = "./DataNLI")
parser.add_argument('--STSdata_path', type=str, default = "./STSdata_path")
parser.add_argument('--pre_trained_path', type=str, default = "./PhoBERT")
parser.add_argument('--vncorenlp_path', type=str, default = "./VnCoreNLP/VnCoreNLP-1.1.1.jar")
parser.add_argument('--bpe_path', type=str, default = "./PhoBERT")
args = parser.parse_args()


#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout

if not os.path.exists(args.ckpt_path):
    os.mkdir(args.ckpt_path)

# #You can specify any huggingface/transformers pre-trained model here, for example, bert-base-uncased, roberta-base, xlm-roberta-base
# model_name = sys.argv[1] if len(sys.argv) > 1 else 'bert-base-uncased'

# Read the dataset
nli_reader = NLIDataReader(args.NLIdata_path)
sts_reader = STSBenchmarkDataReader(args.STSdata_path)

train_num_labels = nli_reader.get_num_labels()


# Use Huggingface/transformers model (like BERT, RoBERTa, XLNet, XLM-R) for mapping tokens to embeddings
word_embedding_model = models.PhoBERT(args.pre_trained_path, tokenizer_args={'vncorenlp_path':args.vncorenlp_path, 'bpe_path':args.bpe_path})

# Apply mean pooling to get one fixed sized sentence vector
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                               pooling_mode_mean_tokens=True,
                               pooling_mode_cls_token=False,
                               pooling_mode_max_tokens=False)

model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

# for param in model.parameters():
#     param.requires_grad = False

# Convert the dataset to a DataLoader ready for training
logging.info("Read XNLI train dataset")
train_data = SentencesDataset(nli_reader.get_examples('train.gz'), model=model)
train_dataloader = DataLoader(train_data, shuffle=True, batch_size=args.batch_size)
train_loss = losses.SoftmaxLoss(model=model, sentence_embedding_dimension=model.get_sentence_embedding_dimension(), num_labels=train_num_labels)



#convert the dataset to a Dataloader ready for dev
logging.info("Read STSbenchmark dev dataset")
dev_data = SentencesDataset(examples=sts_reader.get_examples('sts-dev_vi.csv'), model=model)
dev_dataloader = DataLoader(dev_data, shuffle=False, batch_size=args.batch_size)
evaluator = EmbeddingSimilarityEvaluator(dev_dataloader)

# Configure the training
# warmup_steps = math.ceil(len(train_dataloader) * args.num_epochs / args.batch_size * 0.1) #10% of train data for warm-up
warmup_steps = math.ceil(len(train_dataloader) * args.num_epochs * 0.1) #10% of train data for warm-up - recommended
logging.info("Warmup-steps: {}".format(warmup_steps))



# Train the model
model.fit(train_objectives=[(train_dataloader, train_loss)],
          evaluator=evaluator,
          epochs=args.num_epochs,
          evaluation_steps=args.evaluation_steps,
          warmup_steps=warmup_steps,
          output_path=args.ckpt_path
          )



##############################################################################
#
# Load the stored model and evaluate its performance on XNLI dataset
#
##############################################################################

model = SentenceTransformer(args.ckpt_path)
test_data = SentencesDataset(examples=sts_reader.get_examples("sts-test_vi.csv"), model=model)
test_dataloader = DataLoader(test_data, shuffle=False, batch_size=args.batch_size)
evaluator = EmbeddingSimilarityEvaluator(test_dataloader)

model.evaluate(evaluator)