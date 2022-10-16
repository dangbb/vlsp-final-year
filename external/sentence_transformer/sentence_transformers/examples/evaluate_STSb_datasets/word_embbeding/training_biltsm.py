"""
This example runs a BiLSTM after the word embedding lookup. The output of the BiLSTM is than pooled,
for example with max-pooling (which gives a system like InferSent) or with mean-pooling.

Note, you can also pass BERT embeddings to the BiLSTM.
"""
import torch
from torch.utils.data import DataLoader
import math
from sentence_transformers import models, losses
from sentence_transformers import SentencesDataset, LoggingHandler, SentenceTransformer
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.readers import *
import logging
from datetime import datetime
from sentence_transformers.models.tokenizer.WordTokenizer import VIETNAM_STOP_WORDS_SEGMENTATION
from sentence_transformers.models.tokenizer.VietnameseTokenizer import *
import argparse
import numpy as np
np.random.seed(42)
torch.manual_seed(42)
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--batch_size', type=int, default=24)
parser.add_argument('--ckpt_path', type=str, default = "./output")
parser.add_argument('--num_epochs', type=int, default ="1")
parser.add_argument('--data_path', type=str, default = "./stsbenchmark'")
parser.add_argument('--vncorenlp_path', type=str, default = "./VnCoreNLP/VnCoreNLP-1.1.1.jar")
parser.add_argument('--embeddings_file_path', type=str, default= "./glove.6B.300d.txt.gz")
args = parser.parse_args()

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout
if not os.path.exists(args.ckpt_path):
    os.mkdir(args.ckpt_path)
    
# Read the dataset
sts_reader = STSBenchmarkDataReader(args.data_path)



# Map tokens to traditional word embeddings like GloVe
word_embedding_model = models.WordEmbeddings.from_text_file(embeddings_file_path=args.embeddings_file_path, tokenizer = VietnameseTokenizer(stop_words=VIETNAM_STOP_WORDS_SEGMENTATION, vncorenlp_path=args.vncorenlp_path))

lstm = models.LSTM(word_embedding_dimension=word_embedding_model.get_word_embedding_dimension(), hidden_dim=150)

# Apply mean pooling to get one fixed sized sentence vector
pooling_model = models.Pooling(lstm.get_word_embedding_dimension(),
                               pooling_mode_mean_tokens=False,
                               pooling_mode_cls_token=False,
                               pooling_mode_max_tokens=True)


model = SentenceTransformer(modules=[word_embedding_model, lstm, pooling_model])


# Convert the dataset to a DataLoader ready for training
logging.info("Read STSbenchmark train dataset")
train_data = SentencesDataset(sts_reader.get_examples('sts-train_vi.csv'), model=model)
train_dataloader = DataLoader(train_data, shuffle=True, batch_size=args.batch_size)
train_loss = losses.CosineSimilarityLoss(model=model)

logging.info("Read STSbenchmark dev dataset")
dev_data = SentencesDataset(examples=sts_reader.get_examples('sts-dev_vi.csv'), model=model)
dev_dataloader = DataLoader(dev_data, shuffle=False, batch_size=args.batch_size)
evaluator = EmbeddingSimilarityEvaluator(dev_dataloader)

# Configure the training
warmup_steps = math.ceil(len(train_data) * args.num_epochs / args.batch_size * 0.1) #10% of train data for warm-up
logging.info("Warmup-steps: {}".format(warmup_steps))

# Train the model
model.fit(train_objectives=[(train_dataloader, train_loss)],
          evaluator=evaluator,
          epochs=args.num_epochs,
          warmup_steps=warmup_steps,
          output_path=args.ckpt_path
          )



##############################################################################
#
# Load the stored model and evaluate its performance on STS benchmark dataset
#
##############################################################################

model = SentenceTransformer(args.ckpt_path)
test_data = SentencesDataset(examples=sts_reader.get_examples("sts-test_vi.csv"), model=model)
test_dataloader = DataLoader(test_data, shuffle=False, batch_size=args.batch_size)
evaluator = EmbeddingSimilarityEvaluator(test_dataloader)

model.evaluate(evaluator)