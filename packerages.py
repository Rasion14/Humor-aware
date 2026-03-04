import random
import json
import time
from datetime import datetime
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score,precision_recall_fscore_support
from tqdm import tqdm
import numpy as np
from rank_bm25 import BM25Okapi
from collections import defaultdict
import torch
from torch.utils.data import Dataset,DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch import nn
from torch.nn import functional as F
from sentence_transformers import SentenceTransformer,models
from transformers import (
    RobertaForSequenceClassification,
    RobertaTokenizer,
    Trainer,
    TrainingArguments,
    PreTrainedModel,
    BertModel,
    AutoTokenizer,
    AutoModel,

    get_linear_schedule_with_warmup
)

with open("/stu-3035/data/all-data/Task1/PT/joker_2025_task1_queries_test_pt.json",'r',encoding='utf-8') as fp:
    PT_result = json.load(fp)

    print(len(PT_result))
# with open("/stu-3035/data/all-data/Task1/EN/result.json",'r',encoding='utf-8') as fp:
#     EN_result = json.load(fp)
# all_result = EN_result + PT_result
# with open("/stu-3035/data/all-data/all_result.json",'w',encoding='utf-8') as fp:
#     json.dump(all_result,fp)

# with open("/stu-3035/data/all-data/Task1/EN/joker_task1_retrieval_qrels_train25_EN.json",'r',encoding='utf-8') as fp:
#     qrels = json.load(fp)
#     qrels_data = defaultdict(list)
#     for qrel in qrels:
#         qrels_data[str(qrel["qid"])].append(int(qrel["docid"]))
#     with open("/stu-3035/data/all-data/Task1/EN/qrels.json",'w',encoding='utf-8') as f:
#         json.dump(qrels_data,f)

