from packerages import *


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class DualEncoder(torch.nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids, attention_mask=attention_mask)
        return self.mean_pooling(outputs, attention_mask)

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

class MultiPositiveDataset(Dataset):
    def __init__(self, samples, tokenizer, max_length=128):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        query, doc, query_id = self.samples[idx]

        query_enc = self.tokenizer(
            query,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        doc_enc = self.tokenizer(
            doc,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'query_input_ids': query_enc['input_ids'].squeeze(0),
            'query_attention_mask': query_enc['attention_mask'].squeeze(0),
            'doc_input_ids': doc_enc['input_ids'].squeeze(0),
            'doc_attention_mask': doc_enc['attention_mask'].squeeze(0),
            'query_id': torch.tensor(query_id)
        }
def collate_fn(batch):
    collated = {
        'query_input_ids': pad_sequence([x['query_input_ids'] for x in batch], batch_first=True),
        'query_attention_mask': pad_sequence([x['query_attention_mask'] for x in batch], batch_first=True),
        'doc_input_ids': pad_sequence([x['doc_input_ids'] for x in batch], batch_first=True),
        'doc_attention_mask': pad_sequence([x['doc_attention_mask'] for x in batch], batch_first=True),
        'query_ids': torch.stack([x['query_id'] for x in batch])
    }
    return collated


def train(model, dataloader, optimizer, device, temperature=0.05):
    model.train()
    total_loss = 0

    for batch in tqdm(dataloader,desc="Training"):
        # 移动数据到设备
        query_ids = batch['query_ids'].to(device)
        query_input_ids = batch['query_input_ids'].to(device)
        query_attention_mask = batch['query_attention_mask'].to(device)
        doc_input_ids = batch['doc_input_ids'].to(device)
        doc_attention_mask = batch['doc_attention_mask'].to(device)

        # 获取嵌入
        query_embeds = model(query_input_ids, query_attention_mask)
        doc_embeds = model(doc_input_ids, doc_attention_mask)

        # 归一化
        qurey_norm = torch.nn.functional.normalize(query_embeds, p=2, dim=1)
        doc_norm = torch.nn.functional.normalize(doc_embeds, p=2, dim=1)


        # 计算相似度矩阵
        scores = torch.mm(qurey_norm, doc_norm.T) / temperature

        # 构建多标签矩阵
        batch_size = scores.size(0)
        labels = (query_ids.unsqueeze(1) == query_ids.unsqueeze(0)).float().to(device)

        # 计算损失
        loss = F.binary_cross_entropy_with_logits(scores, labels)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with open('/stu-3035/data/all-data/Task1/PT/train_data.json','r',encoding='utf-8') as fp:
        data = json.load(fp)
        train_samples = []
        for query_id, (query, docs) in enumerate(data.items()):
            for doc in docs:
                train_samples.append((query, doc, query_id))
    tokenizer = AutoTokenizer.from_pretrained("/stu-3035/pretrain_model/all-MiniLM-L12-v2")
    model = DualEncoder("/stu-3035/pretrain_model/all-MiniLM-L12-v2").to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    # 创建数据加载器
    dataset = MultiPositiveDataset(train_samples, tokenizer)
    dataloader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=True,
        collate_fn=collate_fn
    )

    # 训练循环
    for epoch in range(30):
        avg_loss = train(model, dataloader, optimizer, device)
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), '/stu-3035/save_pretrain_model/PT_retrieval_model.pth')



# with open("/stu-3035/data/all-data/Task1/PT/corpus.json",'r',encoding='utf-8') as fp:
#     corpus = json.load(fp)
    # new_queries = {}
    # for query in corpus:
    #     new_queries[query["docid"]] = query["text"]
    # with open("/stu-3035/data/all-data/Task1/PT/corpus.json",'w',encoding='utf-8') as f:
    #     json.dump(new_queries,f)

#
#
# with open("/stu-3035/data/all-data/Task1/PT/joker_2025_task1_qrels_train_pt.json",'r',encoding='utf-8') as fp:
#     qrels = json.load(fp)
#     num = 0
#     post_qrels = []
#     for qrel in qrels:
#         if int(qrel["qrel"]) == 1:
#             post_qrels.append(qrel)
#
#
# with open("/stu-3035/data/all-data/Task1/PT/query_data.json",'r',encoding='utf-8') as fp:
#     queries = json.load(fp)


    # new_queries = {}
    # for query in queries:
    #     new_queries[query["qid"]] = query["query"]
    # with open("/stu-3035/data/all-data/Task1/PT/query_data.json",'w',encoding='utf-8') as f:
    #     json.dump(new_queries,f)

# data = []
# for qrel in qrels:
#     data.append({"qid":queries[qrel["qid"]],"text":corpus[qrel["docid"]],"label":int(qrel["qrel"])})
# with open("/stu-3035/data/all-data/Task1/PT/Humor_train_data.json",'w',encoding='utf-8') as fp:
#     json.dump(data,fp)


