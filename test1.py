from packerages import *
from load_data import *

model_name = "/stu-3035/pretrain_model/paraphrase-multilingual-mpnet-base-v2"
model = DualEncoder(model_name)

# model.load_state_dict(torch.load("/stu-3035/save_pretrain_model/retrieval_model.pth",weights_only=True))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = AutoTokenizer.from_pretrained(model_name)
model.to(device)
with open("/stu-3035/data/all-data/Task1/EN/joker_task1_retrieval_corpus25_EN.json",'r',encoding='utf-8') as fp:
    data = json.load(fp)
    corpus = [doc['text'] for doc in data if type(doc['text']) == str]
    docids = [doc['docid'] for doc in data if type(doc['text']) == str]

# with open("/stu-3035/data/all-data/Task1/EN/joker_task1_retrieval_qrels_train25_EN.json",'r',encoding='utf-8') as fp:
#     labels = json.load(fp)
#     qrels = defaultdict(list)
#     for label in labels:
#         qrels[str(label["qid"])].append(str(label["docid"]))
    # print(qrels)

with open("/stu-3035/data/all-data/Task1/EN/joker_task1_retrieval_queries_test25_EN.json",'r',encoding='utf-8') as fp:
    queries = json.load(fp)
    doc_inputs = tokenizer(corpus, return_tensors='pt', padding=True, truncation=True, max_length=256)
    doc_dataset = torch.utils.data.TensorDataset(
        doc_inputs['input_ids'],
        doc_inputs['attention_mask']
    )
    doc_loader = DataLoader(doc_dataset, batch_size=128)

    doc_embeds = []
    for batch in tqdm(doc_loader, desc="Embedding"):
        batch = [t.to(device) for t in batch]
        with torch.no_grad():
            embeds = model(*batch)
            doc_embeds.append(embeds)
    doc_embeds = torch.cat(doc_embeds)
    test_data = []
    # query_inputs=tokenizer(queries['8'], return_tensors='pt', padding=True, truncation=True, max_length=128)
    # query_inputs = {k: v.to(device) for k, v in query_inputs.items()}
    # with torch.no_grad():
    #     query_embed = model(**query_inputs)
    #
    #
    # # 计算相似度
    # similarities = F.cosine_similarity(query_embed, doc_embeds)
    # values,t_k_indices = similarities.topk(5)
    # print(values)
    # docids = [docid[i] for i in t_k_indices.cpu().numpy()]
    # print(docids)
    for query in tqdm(queries):
        # print(query['qid']
        query_inputs = tokenizer(query['query'], return_tensors='pt', padding=True, truncation=True, max_length=128)
        query_inputs = {k: v.to(device) for k, v in query_inputs.items()}

        with torch.no_grad():
            query_embed = model(query_inputs['input_ids'],query_inputs['attention_mask'])


        # 计算相似度
        similarities = F.cosine_similarity(query_embed, doc_embeds)
        # sort = torch.argsort(similarities,descending=True)
        # for i in sort.cpu().numpy():
        #     print(docids[i])
        # print(similarities[sort])


        postibe_indices = torch.where(similarities > 0.1)[0]
        scores = similarities[postibe_indices]
        for i in zip(postibe_indices.cpu().numpy(),scores.cpu().numpy()):
            # if (corpus[i[0]],int(docid[i[0]]),query["qid"],float(round(i[1],2))) not in test_data:
            test_data.append((corpus[i[0]],int(docids[i[0]]),query["query"],query["qid"],round(float(i[1]),2)))

            # if doc_id in qrels[query["qid"]] and (corpus[i],1) not in train_data:
            #     test_data.append((corpus[i],1))
            # elif doc_id not in qrels[qid] and (corpus[i],0) not in train_data:
            #     train_data.append((corpus[i],0))
    with open("/stu-3035/data/all-data/Task1/EN/test.json",'w',encoding='utf-8') as f:
        json.dump(test_data,f)






