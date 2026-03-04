from packerages import *

class TestDataset(Dataset):
    def __init__(self, corpus, tokenizer, max_length=128):
        self.samples = corpus
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text, docid,query, relevance_score = self.samples[idx]




        input_ids = self.tokenizer.encode_plus(
            query,
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': input_ids['input_ids'].squeeze(0),
            'attention_mask': input_ids['attention_mask'].squeeze(0),
            'doc_id': torch.tensor(docid,dtype=torch.long),
            'scores':torch.tensor(relevance_score,dtype=torch.float)
        }

with open ("/stu-3035/data/all-data/Task1/EN/test.json",'r',encoding='utf-8') as fp:
    data = json.load(fp)
    # print(data)
    # print(len(data))
    data_dict = defaultdict(list)
    for text in tqdm(data,desc="Classify"):
        data_dict[text[3]].append((text[0],text[1],text[2],text[4]))
    # with open("/stu-3035/data/all-data/Task1/EN/test2.json",'w',encoding='utf-8') as f:
    #     json.dump(data_dict, f)
    outputs_data = []
    qid_list = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained('/stu-3035/pretrain_model/roberta-base')
    model = RobertaForSequenceClassification.from_pretrained(
            '/stu-3035/pretrain_model/roberta-base',
            num_labels=2
        )
    model.load_state_dict(torch.load("/stu-3035/save_pretrain_model/EN_Humor_aware_model.bin",weights_only=True))
    model.to(device)
    model.eval()
    for qid,corpus in tqdm(data_dict.items(),desc="Testting"):

        docids = torch.empty(0,dtype=torch.long)
        scores = torch.empty(0,dtype=torch.float)
        total_pred = torch.empty(0,dtype=torch.long).to(device)
        datasets = TestDataset(corpus,tokenizer,max_length=256)
        dataload = DataLoader(datasets,batch_size=400,shuffle=True)
        with torch.no_grad():
            for batch in tqdm(dataload):
                input_ids=batch['input_ids'].to(device)
                attention_mask=batch['attention_mask'].to(device)
                docid = batch['doc_id']
                score = batch['scores']


                outputs = model(
                    input_ids,
                    attention_mask
                )
                logits = outputs.logits
                logits = torch.softmax(logits,dim=1)
                values,preds = torch.max(logits.detach(),dim=1)
                score = 0.3 * score + 0.7 * values.cpu()
                scores = torch.cat((scores,score))
                docids = torch.cat((docids,docid))
                total_pred = torch.cat((total_pred,preds))
        total_pred = total_pred.cpu()
        post_indices = torch.where(total_pred==1)[0]
        scores = scores[post_indices]
        docids = docids[post_indices]
        rank_indices = torch.argsort(scores,descending=True)
        for i,j in enumerate(rank_indices):
            qid_list.append(
                {
                    "run_id":"Rasion_task_1_SenTransF+Roberta",
                    "manual":0,
                    "qid":qid,
                    "docid":str(int(docids[j])),
                    "rank":i+1,
                    "score":round(float(scores[j]),2)
                }
            )
            if i+1 >= 1000:
                break

    with open("/stu-3035/data/all-data/Task1/EN/result2.json",'w',encoding='utf-8') as f:
        json.dump(qid_list,f)





