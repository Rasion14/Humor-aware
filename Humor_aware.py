from packerages import *
from load_data import *


class TextDataset(Dataset):
    def __init__(self, data,tokenizer, max_len):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = str(self.data[idx][1])
        label = int(self.data[idx][2])
        query = str(self.data[idx][0])

        encoding = self.tokenizer.encode_plus(
            query,
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            # 'token_type_ids':encoding['token_type_ids'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }
def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
set_seed()

# 配置参数
class Config:
    MODEL_NAME = '/stu-3035/pretrain_model/roberta-base'
    MAX_LEN = 256
    BATCH_SIZE = 128
    EVAL_BATCH_SIZE=128
    EPOCHS = 10
    LEARNING_RATE = 4e-5
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_model(model, data_loader, optimizer, scheduler, device):
    model.train()
    total_loss = 0

    for batch in tqdm(data_loader,desc="Training"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        # token_type_ids = batch['token_type_ids'].to(device)
        labels = batch['label'].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            # token_type_ids = token_type_ids,
            labels=labels
        )

        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    return total_loss / len(data_loader)


def eval_model(model, data_loader, device):
    model.eval()
    predictions = []
    true_labels = []
    total_loss = 0

    with torch.no_grad():
        for batch in tqdm(data_loader,desc="Evalling"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            # token_type_ids = batch['token_type_ids'].to(device)
            labels = batch['label'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                # token_type_ids=token_type_ids,
                labels=labels
            )

            loss = outputs.loss
            total_loss += loss.item()

            logits = outputs.logits
            preds = torch.argmax(logits, dim=1).cpu().numpy()

            predictions.extend(preds)
            true_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions, average='weighted')
    precision, recall, _, _ = precision_recall_fscore_support(
        y_true=true_labels,
        y_pred=predictions,
        average="binary"
    )
    avg_loss = total_loss / len(data_loader)

    return accuracy, f1, avg_loss, precision,recall

def load_data():
    with open("/stu-3035/data/all-data/Task1/EN/train.json", 'r', encoding='utf-8') as fp:
        data = json.load(fp)
    with open("/stu-3035/data/all-data/Task1/EN/qrels.json", 'r', encoding='utf-8') as f:
        qrels = json.load(f)
        post_data = []
        neg_data = []
        for sample in tqdm(data,desc="Loading"):
            if int(sample[1]) in qrels[str(sample[3])]:
                post_data.append((sample[2],sample[0],1))
            else:
                neg_data.append((sample[2],sample[0],0))
        indices = np.random.randint(0,len(neg_data),size=int(len(neg_data)/20))
        neg_data1 = [neg_data[i] for i in indices]
        train_data = post_data + neg_data1
        random.shuffle(train_data)

    return train_data,train_data
    # return train_data,val_data
def main():
    config = Config()

    # 加载数据
    train_df, val_df = load_data()

    # 初始化tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)

    # 创建数据集
    train_dataset = TextDataset(
        data=train_df,
        tokenizer=tokenizer,
        max_len=config.MAX_LEN
    )

    val_dataset = TextDataset(
        data=val_df,
        tokenizer=tokenizer,
        max_len=config.MAX_LEN
    )

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=2
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.EVAL_BATCH_SIZE,
        shuffle=False,
        num_workers=2
    )

    # 初始化模型
    model = RobertaForSequenceClassification.from_pretrained(
        config.MODEL_NAME,
        num_labels=2
    )
    model = model.to(config.DEVICE)

    # 设置优化器和调度器
    optimizer = AdamW(model.parameters(), lr=config.LEARNING_RATE)
    total_steps = len(train_loader) * config.EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=total_steps * 0.1,
        num_training_steps=total_steps
    )

    # 训练循环
    best_f1 = 0
    for epoch in range(config.EPOCHS):
        print(f'Epoch {epoch + 1}/{config.EPOCHS}')
        print('-' * 10)

        train_loss = train_model(model, train_loader, optimizer, scheduler, config.DEVICE)
        val_accuracy, val_f1, val_loss, precision,recall = eval_model(model, val_loader, config.DEVICE)

        print(f'Train loss: {train_loss:.4f}')
        print(f'Validation loss: {val_loss:.4f}')
        print(f'Validation Accuracy: {val_accuracy:.4f}')
        print(f'Validation F1: {val_f1:.4f}')
        print(f'Validation Precision: {precision:.4f}')
        print(f'Validation Recall: {recall:.4f}')

        # 保存最佳模型
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), '/stu-3035/save_pretrain_model/EN_Humor_aware_model.bin')
            print(f'Best model saved with F1: {best_f1:.4f}')

    print('Training complete!')

    # 测试模型示例
    # test_text = "This product exceeded my expectations"
    # inputs = tokenizer(
    #     test_text,
    #     padding='max_length',
    #     truncation=True,
    #     max_length=config.MAX_LEN,
    #     return_tensors="pt"
    # ).to(config.DEVICE)
    #
    # model.load_state_dict(torch.load('best_model.bin'))
    # model.eval()
    # with torch.no_grad():
    #     outputs = model(**inputs)
    #     logits = outputs.logits
    #     prediction = torch.argmax(logits, dim=1).item()
    #
    # print(f"\nTest text: '{test_text}'")
    # print(f'Prediction: {"Positive" if prediction == 1 else "Negative"}')


if __name__ == "__main__":
    while True:
        try:

            main()
            break
        except torch.OutOfMemoryError:
            print("显存资源不够")
            currect_time = datetime.now()
            print(currect_time.strftime("%Y-%m-%d %H:%M:%S"))
            time.sleep(300.0)















