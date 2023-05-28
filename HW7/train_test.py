import gc

from dataset import QADataset
import torch
from torch.utils.data import DataLoader
from transformers import AdamW, BertForQuestionAnswering, BertTokenizerFast,get_linear_schedule_with_warmup
from tqdm.auto import tqdm
from function import read_data,evaluate

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # get model
    model = BertForQuestionAnswering.from_pretrained("./pre_model").to(device)
    tokenizer = BertTokenizerFast.from_pretrained("./pre_model/")
    # get data
    train_questions, train_paragraphs = read_data("./hw7_train.json")
    dev_questions, dev_paragraphs = read_data("./hw7_dev.json")
    test_questions, test_paragraphs = read_data("./hw7_test.json")
    train_questions_tokenized = tokenizer([train_question["question_text"] for train_question in train_questions], add_special_tokens=False)
    dev_questions_tokenized = tokenizer([dev_question["question_text"] for dev_question in dev_questions], add_special_tokens=False)
    test_questions_tokenized = tokenizer([test_question["question_text"] for test_question in test_questions], add_special_tokens=False)
    train_paragraphs_tokenized = tokenizer(train_paragraphs, add_special_tokens=False)
    dev_paragraphs_tokenized = tokenizer(dev_paragraphs, add_special_tokens=False)
    test_paragraphs_tokenized = tokenizer(test_paragraphs, add_special_tokens=False)
    # get dataloader
    train_set = QADataset("train", train_questions, train_questions_tokenized, train_paragraphs_tokenized)
    dev_set = QADataset("dev", dev_questions, dev_questions_tokenized, dev_paragraphs_tokenized)
    test_set = QADataset("test", test_questions, test_questions_tokenized, test_paragraphs_tokenized)
    train_batch_size = 8
    train_loader = DataLoader(train_set, batch_size=train_batch_size, shuffle=True, pin_memory=True)
    dev_loader = DataLoader(dev_set, batch_size=1, shuffle=False, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, pin_memory=True)
    # train
    num_epoch = 2
    print_step = 100
    total_steps = num_epoch * len(train_loader)
    acc_steps = 4
    learning_rate = 1e-5
    max_acc = 0
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=total_steps // acc_steps)
    model.train()
    print("Start Training")
    for epoch in range(num_epoch):
        step = 1
        train_loss = 0
        train_acc = 0
        optimizer.zero_grad()
        for data in tqdm(train_loader):
            data = [i.to(device) for i in data]
            output = model(input_ids=data[0], token_type_ids=data[1], attention_mask=data[2], start_positions=data[3],
                           end_positions=data[4])
            start_index = torch.argmax(output.start_logits, dim=1)
            end_index = torch.argmax(output.end_logits, dim=1)
            train_acc += ((start_index == data[3]) & (end_index == data[4])).float().mean()
            train_loss += output.loss
            output.loss.backward()

            step += 1
            if step % acc_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

            if step % print_step == 0:
                lr = optimizer.state_dict()['param_groups'][0]['lr']
                print(f"Epoch {epoch} | Step {step} | loss = {train_loss.item() / print_step:.3f}, acc = {train_acc / print_step:.3f}, lr={lr}")
                train_loss =0
                train_acc = 0

        print("Start Valid")
        model.eval()
        with torch.no_grad():
            dev_acc = 0
            for i, data in enumerate(tqdm(dev_loader)):
                output = model(input_ids=data[0].squeeze(dim=0).to(device),
                               token_type_ids=data[1].squeeze(dim=0).to(device),
                               attention_mask=data[2].squeeze(dim=0).to(device))

                dev_acc += evaluate(data, output, tokenizer) == dev_questions[i]["answer_text"]
            acc = dev_acc/len(dev_loader)
            print(f"Validation | Epoch {epoch} | acc = {dev_acc / len(dev_loader):.3f}")
            if acc > max_acc:
                max_acc = acc
                print("Start Saving Model")
                model_save_dir = "saved_model"
                model.save_pretrained(model_save_dir)
        model.train()
