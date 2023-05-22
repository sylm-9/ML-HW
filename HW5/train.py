from loss import LabelSmoothedCrossEntropyCriterion
from optimizer import NoamOpt
from data_loader import load_data_iterator, get_task, get_config
from model import get_model
import torch
from function import train_one_epoch,validate_and_save


if __name__ == "__main__":
    config = get_config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    task = get_task()
    task.load_dataset(split="train", epoch=1, combine=True)
    task.load_dataset(split="valid", epoch=1)
    model = get_model()
    criterion = LabelSmoothedCrossEntropyCriterion(smoothing=0.1, ignore_index=task.target_dictionary.pad())
    model = model.to(device)
    criterion = criterion.to(device)
    optimizer = NoamOpt(model_size=256, factor=2., warmup=4000,
                        optimizer=torch.optim.AdamW(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9,
                                                    weight_decay=0.0001))
    epoch_itr = load_data_iterator(task, "train", config.start_epoch, config.max_tokens, config.num_workers)
    while epoch_itr.next_epoch_idx <= config.max_epoch:
        train_one_epoch(epoch_itr, model, task, criterion, optimizer, config.accum_steps)
        stats = validate_and_save(model, task, criterion, optimizer, epoch=epoch_itr.epoch)
        epoch_itr = load_data_iterator(task, "train", epoch_itr.next_epoch_idx, config.max_tokens, config.num_workers)
