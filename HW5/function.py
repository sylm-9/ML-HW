import shutil
from pathlib import Path
import numpy as np
from data_loader import load_data_iterator, get_task, get_config
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm.auto as tqdm
from fairseq import utils
import sacrebleu
from fairseq.data import iterators
from torch.cuda.amp import GradScaler, autocast

config = get_config()
gnorms = []
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_one_epoch(epoch_itr, model, task, criterion, optimizer, accum_steps=1):
    itr = epoch_itr.next_epoch_itr(shuffle=True)
    itr = iterators.GroupedIterator(itr, accum_steps)

    stats = {"loss": []}
    scaler = GradScaler()

    model.train()
    progress = tqdm.tqdm(itr, desc=f"train epoch {epoch_itr.epoch}", leave=False)
    for samples in progress:
        model.zero_grad()
        accum_loss = 0
        sample_size = 0
        for i, sample in enumerate(samples):
            if i == 1:
                torch.cuda.empty_cache()

            sample = utils.move_to_cuda(sample, device=device)
            target = sample["target"]
            sample_size_i = sample["ntokens"]
            sample_size += sample_size_i

            with autocast():
                net_output = model.forward(**sample["net_input"])
                lprobs = F.log_softmax(net_output[0], -1)
                loss = criterion(lprobs.view(-1, lprobs.size(-1)), target.view(-1))
                accum_loss += loss.item()
                scaler.scale(loss).backward()

        scaler.unscale_(optimizer)
        optimizer.multiply_grads(1 / (sample_size or 1.0))
        gnorm = nn.utils.clip_grad_norm_(model.parameters(), config.clip_norm)
        gnorms.append(gnorm.cpu().item())

        scaler.step(optimizer)
        scaler.update()

        loss_print = accum_loss / sample_size
        stats["loss"].append(loss_print)
        progress.set_postfix(loss=loss_print)
    loss_print = np.mean(stats["loss"])
    print(f"training loss: {loss_print:.4f}")
    return stats


def decode(toks, dictionary):
    s = dictionary.string(
        toks.int().cpu(),
        "sentencepiece",
    )
    return s if s else "<unk>"


def inference_step(sample, model, sequence_generator,task):
    gen_out = sequence_generator.generate([model], sample)
    srcs = []
    hyps = []
    refs = []
    for i in range(len(gen_out)):
        srcs.append(decode(
            utils.strip_pad(sample["net_input"]["src_tokens"][i], task.source_dictionary.pad()),
            task.source_dictionary,
        ))
        hyps.append(decode(
            gen_out[i][0]["tokens"],
            task.target_dictionary,
        ))
        refs.append(decode(
            utils.strip_pad(sample["target"][i], task.target_dictionary.pad()),
            task.target_dictionary,
        ))
    return srcs, hyps, refs


def validate(model, task, criterion):
    itr = load_data_iterator(task, "valid", 1, config.max_tokens, config.num_workers).next_epoch_itr(shuffle=False)

    stats = {"loss": [], "bleu": 0, "srcs": [], "hyps": [], "refs": []}
    srcs = []
    hyps = []
    refs = []

    model.eval()
    progress = tqdm.tqdm(itr, desc=f"validation", leave=False)
    sequence_generator = task.build_generator([model], config)
    with torch.no_grad():
        for i, sample in enumerate(progress):
            sample = utils.move_to_cuda(sample, device=device)
            net_output = model.forward(**sample["net_input"])
            lprobs = F.log_softmax(net_output[0], -1)
            target = sample["target"]
            sample_size = sample["ntokens"]
            loss = criterion(lprobs.view(-1, lprobs.size(-1)), target.view(-1)) / sample_size
            progress.set_postfix(valid_loss=loss.item())
            stats["loss"].append(loss)

            s, h, r = inference_step(sample, model,sequence_generator,task)
            srcs.extend(s)
            hyps.extend(h)
            refs.extend(r)

    tok = 'zh' if task.cfg.target_lang == 'zh' else '13a'
    stats["loss"] = torch.stack(stats["loss"]).mean().item()
    stats["bleu"] = sacrebleu.corpus_bleu(hyps, [refs], tokenize=tok)
    stats["srcs"] = srcs
    stats["hyps"] = hyps
    stats["refs"] = refs
    showid = np.random.randint(len(hyps))
    print("example source: " + srcs[showid])
    print("example hypothesis: " + hyps[showid])
    print("example reference: " + refs[showid])
    print(f"validation loss:\t{stats['loss']:.4f}")
    print(stats["bleu"].format())
    return stats


def validate_and_save(model, task, criterion, optimizer, epoch, save=True):
    stats = validate(model, task, criterion)
    bleu = stats['bleu']
    loss = stats['loss']
    if save:
        savedir = Path(config.savedir).absolute()
        savedir.mkdir(parents=True, exist_ok=True)

        check = {
            "model": model.state_dict(),
            "stats": {"bleu": bleu.score, "loss": loss},
            "optim": {"step": optimizer._step}
        }
        torch.save(check, savedir / f"checkpoint{epoch}.pt")
        shutil.copy(savedir / f"checkpoint{epoch}.pt", savedir / f"checkpoint_last.pt")
        print(f"saved epoch checkpoint: {savedir}/checkpoint{epoch}.pt")

        with open(savedir / f"samples{epoch}.{config.source_lang}-{config.target_lang}.txt", "w") as f:
            for s, h in zip(stats["srcs"], stats["hyps"]):
                f.write(f"{s}\t{h}\n")

        if getattr(validate_and_save, "best_bleu", 0) < bleu.score:
            validate_and_save.best_bleu = bleu.score
            torch.save(check, savedir / f"checkpoint_best.pt")

        del_file = savedir / f"checkpoint{epoch - config.keep_last_epochs}.pt"
        if del_file.exists():
            del_file.unlink()

    return stats
