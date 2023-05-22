from argparse import Namespace

from fairseq import utils
from fairseq.tasks.translation import TranslationConfig, TranslationTask


def load_data_iterator(task, split, epoch=1, max_tokens=4000, num_workers=1, cached=True):
    batch_iterator = task.get_batch_iterator(
        dataset=task.dataset(split),
        max_tokens=max_tokens,
        max_sentences=None,
        max_positions=utils.resolve_max_positions(
            task.max_positions(),
            max_tokens,
        ),
        ignore_invalid_inputs=True,
        num_workers=num_workers,
        epoch=epoch,
        disable_iterator_cache=not cached,
    )
    return batch_iterator


def get_config():
    config = Namespace(
        datadir="./data/data-bin",
        savedir="./",
        source_lang="en",
        target_lang="zh",
        num_workers=4,
        max_tokens=8192,
        accum_steps=2,
        lr_factor=2.,
        lr_warmup=4000,
        clip_norm=1.0,
        max_epoch=30,
        start_epoch=1,
        beam=5,
        max_len_a=1.2,
        max_len_b=10,
        post_process="sentencepiece",
        keep_last_epochs=5,
        resume=None,
    )
    return config

def get_task():
    datadir = "./data/data-bin"
    source_lang = "en"
    target_lang = "zh"
    task_cfg = TranslationConfig(
        data=datadir,
        source_lang=source_lang,
        target_lang=target_lang,
        train_subset="train",
        required_seq_len_multiple=8,
        dataset_impl="mmap",
        upsample_primary=1,
    )
    task = TranslationTask.setup_task(task_cfg)
    return task