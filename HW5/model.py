from argparse import Namespace

from fairseq.models import (
    FairseqEncoder,
    FairseqIncrementalDecoder,
    FairseqEncoderDecoderModel
)
from fairseq.modules import MultiheadAttention
from fairseq.models.transformer import (
    TransformerEncoder,
    TransformerDecoder,
)
from data_loader import  get_task
import torch.nn as nn
from fairseq.models.transformer import base_architecture
arch_args = Namespace(
    encoder_embed_dim=256,
    encoder_ffn_embed_dim=1024,
    encoder_layers=4,
    decoder_embed_dim=256,
    decoder_ffn_embed_dim=1024,
    decoder_layers=4,
    share_decoder_input_output_embed=True,
    dropout=0.15,
)


class Seq2Seq(FairseqEncoderDecoderModel):
    def __init__(self, args, encoder, decoder):
        super().__init__(encoder, decoder)
        self.args = args

    def forward(self, src_tokens, src_lengths, prev_output_tokens, return_all_hiddens: bool = True):
        encoder_out = self.encoder(src_tokens, src_lengths=src_lengths, return_all_hiddens=return_all_hiddens)
        logits, extra = self.decoder(prev_output_tokens, encoder_out=encoder_out, src_lengths=src_lengths,
                                     return_all_hiddens=return_all_hiddens)
        return logits, extra


def build_model(args, task):
    src_dict, tgt_dict = task.source_dictionary, task.target_dictionary
    encoder_embed_tokens = nn.Embedding(len(src_dict), args.encoder_embed_dim, src_dict.pad())
    decoder_embed_tokens = nn.Embedding(len(tgt_dict), args.decoder_embed_dim, tgt_dict.pad())
    encoder = TransformerEncoder(args, src_dict, encoder_embed_tokens)
    decoder = TransformerDecoder(args, tgt_dict, decoder_embed_tokens)
    model = Seq2Seq(args, encoder, decoder)

    def init_params(module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        if isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        if isinstance(module, MultiheadAttention):
            module.q_proj.weight.data.normal_(mean=0.0, std=0.02)
            module.k_proj.weight.data.normal_(mean=0.0, std=0.02)
            module.v_proj.weight.data.normal_(mean=0.0, std=0.02)
        if isinstance(module, nn.RNNBase):
            for name, param in module.named_parameters():
                if "weight" in name or "bias" in name:
                    param.data.uniform_(-0.1, 0.1)

    model.apply(init_params)
    return model


def get_model():
    add_transformer_args(arch_args)
    task = get_task()
    model = build_model(arch_args, task)
    return model


def add_transformer_args(args):
    args.encoder_attention_heads = 4
    args.encoder_normalize_before = True

    args.decoder_attention_heads = 4
    args.decoder_normalize_before = True

    args.activation_fn = "relu"
    args.max_source_positions = 1024
    args.max_target_positions = 1024
    base_architecture(arch_args)