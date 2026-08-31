import os

import torch


UNIGRAM1000_LIST = (
    ['<blank>']
    + [_.split()[0] for _ in open(os.path.join(os.path.dirname(__file__), "labels", "unigram1000_units.txt")).read().splitlines()]
    + ['<eos>']
)


def ids_to_str(token_ids, char_list):
    tokenid_as_list = list(map(int, token_ids))
    token_as_list = [char_list[idx] for idx in tokenid_as_list]
    return "".join(token_as_list).replace("<space>", " ")


def set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val


def strip_compile_prefix(state_dict):
    """
    Strip torch.compile's '_orig_mod.' prefix from checkpoint keys.

    torch.compile(model) wraps model in an OptimizedModule that stores the
    real parameters under a '_orig_mod' submodule, so a checkpoint saved from
    a compiled model has every key prefixed with '_orig_mod.'. Loading such a
    checkpoint into an UNcompiled model (compile_model=False) then silently
    matches zero keys under strict=False, leaving the model at random init
    with no error. Stripping the prefix here makes checkpoint loading correct
    regardless of whether the model being loaded into is compiled or not.
    """
    prefix = "_orig_mod."
    return {
        (k[len(prefix):] if k.startswith(prefix) else k): v
        for k, v in state_dict.items()
    }
