import os.path as osp
from itertools import product

import pytest
import torch
from numpy import full
from omegaconf import DictConfig, OmegaConf

from ordinalclip.models.ordinalclip import load_clip_to_cpu
from ordinalclip.models.prompt_leaners import PROMPT_LEARNERS
from ordinalclip.models.prompt_leaners.rank_prompt_learner import RankPromptLearner

arg_names = "rank_tokens_position,init_context,init_rank_path,rank_specific_context,interpolation_type"
args = product(
    ["tail", "front", "middle"],
    ["i like lego", None],
    [osp.join(osp.dirname(__file__), "data", "rank_names.txt"), None],
    [True, False],
    ["linear", "inv_prop"],
)

# args = product(
#     ["tail"],
#     ["i like lego"],
#     [osp.join(osp.dirname(__file__), "data", "rank_names.txt")],
#     [True, False],
# )


# @pytest.fixture
def prepare_model():
    clip_model = load_clip_to_cpu(text_encoder_name="RN50", image_encoder_name="RN50", root=".cache/clip")
    with torch.no_grad():
        null_embed = clip_model.token_embedding(torch.LongTensor([0]))[0]
        sot_embed = clip_model.token_embedding(torch.LongTensor([49406]))[0]
        eot_embed = clip_model.token_embedding(torch.LongTensor([49407]))[0]
        full_stop_embed = clip_model.token_embedding(torch.LongTensor([269]))[0]
    return clip_model, null_embed, sot_embed, eot_embed, full_stop_embed


@pytest.mark.parametrize(arg_names, args)
def test_plain_prompt_learner(
    rank_tokens_position, init_context, init_rank_path, rank_specific_context, interpolation_type
):
    clip_model, null_embed, sot_embed, eot_embed, full_stop_embed = prepare_model()
    num_ranks = 10
    num_base_ranks = 5
    num_tokens_per_rank = 2
    num_context_tokens = 5
    cfg = OmegaConf.create(
        dict(
            prompt_learner_cfg=dict(
                # type="PlainPromptLearner",
                num_ranks=num_ranks,
                num_base_ranks=num_base_ranks,
                num_tokens_per_rank=num_tokens_per_rank,
                num_context_tokens=num_context_tokens,
                rank_tokens_position=rank_tokens_position,
                init_rank_path=init_rank_path,
                init_context=init_context,
                rank_specific_context=rank_specific_context,
                interpolation_type=interpolation_type,
            )
        )
    )

    def print_cfg(cfg, indent=0):
        for k, v in cfg.items():
            if isinstance(v, (dict, DictConfig)):
                print(f"{'    '*indent}{k}:")
                print_cfg(v, indent + 1)
            else:
                print(f"{'    '*indent}{k}: {v}")

    print_cfg(cfg)

    prompt_learner_cfg = OmegaConf.to_container(cfg.prompt_learner_cfg)
    prompt_learner_cfg.update(dict(clip_model=clip_model))
    model = RankPromptLearner(**prompt_learner_cfg)
    model = model.float()
    output = model()

    assert list(model.psudo_sentence_tokens.argmax(dim=-1)) == [
        i + 3 + model.num_context_tokens - 1 for i in model.num_tokens_per_rank
    ]
    assert torch.allclose(output[torch.arange(model.num_ranks), model.psudo_sentence_tokens.argmax(dim=-1)], eot_embed)
    assert torch.allclose(
        output[torch.arange(model.num_ranks), model.psudo_sentence_tokens.argmax(dim=-1) - 1], full_stop_embed
    )
    assert torch.allclose(output[:, 0], sot_embed)

    argmax = model.psudo_sentence_tokens.argmax(dim=-1)
    for i, _output in enumerate(output):
        assert torch.allclose(_output[argmax[i] + 1 :], null_embed)
    assert output.requires_grad == True

    if rank_specific_context is True:
        assert list(model.context_embeds.shape) == [num_ranks, model.num_context_tokens, model.embeddings_dim]
    else:
        assert list(model.context_embeds.shape) == [model.num_context_tokens, model.embeddings_dim]

    for k, v in model.named_parameters():
        print(f"name: {k}, shape: {v.shape}, requires_grad: {v.requires_grad}")
