from typing import List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from clip import clip
from clip.model import CLIP

from ordinalclip.utils import get_logger

from .builder import PROMPT_LEARNERS

logger = get_logger(__name__)


@PROMPT_LEARNERS.register_module()
class PlainPromptLearner(nn.Module):
    clip_max_num_tokens = 77  # CLIP num_context_tokens = 77
    rank_tokens_position_candidates = {"tail", "middle", "front"}

    def __init__(
        self,
        clip_model: CLIP,
        num_ranks: int,
        num_tokens_per_rank: Union[int, List],
        num_context_tokens: int,
        rank_tokens_position: str = "tail",
        init_rank_path: Optional[str] = None,
        init_context: Optional[str] = None,
        rank_specific_context: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()

        if kwargs:
            logger.info(f"irrelevant kwargs: {kwargs}")

        dtype = clip_model.token_embedding.weight.dtype

        # context embeds
        context_embeds, _num_context_tokens = self.create_context_embeds(
            clip_model, num_ranks, num_context_tokens, init_context, rank_specific_context, logger, dtype
        )
        num_context_tokens = _num_context_tokens
        self.context_embeds = nn.Parameter(
            context_embeds
        )  # (num_context_tokens, embeds_dim) or (num_ranks, num_context_tokens, embeds_dim)

        # rank embeds
        if isinstance(num_tokens_per_rank, int):
            num_tokens_per_rank = [num_tokens_per_rank] * num_ranks
        rank_embeds, _num_tokens_per_rank = self.create_rank_embeds(
            clip_model, num_ranks, num_tokens_per_rank, init_rank_path, logger, dtype, num_context_tokens
        )
        num_tokens_per_rank = _num_tokens_per_rank
        self.rank_embeds = nn.Parameter(rank_embeds)  # (num_ranks, max_num_tokens_per_rank, embeddings_dim)
        assert len(rank_embeds) == num_ranks, f"len(rank_embeds) {len(rank_embeds)} == num_ranks {num_ranks}"

        # psudo sentence tokens
        psudo_sentence_tokens = self.create_psudo_sentence_tokens(
            num_tokens_per_rank, num_context_tokens, num_ranks
        )  # (num_ranks, clip_max_num_tokens)
        self.register_buffer("psudo_sentence_tokens", psudo_sentence_tokens, persistent=False)

        self.num_context_tokens = num_context_tokens
        self.num_tokens_per_rank = num_tokens_per_rank
        if rank_tokens_position not in self.rank_tokens_position_candidates:
            raise ValueError(f"Invalid rank_tokens_position: {rank_tokens_position}")
        self.rank_tokens_positon = rank_tokens_position
        self.num_ranks = num_ranks
        self.embeddings_dim = clip_model.token_embedding.embedding_dim

        self.create_sentence_embeds_template(clip_model, num_ranks, psudo_sentence_tokens)

    def forward(self):
        # context_embeds: (num_ranks, num_context_tokens, embeds_dim)
        # rank_embeds: (num_ranks, max_num_tokens_per_rank, embeddings_dim)
        context_embeds = self.context_embeds
        if context_embeds.dim() == 2:
            context_embeds = context_embeds[None].expand(self.num_ranks, *context_embeds.shape)

        # sentence_embeds: (num_ranks, self.clip_max_num_tokens, embeddings_dim)
        sentence_embeds = self.sentence_embeds.clone()
        if self.rank_tokens_positon == "tail":
            for i in range(self.num_ranks):
                _num_tokens_per_rank = self.num_tokens_per_rank[i]
                pure_sentence_length = self.num_context_tokens + _num_tokens_per_rank
                sentence_embeds[i, 1 : 1 + pure_sentence_length] = torch.cat(
                    [context_embeds[i], self.rank_embeds[i, :_num_tokens_per_rank]], dim=0
                )
        elif self.rank_tokens_positon == "front":
            for i in range(self.num_ranks):
                _num_tokens_per_rank = self.num_tokens_per_rank[i]
                pure_sentence_length = self.num_context_tokens + _num_tokens_per_rank
                sentence_embeds[i, 1 : 1 + pure_sentence_length] = torch.cat(
                    [self.rank_embeds[i, :_num_tokens_per_rank], context_embeds[i]], dim=0
                )
        elif self.rank_tokens_positon == "middle":
            for i in range(self.num_ranks):
                _num_tokens_per_rank = self.num_tokens_per_rank[i]
                pure_sentence_length = self.num_context_tokens + _num_tokens_per_rank
                _context_embeds = context_embeds[i]
                half_range = self.num_context_tokens // 2
                sentence_embeds[i, 1 : 1 + pure_sentence_length] = torch.cat(
                    [
                        _context_embeds[:half_range],
                        self.rank_embeds[i, :_num_tokens_per_rank],
                        _context_embeds[half_range:],
                    ],
                    dim=0,
                )

        return sentence_embeds

    def create_sentence_embeds_template(self, clip_model, num_ranks, psudo_sentence_tokens):
        with torch.no_grad():
            null_embed = clip_model.token_embedding(torch.LongTensor([0]))[0]
            sot_embed = clip_model.token_embedding(torch.LongTensor([49406]))[0]
            eot_embed = clip_model.token_embedding(torch.LongTensor([49407]))[0]
            full_stop_embed = clip_model.token_embedding(torch.LongTensor([269]))[0]

        sentence_embeds = null_embed[None, None].repeat(
            num_ranks, self.clip_max_num_tokens, 1
        )  # not the same null_embed!
        argmax_index = psudo_sentence_tokens.argmax(dim=-1)
        rank_index = torch.arange(num_ranks)

        sentence_embeds[:, 0, :] = sot_embed
        sentence_embeds[rank_index, argmax_index] = eot_embed
        sentence_embeds[rank_index, argmax_index - 1] = full_stop_embed

        self.register_buffer("sentence_embeds", sentence_embeds, persistent=False)

    def create_psudo_sentence_tokens(self, num_tokens_per_rank, num_context_tokens, num_ranks):
        psudo_sentence_tokens = torch.zeros(num_ranks, self.clip_max_num_tokens, dtype=torch.long)

        if isinstance(num_tokens_per_rank, List):
            assert num_ranks == len(num_tokens_per_rank)
            for i, _num_tokens_per_rank in enumerate(num_tokens_per_rank):
                # <sot>, <context_0>, ..., <context_N>, <rank_i>, <full_stop>, <eot>
                sentence_length = 1 + num_context_tokens + _num_tokens_per_rank + 1 + 1
                psudo_sentence_tokens[i, :sentence_length] = torch.arange(0, sentence_length, dtype=torch.long)
        else:
            # <sot>, <context_0>, ..., <context_N>, <rank_i>, <full_stop>, <eot>
            sentence_length = 1 + num_context_tokens + num_tokens_per_rank + 1 + 1
            psudo_sentence_tokens[:, :sentence_length] = torch.arange(0, sentence_length, dtype=torch.long)
        return psudo_sentence_tokens

    def create_rank_embeds(
        self, clip_model, num_ranks, num_tokens_per_rank, init_rank_path, logger, dtype, num_context_tokens
    ):
        if init_rank_path is not None:
            logger.info(f"load init rank from: {init_rank_path}.")

            rank_names = self.read_rank_file(init_rank_path, logger)
            if len(rank_names) != num_ranks:
                raise ValueError(
                    f"The length of rank_names is {len(rank_names)}, which is not equal to num_ranks {num_ranks}"
                )

            _rank_tokens = [clip._tokenizer.encode(rank_name) for rank_name in rank_names]
            _num_tokens_per_rank = [len(rank_token) for rank_token in _rank_tokens]
            logger.info(f"num_tokens_per_rank: {num_tokens_per_rank} -> {_num_tokens_per_rank}")
            num_tokens_per_rank = _num_tokens_per_rank
            max_num_tokens_per_rank = np.max(num_tokens_per_rank)

            rank_tokens = torch.zeros(len(_rank_tokens), max_num_tokens_per_rank, dtype=torch.long)
            for i, rank_token in enumerate(_rank_tokens):
                # 3 is <eot>, <sot>, and <full_stop>
                valid_length = self.clip_max_num_tokens - num_context_tokens - 3
                if len(rank_token) > valid_length:
                    rank_token = rank_token[:valid_length]
                    raise ValueError(f"rank tokens are too long: {rank_token}")
                rank_tokens[i, : len(rank_token)] = torch.LongTensor(rank_token)
            rank_embeds = clip_model.token_embedding(rank_tokens).type(dtype)
            rank_embeds = rank_embeds[:, :max_num_tokens_per_rank]

        else:
            logger.info(f"num rank: {num_ranks}")
            logger.info(f"num_tokens_per_rank: {num_tokens_per_rank}")
            embeddings_dim = clip_model.token_embedding.embedding_dim
            if isinstance(num_tokens_per_rank, List):
                max_num_tokens_per_rank = np.max(num_tokens_per_rank)
            else:
                max_num_tokens_per_rank = num_tokens_per_rank
            if self.clip_max_num_tokens < num_context_tokens + max_num_tokens_per_rank + 3:
                raise ValueError(f"rank tokens are too long: {rank_token}")
            rank_embeds = torch.empty((num_ranks, max_num_tokens_per_rank, embeddings_dim), dtype=dtype)
            nn.init.normal_(rank_embeds, std=0.02)

        return (rank_embeds, num_tokens_per_rank)

    def read_rank_file(self, init_rank_path, logger):
        rank_names = []
        with open(init_rank_path, "r") as f:
            for line in f.readlines():
                line = line.strip().replace("_", " ")
                rank_names.append(line)
        logger.info(f"num rank: {len(rank_names)}:\n\t{rank_names[:5]}\n\t{rank_names[-5:]}")
        return rank_names

    def create_context_embeds(
        self,
        clip_model,
        num_ranks: int,
        num_context_tokens: int,
        init_context: Optional[str],
        rank_specific_context: bool,
        logger,
        dtype,
    ):
        # context embeddings
        logger.info("init context token")
        if init_context is not None:
            init_context = init_context.replace("_", " ")
            logger.info(f"init context: {init_context}")

            prompt_tokens = clip.tokenize(init_context)
            prompt_tokens = prompt_tokens[0]  # (num_context_tokens=77)
            _num_context_tokens = torch.argmax(prompt_tokens).item() - 1
            logger.info(f"num_context_tokens: {num_context_tokens} -> {_num_context_tokens}")
            num_context_tokens = _num_context_tokens

            with torch.no_grad():
                context_embeds = clip_model.token_embedding(prompt_tokens).type(dtype)
            context_embeds = context_embeds[1 : 1 + num_context_tokens]

            logger.info(f"rank_specific_context: {rank_specific_context}")
            if rank_specific_context is True:
                context_embeds = context_embeds[None].repeat(num_ranks, 1, 1)
        else:
            embeds_dim = clip_model.token_embedding.embedding_dim
            init_context = " ".join(["X"] * num_context_tokens)
            logger.info(f"random context: {init_context}")
            logger.info(f"num context tokens: {num_context_tokens}")
            logger.info(f"rank_specific_context: {rank_specific_context}")

            if rank_specific_context is True:
                context_embeds = torch.empty((num_ranks, num_context_tokens, embeds_dim), dtype=dtype)
            else:
                context_embeds = torch.empty((num_context_tokens, embeds_dim), dtype=dtype)
            nn.init.normal_(context_embeds, std=0.02)

        return context_embeds, num_context_tokens
