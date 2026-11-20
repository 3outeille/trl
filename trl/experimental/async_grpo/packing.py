# Copyright 2020-2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import sys
from functools import cache
from itertools import chain
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AttentionInterface
from transformers.integrations.flash_attention import get_target_dtype
from transformers.masking_utils import AttentionMaskInterface


class TreeRow:
    """
    One DP rank's forward: its samples folded into a prefix trie, so a token shared by several samples is forwarded
    once.

    Multi-turn rollouts overlap heavily — a shared prompt, and after a fork a shared conversation — so the decoder's
    per-token work drops by the packing ratio (raw tokens ÷ unique tokens). Attention still shows each sample exactly
    what it saw: each segment of the trie attends to the K/V of its root-to-segment path (see [`tree_attention`]). With
    nothing shared, the trie is one chain per sample, which is plain padding-free packing.

    Any common prefix is shared: the rollouts of a prompt, and across prompts a common system prompt. The planner
    checks a sample against the row ([`~TreeRow.fits`]) before placing it, so the token budget bounds unique tokens,
    not raw ones, and it sends a sample to a row already holding its `group_id` ([`~TreeRow.groups`]), where its prompt
    is.

    A node is keyed on its token *and* the next one, so samples share a token only when they also agree on what follows
    it. The trie therefore forks one token before the samples diverge, and every node has a single next-token label: a
    branch costs one extra token per extra branch, and the loss can go through plain per-position labels. A sample's
    last token is keyed on `None`, which no other sample extends, so every sample ends on a leaf.
    """

    def __init__(self):
        self.samples: list[dict[str, Any]] = []
        self.sample_nodes: list[list[int]] = []  # sample -> trie node of each of its tokens
        self.tokens: list[int] = []  # node -> its token
        self.children: list[dict[tuple[int, int | None], int]] = []  # node -> {(token, next token): child node}
        self.roots: dict[tuple[int, int | None], int] = {}  # {(token, next token): root node}
        self.groups: set[Any] = set()
        self.trained = 0
        self.load = 0

    def fits(self, sample: dict[str, Any], budget: int) -> bool:
        """Whether the row stays within `budget` unique tokens and `budget` trained tokens with `sample` added."""
        input_ids = sample["input_ids"].tolist()
        children, shared = self.roots, 0
        for key in zip(input_ids, [*input_ids[1:], None], strict=True):
            if key not in children:
                break
            children = self.children[children[key]]
            shared += 1
        novel = len(input_ids) - shared
        return len(self.tokens) + novel <= budget and self.trained + int(sample["completion_mask"].sum()) <= budget

    def cost(self, sample: dict[str, Any]) -> int:
        """Load `add(sample)` would add to the row: `depth + 1` for each of its tokens not already in the trie."""
        input_ids = sample["input_ids"].tolist()
        children, shared = self.roots, 0
        for key in zip(input_ids, [*input_ids[1:], None], strict=True):
            if key not in children:
                break
            children = self.children[children[key]]
            shared += 1
        return sum(depth + 1 for depth in range(shared, len(input_ids)))

    def add(self, sample: dict[str, Any]) -> None:
        input_ids = sample["input_ids"].tolist()
        children = self.roots
        nodes = []
        for depth, key in enumerate(zip(input_ids, [*input_ids[1:], None], strict=True)):
            if key not in children:
                children[key] = len(self.tokens)
                self.tokens.append(key[0])
                self.children.append({})
                self.load += depth + 1
            nodes.append(children[key])
            children = self.children[nodes[-1]]
        self.samples.append(sample)
        self.sample_nodes.append(nodes)
        self.groups.add(sample["group_id"])
        self.trained += int(sample["completion_mask"].sum())

    def pack(self) -> dict[str, torch.Tensor]:
        """
        Lay the trie out in DFS pre-order, and select the loss terms taken over it.

        Unary chains collapse into contiguous segments. A segment ends at a branch or at a sample end, so every sample
        is exactly the concatenation of whole segments along one root-to-node path. A trained token is scored off its
        predecessor's hidden state, so each loss term points at the packed position of that predecessor's node.

        Returns:
            `dict[str, torch.Tensor]`, of `N` forwarded tokens, `S` segments and `M` trained tokens:
                - `input_ids` (`N`): the unique tokens.
                - `position_ids` (`N`): each token's position in its samples, which is its depth in the trie.
                - `shift_labels` (`N`): next token each position predicts, `-100` where it is not trained.
                - `sampling_mask` (`T`), `sampling_mask_counts` (`N`): candidate sets the sampler drew the next tokens
                  from, laid end to end in position order, and the size of each position's set; `T` is `0` without
                  top-k/top-p truncation. Ragged, because the sets average a few tokens but can reach the top-k cap.
                - `sampler_ids` (`N`, `K`), `sampler_log_probs` (`N`, `K`): the sampler's most likely next tokens at
                  the position and their log-probs, padded with `-1` and `-inf`; `K` is `0` unless requested.
                - `tree_kv_index` (`K`): packed position of every K/V row [`tree_attention`] gathers, the
                  root-to-segment paths of all segments laid end to end.
                - `tree_cu_seqlens_q` (`S + 1`), `tree_cu_seqlens_k` (`S + 1`): segment boundaries in the row, path
                  boundaries in `tree_kv_index`.
                - `tree_segment_parents` (`S`), `tree_segment_depths` (`S`): the segment tree, for the recurrent layers
                  patched by [`patch_tree_gdn`].
                - `tree_sizes` (`4 + depth`): longest segment, longest path, `K`, 64-token chunks, then the segments
                  per depth. Every length the kernels launch with, so a rank reads them in one go (see [`unpad_tree`]).
                - `token_index` (`M`): position whose prediction is each trained token. A shared position serves every
                  sample that shares it, so there are more loss terms than labelled positions.
                - `old_log_probs` (`M`), `advantages` (`M`): generator log-probability and advantage of each target.
                - `segment_id` (`M`): index of the sample each target came from.
        """
        input_ids, position_ids, offsets, paths, kv_index, cu_seqlens_k = [], [], [0], [], [], [0]
        parents, depths = [], []
        packed = [0] * len(self.tokens)
        stack = [(node, None, 0) for node in reversed(self.roots.values())]
        while stack:
            node, parent, depth = stack.pop()
            segment = len(paths)
            start = len(input_ids)
            parents.append(-1 if parent is None else parent)
            depths.append(0 if parent is None else depths[parent] + 1)
            while True:
                packed[node] = len(input_ids)
                input_ids.append(self.tokens[node])
                position_ids.append(depth)
                depth += 1
                if len(self.children[node]) != 1:
                    break
                node = next(iter(self.children[node].values()))
            offsets.append(len(input_ids))
            paths.append((paths[parent] if parent is not None else []) + list(range(start, len(input_ids))))
            kv_index.extend(paths[-1])
            cu_seqlens_k.append(len(kv_index))
            stack.extend((child, segment, depth) for child in reversed(self.children[node].values()))

        lengths = [end - start for start, end in zip(offsets[:-1], offsets[1:], strict=True)]
        shift_labels = [-100] * len(input_ids)
        sampling_mask = [[] for _ in input_ids]
        num_logprobs = max(sample["sampler_ids"].shape[1] for sample in self.samples)
        sampler_ids = np.full((len(input_ids), num_logprobs), -1, dtype=np.int64)
        sampler_log_probs = np.full((len(input_ids), num_logprobs), -math.inf, dtype=np.float32)
        token_index, old_log_probs, advantages, segment_id = [], [], [], []
        for i, (sample, nodes) in enumerate(zip(self.samples, self.sample_nodes, strict=True)):
            ids, mask, logps = (sample[key].tolist() for key in ("input_ids", "completion_mask", "old_log_probs"))
            first = len(token_index)
            for p in range(1, len(nodes)):
                if mask[p]:
                    shift_labels[packed[nodes[p - 1]]] = ids[p]
                    sampling_mask[packed[nodes[p - 1]]] = sample["sampling_mask"][p]
                    token_index.append(packed[nodes[p - 1]])
                    old_log_probs.append(logps[p])
                    advantages.append(sample["advantage"])
                    segment_id.append(i)
            trained = np.flatnonzero(sample["completion_mask"][1:]) + 1
            sampler_ids[token_index[first:], : sample["sampler_ids"].shape[1]] = sample["sampler_ids"][trained]
            sampler_log_probs[token_index[first:], : sample["sampler_ids"].shape[1]] = sample["sampler_log_probs"][
                trained
            ]
        return {
            "input_ids": torch.tensor(input_ids),
            "position_ids": torch.tensor(position_ids),
            "shift_labels": torch.tensor(shift_labels),
            "sampling_mask": torch.tensor(list(chain.from_iterable(sampling_mask)), dtype=torch.long),
            "sampling_mask_counts": torch.tensor(list(map(len, sampling_mask))),
            "sampler_ids": torch.from_numpy(sampler_ids),
            "sampler_log_probs": torch.from_numpy(sampler_log_probs),
            "tree_kv_index": torch.tensor(kv_index, dtype=torch.long),
            "tree_cu_seqlens_q": torch.tensor(offsets, dtype=torch.int32),
            "tree_cu_seqlens_k": torch.tensor(cu_seqlens_k, dtype=torch.int32),
            "tree_segment_parents": torch.tensor(parents, dtype=torch.int32),
            "tree_segment_depths": torch.tensor(depths, dtype=torch.int32),
            "tree_sizes": torch.tensor(
                [max(lengths), max(map(len, paths)), len(kv_index), sum(-(-length // 64) for length in lengths)]
                + [depths.count(depth) for depth in range(max(depths) + 1)]
            ),
            "token_index": torch.tensor(token_index, dtype=torch.long),
            "old_log_probs": torch.tensor(old_log_probs, dtype=torch.float32),
            "advantages": torch.tensor(advantages, dtype=torch.float32),
            "segment_id": torch.tensor(segment_id, dtype=torch.long),
        }


def unpad_tree(inputs: dict[str, torch.Tensor]) -> dict[str, Any]:
    """
    This rank's tree from [`TreeRow.pack`], as the model's forward kwargs: the collator's inter-rank padding sliced
    off, and the sizes the kernels launch with read to the host in a single transfer.
    """
    row = {key: value[0] for key, value in inputs.items() if key.startswith("tree_")}
    max_seqlen_q, max_seqlen_k, n_kv, n_chunks, *level_sizes = [n for n in row["tree_sizes"].tolist() if n > 0]
    n_segments = sum(level_sizes)
    return {
        "tree_kv_index": row["tree_kv_index"][:n_kv],
        "tree_cu_seqlens_q": row["tree_cu_seqlens_q"][: n_segments + 1],
        "tree_cu_seqlens_k": row["tree_cu_seqlens_k"][: n_segments + 1],
        "tree_max_seqlen_q": max_seqlen_q,
        "tree_max_seqlen_k": max_seqlen_k,
        "tree_segment_parents": row["tree_segment_parents"][:n_segments],
        "tree_segment_depths": row["tree_segment_depths"][:n_segments],
        "tree_level_sizes": level_sizes,
        "tree_n_chunks": n_chunks,
    }


_TREE_MAPS: dict[tuple[int, ...], tuple[tuple[torch.Tensor, ...], dict[str, torch.Tensor]]] = {}


def tree_maps(tree: dict[str, Any]) -> dict[str, torch.Tensor]:
    """
    A tree-packed row's per-position maps, shared by every layer of a forward and built once for all of them.

    The maps are cached on the tree tensors themselves, so a layer that sees the same row again (the recompute of a
    gradient checkpointed layer) reuses them. The cached tensors are held, which is what keeps their pointers unique.

    Args:
        tree (`dict[str, Any]`): the tree kwargs of a forward, as [`unpad_tree`] returns them.

    Returns:
        `dict[str, torch.Tensor]`, for the row's `N` positions:
            - `prev` (`N`): the position before each one on its own path, `-1` at a path start.
            - `depth` (`N`): its depth on that path, which is the row's `position_ids`.
            - `visible` (`N`, `N`): whether a position sees another, i.e. the other is on its path and comes first.
    """
    cu_seqlens_q, cu_seqlens_k, kv_index = (
        tree["tree_cu_seqlens_q"],
        tree["tree_cu_seqlens_k"],
        tree["tree_kv_index"],
    )
    key = (
        cu_seqlens_q.data_ptr(),
        cu_seqlens_k.data_ptr(),
        kv_index.data_ptr(),
        cu_seqlens_q.numel(),
        kv_index.numel(),
    )
    if key in _TREE_MAPS:
        return _TREE_MAPS[key][1]

    device = kv_index.device
    parents = tree["tree_segment_parents"].long()
    cu_q, cu_k = cu_seqlens_q.tolist(), cu_seqlens_k.tolist()
    # Each segment's keys repeat its whole path, so the row has `cu_q[-1]` positions and `kv_index` is longer
    n = cu_q[-1]
    segment_parents = parents.tolist()
    lengths = [end - start for start, end in zip(cu_q[:-1], cu_q[1:], strict=True)]
    # Parents are laid out before their children, so one pass gives every segment the depth its first token sits at
    depths = []
    for parent in segment_parents:
        depths.append(0 if parent < 0 else depths[parent] + lengths[parent])

    prev = torch.arange(n, device=device) - 1
    # A segment starts after its parent's last token, which is the parent segment's last position; a root starts a path
    parent_last = torch.where(parents >= 0, cu_seqlens_q.long()[(parents + 1).clamp_min(0)] - 1, -1)
    prev[cu_seqlens_q[:-1].long()] = parent_last
    segments = torch.repeat_interleave(torch.arange(len(lengths), device=device), torch.tensor(lengths, device=device))
    depth = (
        torch.arange(n, device=device)
        - torch.tensor(cu_q[:-1], device=device)[segments]
        + torch.tensor(depths, device=device)[segments]
    )
    visible = torch.zeros(n, n, dtype=torch.bool, device=device)
    for start, end, path_start, path_end in zip(cu_q[:-1], cu_q[1:], cu_k[:-1], cu_k[1:], strict=True):
        # A path is laid out before the segment that follows it, so its positions are ascending
        path = kv_index[path_start:path_end].long()
        visible[start:end, path] = path.unsqueeze(0) <= torch.arange(start, end, device=device).unsqueeze(1)

    entry = ((cu_seqlens_q, cu_seqlens_k, kv_index), {"prev": prev, "depth": depth, "visible": visible})
    while len(_TREE_MAPS) >= 2:
        _TREE_MAPS.pop(next(iter(_TREE_MAPS)))
    _TREE_MAPS[key] = entry
    return entry[1]


def tree_ngram_shift_right(
    token_ids: torch.Tensor, shift: int, eos_token_id: int, tree: dict[str, Any]
) -> torch.Tensor:
    """
    The n-gram embedding's shift over a tree-packed row: each position reads the `shift`-th token before it on its own
    path, and EOS wherever that walk leaves the path or crosses one.

    `token_ids` also holds the EOS context columns the caller prepends to the row, which stay EOS here.
    """
    if shift == 0:
        return token_ids
    prev = tree_maps(tree)["prev"]
    n = prev.numel()
    tokens = token_ids[:, -n:]
    eos = tokens == eos_token_id
    index = torch.arange(n, device=prev.device)
    valid = torch.ones(tokens.shape, dtype=torch.bool, device=prev.device)
    for _ in range(shift):
        index = torch.where(index >= 0, prev[index.clamp_min(0)], index)
        valid &= (index >= 0) & ~eos.gather(1, index.clamp_min(0).expand(tokens.shape[0], -1))
    shifted = tokens.gather(1, index.clamp_min(0).expand(tokens.shape[0], -1))
    shifted = torch.where(valid, shifted, torch.full_like(shifted, eos_token_id))
    output = token_ids.new_full(token_ids.shape, eos_token_id)
    output[:, -n:] = shifted
    return output


def tree_ple_short_conv(hidden_states: torch.Tensor, conv1d: torch.nn.Conv1d, tree: dict[str, Any]) -> torch.Tensor:
    """
    The per-layer embedding's dilated depthwise convolution over a tree-packed row: each tap reads the position its
    dilation puts it at on the same path, and zeros where that leaves the path.
    """
    prev = tree_maps(tree)["prev"]
    n, device = prev.numel(), prev.device
    weight, bias = conv1d.weight, conv1d.bias
    states = hidden_states[..., -n:]
    output = torch.zeros_like(states)
    taps = weight.shape[-1]
    index = torch.arange(n, device=device)
    alive = torch.ones(n, dtype=torch.bool, device=device)
    for tap in reversed(range(taps)):
        for _ in range(0 if tap == taps - 1 else conv1d.dilation[0]):
            alive &= index >= 0
            index = torch.where(index >= 0, prev[index.clamp_min(0)], index)
        alive &= index >= 0
        source = index.clamp_min(0).expand(states.shape[0], states.shape[1], -1)
        output.addcmul_(states.gather(2, source).masked_fill_(~alive, 0), weight[:, 0, tap].view(1, -1, 1))
    if bias is not None:
        output.add_(bias.view(1, -1, 1))
    return F.silu(output)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the first half of the head dimension onto the second, as the model's rotary embedding does."""
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate `x` with already gathered `cos` and `sin`, as the model's indexer does for its pooled blocks."""
    rotary_dim = cos.shape[-1]
    rotated = x[..., :rotary_dim] * cos + _rotate_half(x[..., :rotary_dim]) * sin
    return torch.cat([rotated, x[..., rotary_dim:]], dim=-1)


def tree_qsa_select_tokens(
    q: torch.Tensor,
    raw_keys: torch.Tensor,
    full_cos: torch.Tensor,
    full_sin: torch.Tensor,
    k_layernorm: torch.nn.Module,
    block_topk: int,
    compress_ratio: int,
    tree: dict[str, Any],
) -> torch.Tensor:
    """
    The sparse-attention indexer's selection over a tree-packed row, as a bool mask of shape `(1, 1, q, kv)`.

    A block is `compress_ratio` consecutive depths on one path, keyed by its last token. The query scores the blocks
    whose key is on its own path, keeps the best `block_topk` of them, and adds the keys after its last complete block.
    """
    maps = tree_maps(tree)
    prev, depth, visible = maps["prev"], maps["depth"], maps["visible"]
    device, n = prev.device, prev.numel()
    head_dim = q.shape[-1]

    block_last = torch.nonzero(depth % compress_ratio == compress_ratio - 1).flatten()
    members = [block_last]
    for _ in range(compress_ratio - 1):
        members.insert(0, prev[members[0].clamp_min(0)])
    members = torch.stack(members, dim=1)  # (blocks, compress_ratio), by increasing depth
    pooled = raw_keys[0].index_select(0, members.flatten()).view(len(block_last), compress_ratio, -1)
    pooled = k_layernorm(pooled.float().mean(dim=1).to(raw_keys.dtype))
    block_states = _apply_rope(
        pooled.unsqueeze(1),
        full_cos[0].index_select(0, members[:, 0]).unsqueeze(1),
        full_sin[0].index_select(0, members[:, 0]).unsqueeze(1),
    )
    block_states = block_states.squeeze(1).float()

    queries = q[0].float()
    num_blocks, num_selected = len(block_last), min(block_topk, len(block_last))
    selected = torch.zeros(n, n, dtype=torch.uint8, device=device)
    flat = selected.view(-1)
    # Bound the score tensor like the causal selection does, so a long row does not materialise all of it at once
    queries_per_chunk = max(1, 2**24 // max(num_blocks * queries.shape[1], 1))
    for start in range(0, n, queries_per_chunk):
        stop = min(start + queries_per_chunk, n)
        rows = torch.arange(start, stop, device=device)
        scores = torch.einsum("qhd,bd->qbh", queries[start:stop], block_states)
        scores = scores.relu().sum(dim=-1) / math.sqrt(head_dim)
        seen = visible[start:stop].index_select(1, block_last)
        top = scores.masked_fill(~seen, float("-inf")).topk(num_selected, dim=-1)
        chosen = torch.where(top.values > float("-inf"), top.indices, -1)
        kept = chosen >= 0
        ones = torch.ones(chosen.shape, dtype=torch.uint8, device=device)
        for member in range(compress_ratio):
            source = members[:, member].index_select(0, chosen.flatten().clamp_min(0)).view(chosen.shape)
            flat.index_add_(0, (source + rows.view(-1, 1) * n)[kept], ones[kept])
        cutoff = (depth[rows] + 1) // compress_ratio * compress_ratio
        tail = visible[start:stop] & (depth.view(1, -1) >= cutoff.view(-1, 1))
        selected[start:stop].bitwise_or_(tail.to(torch.uint8))

    return (selected.to(torch.bool) & visible).view(1, 1, n, n)


_QWEN4_PATCHED = set()


def patch_tree_qwen4_exp(model) -> bool:
    """
    Make the model's n-gram embedding, per-layer short convolution and sparse-attention indexer follow the tree of a
    tree-packed forward.

    Qwen4-Exp runs those three token-mixing steps through module-level functions that the layers call with the
    forward's keyword arguments. All three are wrapped: given the tree of [`TreeRow.pack`] they follow each sample's
    own path, otherwise they run unchanged.

    Returns:
        `bool`: whether the model's modeling module has those functions.
    """
    modeling = sys.modules[type(model).forward.__module__]
    names = ("ngram_shift_right", "ple_short_conv", "qsa_select_tokens")
    originals = [vars(modeling).get(name) for name in names]
    if any(original is None for original in originals):
        return False
    if modeling in _QWEN4_PATCHED:
        return True
    ngram_shift_right, ple_short_conv, qsa_select_tokens = originals

    def ngram_shift_right_fn(token_ids, shift, eos_token_id, **kwargs):
        if "tree_segment_parents" not in kwargs:
            return ngram_shift_right(token_ids, shift, eos_token_id, **kwargs)
        return tree_ngram_shift_right(token_ids, shift, eos_token_id, kwargs)

    def ple_short_conv_fn(hidden_states, conv1d, **kwargs):
        if "tree_segment_parents" not in kwargs:
            return ple_short_conv(hidden_states, conv1d, **kwargs)
        return tree_ple_short_conv(hidden_states, conv1d, kwargs)

    def qsa_select_tokens_fn(
        q,
        raw_keys,
        full_cos,
        full_sin,
        attention_mask,
        k_layernorm,
        block_topk,
        compress_ratio,
        token_budget,
        **kwargs,
    ):
        if "tree_segment_parents" not in kwargs:
            return qsa_select_tokens(
                q, raw_keys, full_cos, full_sin, attention_mask, k_layernorm, block_topk, compress_ratio, token_budget,
                **kwargs,
            )
        return tree_qsa_select_tokens(q, raw_keys, full_cos, full_sin, k_layernorm, block_topk, compress_ratio, kwargs)

    modeling.ngram_shift_right, modeling.ple_short_conv, modeling.qsa_select_tokens = (
        ngram_shift_right_fn,
        ple_short_conv_fn,
        qsa_select_tokens_fn,
    )
    _QWEN4_PATCHED.add(modeling)
    return True


TREE_ATTENTION = "tree_fa3"


QWEN4_EXP = "qwen4_exp"


def tree_attn_implementation(config) -> str:
    """
    The attention implementation tree packing asks a model for: `sdpa` for Qwen4-Exp, whose sparse attention selects
    each query's keys itself and therefore needs an explicit mask, tree attention for every other model.
    """
    return "sdpa" if config.model_type == QWEN4_EXP else TREE_ATTENTION


@cache
def _flash_attn():
    from kernels import get_kernel

    if torch.version.hip:
        return get_kernel("kernels-community/aiter-flash-attn", version=2)
    # FlashAttention-3 has no kernels for Blackwell GPUs (compute capability 10 and above), FlashAttention-4 does
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 10:
        return get_kernel("kernels-community/flash-attn4", version=1)
    return get_kernel("kernels-community/flash-attn3", version=1)


def tree_attention(
    module,
    query,
    key,
    value,
    attention_mask,
    *,
    tree_kv_index,
    tree_cu_seqlens_q,
    tree_cu_seqlens_k,
    tree_max_seqlen_q,
    tree_max_seqlen_k,
    scaling=None,
    sliding_window=None,
    softcap=None,
    **kwargs,
):
    dtype = get_target_dtype(query, module) or query.dtype
    query, key, value = (x.transpose(1, 2)[0].to(dtype) for x in (query, key, value))
    out = _flash_attn().flash_attn_varlen_func(
        query,
        key.index_select(0, tree_kv_index),
        value.index_select(0, tree_kv_index),
        cu_seqlens_q=tree_cu_seqlens_q,
        cu_seqlens_k=tree_cu_seqlens_k,
        max_seqlen_q=tree_max_seqlen_q,
        max_seqlen_k=tree_max_seqlen_k,
        softmax_scale=scaling,
        causal=True,
        window_size=(sliding_window - 1, 0) if sliding_window else (-1, -1),
        **({"softcap": softcap} if softcap else {}),
    )
    out = out[0] if isinstance(out, tuple) else out
    return out.unsqueeze(0), None


def register_tree_attention() -> None:
    AttentionInterface.register(TREE_ATTENTION, tree_attention)
    AttentionMaskInterface.register(TREE_ATTENTION, lambda **kwargs: None)


_PATCHED = set()


def patch_tree_gdn(model) -> bool:
    """
    Make the model's gated DeltaNet layers follow the tree of a tree-packed forward.

    Transformers' hybrids (Qwen3.5, Qwen3-Next, OLMo-Hybrid, ...) run these layers through two module-level functions,
    `causal_conv1d_fn` and `torch_chunk_gated_delta_rule`, and pass the forward's keyword arguments down to both. Both
    are wrapped: given the tree of [`~packing.TreeRow.pack`] they follow it, otherwise they run unchanged.

    Returns:
        `bool`: whether the model's modeling module has those functions.
    """
    modeling = sys.modules[type(model).forward.__module__]
    conv, chunk = vars(modeling).get("causal_conv1d_fn"), vars(modeling).get("torch_chunk_gated_delta_rule")
    if conv is None or chunk is None:
        return False
    if modeling in _PATCHED:
        return True
    from .tree_gated_delta_rule import tree_causal_conv1d, tree_chunk_gated_delta_rule

    def causal_conv1d_fn(hidden_states, weight, bias=None, activation=None, **kwargs):
        if "tree_segment_parents" not in kwargs:
            return conv(hidden_states, weight, bias, activation=activation, **kwargs)
        return tree_causal_conv1d(hidden_states, weight, bias, activation, kwargs, conv)

    def torch_chunk_gated_delta_rule(query, key, value, g, beta, use_qk_l2norm_in_kernel=False, **kwargs):
        if "tree_segment_parents" not in kwargs:
            return chunk(query, key, value, g=g, beta=beta, use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel, **kwargs)
        return tree_chunk_gated_delta_rule(query, key, value, g, beta, kwargs, use_qk_l2norm_in_kernel), None

    modeling.causal_conv1d_fn, modeling.torch_chunk_gated_delta_rule = causal_conv1d_fn, torch_chunk_gated_delta_rule
    _PATCHED.add(modeling)
    return True
