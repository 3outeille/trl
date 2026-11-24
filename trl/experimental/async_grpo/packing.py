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


TREE_ATTENTION = "tree_fa3"


# (the last row's `tree_cu_seqlens_q`, its maps): every layer asks for the maps of the same row, and building them
# costs a host sync
_MAPS = None


def tree_maps(tree):
    """
    The row's maps, by row position:
    - `prev`: the position just before it on its own path, or `-1` at the start of a path.
    - `visible`: `visible[i, j]` is whether `j` is on the path of `i`, at or before `i`.
    - `depth`: its position in its own sample.

    Example: segments `[0, 1]` (a root), `[2, 3]` and `[4]` (both children of the first) give `prev = [-1, 0, 1, 2, 1]`.
    Position 4 sees positions 0, 1 and 4, so its depth is 2.
    """
    global _MAPS
    if _MAPS is not None and _MAPS[0] is tree["tree_cu_seqlens_q"]:
        return _MAPS[1]
    cu_seqlens_q = tree["tree_cu_seqlens_q"].long()
    starts, ends = cu_seqlens_q[:-1], cu_seqlens_q[1:]  # each segment's first position, and one past its last
    parents = tree["tree_segment_parents"].long()  # each segment's parent segment, -1 at a root

    prev = torch.arange(int(ends[-1]), device=starts.device) - 1  # inside a segment: the position to the left
    prev[starts] = torch.where(parents >= 0, ends[parents] - 1, -1)  # at a segment start: the parent's last position

    # `tree_kv_index` lists each segment's path, in ascending row positions, and the segment ends that path
    visible = torch.zeros(len(prev), len(prev), dtype=torch.bool, device=prev.device)
    path_bounds = tree["tree_cu_seqlens_k"].tolist()
    for start, end, path_start, path_end in zip(
        starts.tolist(), ends.tolist(), path_bounds[:-1], path_bounds[1:], strict=True
    ):
        path = tree["tree_kv_index"][path_start:path_end].long()
        visible[start:end, path] = path.unsqueeze(0) <= torch.arange(start, end, device=prev.device).unsqueeze(1)
    depth = visible.sum(dim=1) - 1  # a position sees itself and the `depth` positions before it on its path

    _MAPS = (tree["tree_cu_seqlens_q"], {"prev": prev, "visible": visible, "depth": depth})
    return _MAPS[1]


def _step_back(prev: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """Each position moved one step back along its path; `-1` (already past the start) stays `-1`."""
    return torch.where(positions >= 0, prev[positions.clamp_min(0)], -1)


def tree_ngram_shift_right(token_ids: torch.Tensor, shift: int, eos_token_id: int, tree) -> torch.Tensor:
    """
    The token `shift` steps back on each position's own path, or EOS where the walk passes the start of the path or
    meets an EOS: the model's `_shift_right_ignore_eos`, along paths instead of along the row.

    `token_ids` is `(batch, context + n)`: the model puts `context` EOS columns before the row's `n` tokens.
    """
    if shift == 0:
        return token_ids
    prev = tree_maps(tree)["prev"]
    n = prev.numel()
    tokens = token_ids[:, -n:]  # (batch, n)
    is_eos = tokens == eos_token_id

    source = torch.arange(n, device=prev.device)  # where each position's walk is
    valid = torch.ones_like(is_eos)
    for _ in range(shift):
        source = _step_back(prev, source)
        valid &= (source >= 0) & ~is_eos[:, source.clamp_min(0)]
    shifted = torch.where(valid, tokens[:, source.clamp_min(0)], eos_token_id)

    output = torch.full_like(token_ids, eos_token_id)  # the context columns stay EOS
    output[:, -n:] = shifted
    return output


def tree_ple_short_conv(hidden_states: torch.Tensor, conv1d: torch.nn.Conv1d, tree) -> torch.Tensor:
    """
    `silu(conv1d(hidden_states))` for the model's dilated depthwise `conv1d`, along paths instead of along the row.

    With 3 taps and dilation 3, a position reads itself, the position 3 steps back on its path and the one 6 steps
    back. A step past the start of the path reads zeros, as the left padding does for one plain sequence.
    `hidden_states` is `(batch, channels, n)`.
    """
    prev = tree_maps(tree)["prev"]
    weight = conv1d.weight[:, 0, :]  # (channels, taps): depthwise, so one filter per channel

    output = torch.zeros_like(hidden_states)
    source = torch.arange(prev.numel(), device=prev.device)  # the last tap reads the position itself
    for tap in reversed(range(weight.shape[-1])):
        tap_input = hidden_states[..., source.clamp_min(0)].masked_fill(source < 0, 0)
        output += tap_input * weight[:, tap].view(1, -1, 1)
        for _ in range(conv1d.dilation[0]):  # the tap before reads `dilation` steps further back
            source = _step_back(prev, source)
    if conv1d.bias is not None:
        output += conv1d.bias.view(1, -1, 1)
    return F.silu(output)


def tree_qsa_select_tokens(q, raw_keys, full_cos, full_sin, k_layernorm, block_topk, compress_ratio, tree):
    """
    The model's `_select_tokens` along paths, as a bool mask `(1, 1, n, n)`: each query keeps the `block_topk` best
    blocks of `compress_ratio` consecutive depths on its path, and the tail after its last full block. A block is named
    by its last position, which fixes its path, so a block shared by several paths is pooled and scored once.
    """
    # Imported here: only Qwen4-Exp calls this, and not every transformers version has its modeling file
    from transformers.models.qwen4_exp.modeling_qwen4_exp import apply_rotary_pos_emb

    maps = tree_maps(tree)
    prev, depth, visible = maps["prev"], maps["depth"], maps["visible"]
    n = prev.numel()

    # The blocks, as (num_blocks, compress_ratio) row positions by increasing depth. A block's last position is at
    # depth >= compress_ratio - 1, so walking back from it never leaves the path.
    block_last = torch.nonzero(depth % compress_ratio == compress_ratio - 1).flatten()
    members = [block_last]
    for _ in range(compress_ratio - 1):
        members.insert(0, prev[members[0]])
    members = torch.stack(members, dim=1)

    # One key per block, as the model builds it: the mean of its raw keys, normalized, rotated at its first position
    pooled = raw_keys[0][members].float().mean(dim=1).to(raw_keys.dtype)  # (num_blocks, head_dim)
    block_keys = apply_rotary_pos_emb(
        k_layernorm(pooled).unsqueeze(1), cos=full_cos[0][members[:, 0]], sin=full_sin[0][members[:, 0]]
    ).squeeze(1)

    queries = q[0].float()  # (n, heads, head_dim)
    num_picked = min(block_topk, len(block_last))
    selected = torch.zeros(n, n + 1, dtype=torch.bool, device=prev.device)  # column n is a bin, dropped at the end
    chunk = max(1, 2**24 // max(len(block_last) * queries.shape[1], 1))  # bounds the (chunk, blocks, heads) scores
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        scores = torch.einsum("qhd,bd->qbh", queries[start:stop], block_keys.float())
        scores = scores.relu().sum(dim=-1) / math.sqrt(q.shape[-1])  # (chunk, num_blocks)
        scores = scores.masked_fill(~visible[start:stop, block_last], float("-inf"))  # only blocks on its path

        top = scores.topk(num_picked, dim=-1)
        picked = members[top.indices]  # (chunk, num_picked, compress_ratio)
        # A query that sees fewer than `num_picked` blocks gets -inf picks: send them to the bin column
        picked = torch.where(top.values.unsqueeze(-1) > float("-inf"), picked, n)
        selected[start:stop].scatter_(1, picked.flatten(1), True)

        tail_start = (depth[start:stop] + 1) // compress_ratio * compress_ratio  # depth after its last full block
        selected[start:stop, :n] |= visible[start:stop] & (depth >= tail_start.unsqueeze(1))

    return selected[:, :n].contiguous().view(1, 1, n, n)


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
    modeling = sys.modules[type(model).__module__]
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
