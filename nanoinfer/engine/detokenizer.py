"""Incremental detokenization: token ids -> text, one token at a time, in O(1) per token.

Decoding the whole list every step would be O(n^2). Instead we decode a small sliding window:
  prefix_text = decode(ids[prefix_offset:read_offset])   (text already emitted, as context)
  new_text    = decode(ids[prefix_offset:])
and emit new_text[len(prefix_text):]. The context matters because a byte-level BPE token can be
half of a UTF-8 character (Chinese, emoji): while the decoded text ends with the replacement
character U+FFFD we wait for the next token instead of emitting garbage.
"""

from nanoinfer.engine.sequence import Sequence


def detokenize_incrementally(tokenizer, seq: Sequence) -> str:
    ids = seq.completion_token_ids
    skip = seq.sampling_params.skip_special_tokens
    prefix_text = tokenizer.decode(ids[seq.prefix_offset:seq.read_offset], skip_special_tokens=skip)
    new_text = tokenizer.decode(ids[seq.prefix_offset:], skip_special_tokens=skip)
    if len(new_text) > len(prefix_text) and not new_text.endswith("\ufffd"):
        seq.prefix_offset, seq.read_offset = seq.read_offset, len(ids)
        return new_text[len(prefix_text):]
    return ""


def check_stop_strings(seq: Sequence, num_new_chars: int) -> str | None:
    """If a stop string appears in the newly added text, cut output_text before it and return it."""
    text = seq.output_text
    best = None
    for stop in seq.sampling_params.stop:
        # only search where the new characters could have completed a match
        start = max(0, len(text) - num_new_chars - len(stop) + 1)
        idx = text.find(stop, start)
        if idx != -1 and (best is None or idx < best[0]):
            best = (idx, stop)
    if best is None:
        return None
    seq.output_text = text[:best[0]]
    return best[1]


def emittable_text(seq: Sequence, finished: bool) -> str:
    """Text that is safe to stream now. While stop strings are active, hold back the last
    (longest stop - 1) characters: they could be the start of a stop string."""
    end = len(seq.output_text)
    if not finished and seq.sampling_params.stop:
        end = max(seq.num_emitted_chars, end - (max(len(s) for s in seq.sampling_params.stop) - 1))
    delta = seq.output_text[seq.num_emitted_chars:end]
    seq.num_emitted_chars = end
    return delta
