"""M2.2 data pipeline: stream monology/pile-uncopyrighted → Pythia-tokenized uint16
mmap on /large. ONE fixed stream shared by every run (paired-data design, LOG entry).

Layout (out dir /large/share/li_qk/h2_m22/):
  val.bin    — first VAL_TOKENS tokens (held out; val loss + probe material)
  tokens.bin — training stream, target TRAIN_TOKENS (docs joined by EOS)
  meta.json  — progress high-water mark {written_train, written_val, done}
Run: python src/h2_m22_data.py   (CPU only; ~2–4 h; safe to resume — appends)
"""
from __future__ import annotations
import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "8")
import json, time
from pathlib import Path
import numpy as np

OUT = Path(os.environ.get("QK_LARGE", "/large")) / "share/li_qk/h2_m22"   # box B: QK_LARGE=$HOME/large (no /large mount; SYNC.md)
OUT.mkdir(parents=True, exist_ok=True)
VAL_TOKENS = 4_000_000
TRAIN_TOKENS = 4_600_000_000
DOC_BATCH = 512


def main():
    from datasets import load_dataset
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("EleutherAI/pythia-160m")
    eos = tok.eos_token_id
    assert eos is not None and tok.vocab_size < 65536
    meta_f = OUT / "meta.json"
    meta = json.loads(meta_f.read_text()) if meta_f.exists() else dict(
        written_train=0, written_val=0, done=False)
    if meta.get("done"):
        print("[data] already complete")
        return
    ds = load_dataset("monology/pile-uncopyrighted", split="train", streaming=True)
    it = iter(ds)
    # resume: skip documents already consumed (approximate by token count — we re-skip
    # deterministically since the stream order is fixed)
    skip_tokens = meta["written_val"] + meta["written_train"]
    val_f = open(OUT / "val.bin", "ab")
    train_f = open(OUT / "tokens.bin", "ab")
    t0, last_print = time.time(), 0
    written = skip_tokens
    skipping = skip_tokens > 0
    consumed = 0
    while written < VAL_TOKENS + TRAIN_TOKENS:
        texts = []
        for _ in range(DOC_BATCH):
            try:
                texts.append(next(it)["text"])
            except StopIteration:
                break
        if not texts:
            break
        enc = tok(texts, add_special_tokens=False)["input_ids"]
        buf = []
        for ids in enc:
            buf.extend(ids)
            buf.append(eos)
        arr = np.asarray(buf, dtype=np.uint16)
        if skipping:                                   # deterministic resume replay
            if consumed + len(arr) <= skip_tokens:
                consumed += len(arr)
                continue
            arr = arr[skip_tokens - consumed:]
            consumed = skip_tokens
            skipping = False
        pos = 0
        if meta["written_val"] < VAL_TOKENS:           # fill val first
            take = min(len(arr), VAL_TOKENS - meta["written_val"])
            arr[:take].tofile(val_f)
            meta["written_val"] += take
            pos = take
        if pos < len(arr):
            rest = arr[pos:len(arr) - max(0, (meta["written_train"] + len(arr) - pos)
                                          - TRAIN_TOKENS)]
            rest.tofile(train_f)
            meta["written_train"] += len(rest)
        written = meta["written_val"] + meta["written_train"]
        if written - last_print >= 100_000_000:
            val_f.flush(); train_f.flush()
            meta_f.write_text(json.dumps(meta))
            rate = (written - skip_tokens) / (time.time() - t0 + 1e-9)
            eta = (VAL_TOKENS + TRAIN_TOKENS - written) / (rate + 1e-9) / 3600
            print(f"[data] {written/1e9:.2f}B tokens ({rate/1e3:.0f}k tok/s, eta {eta:.1f}h)",
                  flush=True)
            last_print = written
    val_f.close(); train_f.close()
    meta["done"] = True
    meta_f.write_text(json.dumps(meta))
    print(f"[data] DONE: val {meta['written_val']/1e6:.1f}M + train "
          f"{meta['written_train']/1e9:.3f}B tokens", flush=True)


if __name__ == "__main__":
    main()
