#!/usr/bin/env python3
"""
tokenize_prompt.py
------------------
Uses the real GPT-2 BPE tokenizer (via HuggingFace) to:
  1. Encode a prompt → token IDs
  2. Save a decode table (id → string) for the C decoder

Usage (called by main.c via popen or pre-run):
  python3 tokenize_prompt.py "Hi how are you?" tokens.bin decode_table.txt

tokens.bin:
  [n: uint32][id_0: int32][id_1: int32]...

decode_table.txt:
  id<TAB>text
  (one line per token in GPT-2 vocabulary, text is UTF-8)
  Only writes tokens that appear in the generated output range.
"""

import sys, struct, os

def main():
    if len(sys.argv) < 4:
        print("Usage: tokenize_prompt.py <prompt> <tokens.bin> <decode_table.txt>")
        sys.exit(1)

    prompt     = sys.argv[1]
    tokens_out = sys.argv[2]
    decode_out = sys.argv[3]

    try:
        from transformers import GPT2Tokenizer
    except ImportError:
        print("ERROR: pip install transformers", file=sys.stderr)
        sys.exit(1)

    tok = GPT2Tokenizer.from_pretrained("gpt2")

    # Encode the prompt
    ids = tok.encode(prompt)

    # Write binary token file
    with open(tokens_out, "wb") as f:
        f.write(struct.pack("<I", len(ids)))
        for i in ids:
            f.write(struct.pack("<i", i))

    # Write decode table: every token ID → UTF-8 string
    # We write the full vocab so the C decoder can look up any generated token
    with open(decode_out, "w", encoding="utf-8") as f:
        for idx in range(len(tok)):
            # decoder maps id → string (may include special unicode chars)
            text = tok.decode([idx], clean_up_tokenization_spaces=False)
            # Escape newlines/tabs in the text so the table is line-based
            text_esc = text.replace("\\", "\\\\").replace("\n", "\\n").replace("\t", "\\t")
            f.write(f"{idx}\t{text_esc}\n")

    # Also print to stdout for the C program to capture
    print(" ".join(str(i) for i in ids))

if __name__ == "__main__":
    main()