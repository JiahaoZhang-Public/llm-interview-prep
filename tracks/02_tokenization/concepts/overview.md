# Flashcards

## Why are subword tokenizers dominant?

They balance vocabulary size, OOV robustness, and compressibility.

## What is the practical difference between BPE and WordPiece?

BPE optimizes merge frequency, while WordPiece often chooses merges by likelihood gain.

## Why must encode/decode be reversible in a practice repo?

A non-reversible tokenizer hides bugs in whitespace and special-token handling.

## What is the point of the attention mask after padding?

It lets the model ignore padded positions during attention and loss computation.
