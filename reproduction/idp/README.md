# IDP development stress test

This directory contains the executable development code for the InvertedDoublePendulum-v5 cross-family stress test.

The sequence was deliberately falsification-first:

1. semantic preflight;
2. learned-world-model threshold development (v0.2);
3. prospective control-slope redesign (v0.3B);
4. broad selective-intervention development (v0.4C);
5. bounded post-failure diagnostic.

The broad policy passed the frozen primary safety/coverage criteria on all 8 held-out development world models, but failed the preregistered efficiency gate (0/8 efficiency folds; median MEC/random cost ratio 0.8978 against a target of <=0.85). A bounded follow-up diagnostic did not rescue efficiency, and no broad hidden IDP evaluation was authorized.

## Public-config redaction

The original development configs committed future hidden seeds that were never consumed. Those seed values are not published here because releasing them would invalidate them as future holdouts. `configs/ORIGINAL_CONFIG_SHA256.json` records SHA-256 hashes of the original frozen configs; the `*_PUBLIC.json` files preserve the development settings while replacing only unspent hidden seed/root values with `REDACTED_UNSPENT_HOLDOUT`.

Development seeds, split seeds, mechanisms, model settings, gates, and evaluation logic remain visible.
