# Intent Classification

`dataknobs_llm.intent` ships a pluggable intent-classification surface
usable by any LLM-layer consumer that needs to route user input by
intent — tool routers, reasoning strategies, RAG query classifiers,
or downstream packages (for example, `dataknobs-bots` consumes it
from its wizard `intent_detection:` block).

The protocol is intentionally narrow: one async `classify` method
returning a frozen `IntentMatchResult`. Per-classifier configuration
(vocabulary, LLM provider, tokenizer) lives on the implementation,
injected at construction.

## Quick start

```python
from dataknobs_llm.intent import (
    IntentSpec,
    KeywordIntentClassifier,
)

clf = KeywordIntentClassifier()
result = await clf.classify(
    "yes please",
    [
        IntentSpec(name="accept", target="confirm"),
        IntentSpec(name="decline", target="abort"),
    ],
)
print(result.intent.name if result.intent else "no-match")
# → "accept"
```

`IntentSpec` declares each intent (`name`, `target`, optional
`keywords:` override, optional `extract:` field name). The classifier
returns an `IntentMatchResult` with the matched `intent`, an optional
`extracted` payload (from LLM-tier classifiers), a `rule_based` flag,
the preserved `raw_reply`, and a `confidence: float | None` field that
calibrated-confidence backends populate.

## Built-in classifiers

| Class | When to use |
|---|---|
| `KeywordIntentClassifier` | Rule-based vocab + tokenizer. Default tokenizer is word-boundary regex (a bare `"yes"` matches a standalone `"yes"` but not the `"yes"` substring of `"yesterday"`). Inject a custom tokenizer for I18N / fuzzy / N-gram / morphological matching. |
| `LLMIntentClassifier` | LLM-backed. Injectable provider + prompt template. Lenient response parsing accepts both the `DEFAULT_LLM_PROMPT_TEMPLATE` JSON shape (`{"intent": ..., "extracted": ...}`) and a bare intent ID. Provider errors return no-match (logged at WARNING); `asyncio.CancelledError` propagates. |
| `CompositeIntentClassifier` | Chain backends. `"first_match"` (default) returns the first non-None match — the standard "keyword first, optional LLM fallback" shape. `"vote"` queries every backend and breaks ties by classifier order. |
| `NegationFilter` | Decorator wrapping any classifier. Drops matches when `dataknobs_llm.extraction.grounding.has_negation` fires (closes the `"no, I don't want to accept that"` foot-gun). |

## Backend registry

`intent_classifier_backends` is a `Registry[IntentClassifierFactory]`
mirroring `event_bus_backends` and `lock_backends`. Built-in factories
`"keyword"`, `"llm"`, `"composite"` auto-register at import; consumers
register their own (embedding similarity, fuzzy match, locale-specific
keyword variants):

```python
from dataknobs_llm.intent import (
    create_intent_classifier,
    intent_classifier_backends,
)

def _embedding_factory(config: dict) -> "IntentClassifier":
    return MyEmbeddingClassifier(threshold=config.get("threshold", 0.8))

intent_classifier_backends.register("embedding", _embedding_factory)

clf = create_intent_classifier("embedding", {"threshold": 0.9})
```

`create_intent_classifier(name, config=None)` resolves a backend by
name and forwards the config dict. Unknown names raise `ValueError`
listing every registered backend (mirrors the `create_event_bus` /
`create_lock` shape).

## Injecting a custom tokenizer

```python
from dataknobs_llm.intent import KeywordIntentClassifier

def fuzzy_tokenizer(keyword: str, message: str) -> bool:
    # Both args arrive pre-lowercased.
    return keyword in message  # substring fallback

clf = KeywordIntentClassifier(tokenizer=fuzzy_tokenizer)
```

The tokenizer signature is `(keyword: str, message: str) -> bool`;
both arguments arrive pre-lowercased.

## Composing keyword + LLM fallback

```python
from dataknobs_llm.intent import (
    CompositeIntentClassifier,
    KeywordIntentClassifier,
    LLMIntentClassifier,
    NegationFilter,
)

base = CompositeIntentClassifier(
    [KeywordIntentClassifier(), LLMIntentClassifier(llm=provider)],
    strategy="first_match",
)
clf = NegationFilter(base, suppress_intents=frozenset({"accept"}))
```

`first_match` short-circuits on the first non-None match — keyword
match is fast and runs first; the LLM tier only runs when the keyword
classifier returns no match. The `NegationFilter` then drops `accept`
matches when negation is detected, leaving `decline` matches alone.

## Re-exports

`DEFAULT_VOCABULARY`, `DEFAULT_LLM_PROMPT_TEMPLATE`,
`DEFAULT_NEGATION_KEYWORDS`, `DEFAULT_AFFIRMATIVE_SIGNALS`,
`DEFAULT_NEGATIVE_SIGNALS`, `word_in_text`, and
`default_word_boundary_tokenizer` are importable directly from
`dataknobs_llm.intent`. The single-token English yes/no vocabularies
live in `dataknobs_llm.intent.defaults` under these public names;
consumers needing the same primitives for boolean recovery or
analogous text-classification tasks import them from there directly.
