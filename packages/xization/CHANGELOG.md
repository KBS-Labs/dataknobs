# Changelog

All notable changes to the dataknobs-xization package will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Added

- **`AuthoritySignal` and `AsyncAuthoritySignal` --- the authority stack as a
  resolution rung.** `dataknobs_common`'s rungs match text a vocabulary
  *enumerates*; these match text it *describes* as well, so a declared pattern
  --- a chip number, an account code, a date --- resolves where no enumeration
  could have carried it. They implement
  `dataknobs_common.entity_resolution.MatchSignal` over any `Authority`, hold
  no entity source, and publish `narrows() is False` because an authority
  stack has no declared types to filter against.

  ```python
  from dataknobs_xization import AuthoritySignal

  rung = AuthoritySignal(AuthoritiesBundle("clinic", auths=[breeds, chips]))
  rung.candidates("my golden retriever K-901 has been limping", k=5)
  ```

  Importing `dataknobs_xization` registers both flavours under
  `kind: "authority"`, which is also what clears the mark `dataknobs_common`
  leaves for the key it declares and cannot implement.

  **The evidence is `DECLARED` at 1.0**, like every rung over forms a
  vocabulary carries, and the entity id is the authority's own value id ---
  a regex arm's `canonical_fn` answer, a dictionary arm's frame **index**. A
  frame left on its default `RangeIndex` therefore resolves to row numbers,
  which the guide states plainly because no layer above can detect it.

  **It keeps less overlap than the `common` default and the guide says so.**
  An authority suppresses a form contained by one it already matched, so a
  vocabulary loaded as one authority per axis returns `golden_retriever` where
  `ScanningSignal` returns `golden_retriever` and `retriever` both. One
  authority per form keeps them. Both directions are pinned by tests.

  Each row is read through the column vocabulary of the authority that *wrote*
  it, so a bundle whose members were built with their own metadata answers
  correctly rather than reading `NaN` through the bundle's names. The order ---
  start ascending, end descending --- is the rung's own rather than a shared
  frame's sort, which can only order by one vocabulary.

  The registration leaves a key alone if something already holds it, so a
  consumer who registered their own `authority` rung keeps it and importing
  this package does not raise out of the import statement.

  The candidates are assembled by `dataknobs_common`'s `declared_candidates`
  rather than here, so a declared hit's evidence has one definition across both
  distributions.

- **`Authority.finders()` --- the authorities whose column vocabulary a row
  may be written in.** The read-back counterpart to
  `find_matches_with_finders`, for the point after a match's rows have been
  added to a shared `Annotations` and the boundaries are gone. A leaf answers
  with itself; an `AuthoritiesBundle` answers with its members' finders,
  recursively, because both of its annotation paths leave a member's rows in
  the member's own columns. An `AnnotatedText` carries one `Annotations` with
  one metadata while the rows in it may be in several vocabularies, and this is
  what lets a reader tell which.

### Fixed

- **A configured annotation column name no longer breaks reading the rows
  back.** `AnnotationsMetaData.sort_fields` holds col *types*, as its own
  parameter documents, and `sort_df` handed them to pandas as though they were
  col *names* --- so an authority built with, say, `start_pos_col="begin"` had
  its frame sorted on `start_pos`, and `KeyError` came out of the `df`
  accessor several frames below anything naming an authority. Each field is now
  translated through `get_col`, and one matching no col type is passed through
  unchanged so a caller who put an actual column name there keeps sorting by
  it. Invisible until something configured a name, because the two
  vocabularies agree for every default --- the feature the parameter exists for
  was the only thing that could reach it.

- **A named or numbered regex group that did not participate in a match is no
  longer annotated.** `RegexAuthority.build_match_annotations` walked the
  *pattern's* groups rather than the match's, so an optional group that matched
  nothing produced a row carrying a null text at span `(-1, -1)`, with
  `canonical_fn` called on `None` to name it. Reachable from any pattern
  written to match two spellings of one thing --- an account code with an
  optional prefix --- where the shorter spelling gained a phantom second
  annotation. Both loops now skip a group whose text is `None`; a group that
  legitimately matched the empty string has real offsets and keeps its row.

- **`Authority.annotate_input` no longer raises for input carrying no text.**
  `None` reached the return with its local never bound, for an
  `UnboundLocalError`; an empty or all-whitespace string failed the wrapping
  guard, stayed a `str`, and was handed to `add_annotations`, which asks it
  for `.annotations`. All three now answer with empty `Annotations`. A
  resolution rung is called with whatever a consumer typed, so every one of
  these was reachable from a cascade rather than only from a test. The
  parameter now declares the domain it accepts. The abstract `Annotator` base
  is deliberately left narrower: its other implementations dereference the
  argument and would raise, so widening the declared type there without
  widening the behaviour would publish a promise three classes do not keep.

### Changed

- **`DirectoryProcessor.process()` no longer refuses a caller already on an
  event loop.** The sync wrapper collected `process_async()` through
  `asyncio.run`, which raises `RuntimeError: asyncio.run() cannot be called
  from a running event loop`; it now goes through `run_coro_sync`, so the walk
  runs on a private loop and never on the caller's. The limitation is gone
  from the module docstring, the method docstring and the ingestion guides
  along with it. `process_directory()` carried the same limitation by
  delegation and loses it the same way. What is unchanged: the call still
  blocks the calling thread for the whole walk — `process_async()` is still
  the right call from async code — and it still collects before returning, so
  `files_skipped` is final when it returns.

- **`DirectoryProcessor.process()` and `process_directory()` take a
  keyword-only `timeout=`.** Both block the calling thread for a walk whose
  size they do not know in advance, and a caller inside a `def` has no
  cancellation of its own — so a source that stops answering was an unbounded
  block with nothing to interrupt it. `timeout=` bounds the whole walk and
  raises `TimeoutError` on expiry. It is keyword-only on `process_directory()`
  so it cannot be mistaken for a third positional argument. The default,
  `None`, waits for as long as the walk takes, so nothing changes for existing
  callers. The bound is on the **walk**: the throwaway loop is torn down
  afterwards and waits up to five seconds for a cancelled walk to unwind
  rather than destroying its cleanup mid-flight, so the worst case is
  `timeout` plus that. Documented in the directory-processor guide.

- **The sequence defaults on `get_lexical_variations` and
  `get_hyphen_slash_expansions_fn` are annotated `Sequence[str]`.** Each was
  declared `List[str]` while defaulting to a tuple, so the annotation
  described neither the default it carried nor what the body does with the
  value, which is iterate it. Callers passing a list are unaffected; a caller
  passing a tuple now type-checks, as it always ran.

### Fixed

- **`BackendDocumentSource` yields `-1` for a size a backend reports as
  `None`.** `DocumentFileRef` documents `-1` as the size "when the source
  cannot report size cheaply"; a file record with no size attribute already
  landed there, but one carrying the attribute as `None` — a remote backend
  that lists without stat-ing — reached `int(None)` and raised `TypeError`
  mid-enumeration. Both mean the same thing and both now answer `-1`.

- **`DocumentSource` declares its two streaming members as async generators.**
  `iter_files` and `read_streaming` were spelled `async def` returning an
  `AsyncIterator`, which describes a coroutine that *resolves* to an iterator
  — a shape every caller would have to `await` before iterating, and one
  neither shipped implementation has. Declared `def` returning an
  `AsyncIterator`, the protocol now matches `LocalDocumentSource` and
  `BackendDocumentSource`, which structurally conformed to it in neither
  direction before. No runtime behaviour changes; what changes is that a third
  implementation written to the declaration can no longer be one that breaks
  `async for` at every call site.

- **Dropping parentheticals no longer takes the text between two of them.**
  `PARENTHETICAL_RE` was `\(.*\)`, which is greedy: it matched from the first
  opening parenthesis to the last closing one, so `drop_parentheticals_fn` on
  `'AI (Artificial Intelligence) and ML (Machine Learning)'` answered `'AI '`,
  losing two words that were never inside a parenthetical. The class is now
  negated, `\([^)]*\)`, so two parentheticals are two matches. Negating the
  class rather than making the quantifier lazy: both stop at the first `)`,
  but `[^)]*` also cannot span one. `get_lexical_variations` drops
  parentheticals by default, so the truncated string was among the variations
  a caller matches against.

### Licensing

- **Relicensed from MIT to Apache-2.0.** This version and every later version
  of `dataknobs-xization` is licensed under the Apache License, Version 2.0. **All
  previously released versions remain under the MIT License**, on the terms
  under which they were published — the change is not retroactive, and the MIT
  text is preserved in `LICENSES/MIT-historical.txt`. Distributions now ship
  `LICENSE` and `NOTICE`, the package metadata declares
  `License-Expression: Apache-2.0`, and every shipped source file carries an
  SPDX `Apache-2.0` header. Building the package now requires
  `hatchling>=1.27`, which is where that metadata became expressible.

### Fixed

- **`DataframeAuthority` annotates text.** The dictionary half of the
  authority stack — `DataframeAuthority`, `TokenAligner`, `TokenMatch`, and
  the `MultiAuthorityFactory` that builds them — could not produce a single
  annotation. `authority.annotate_input(text)` now returns the declared forms
  it found, with offsets into the original text:

  ```python
  authdata = AuthorityData(pd.DataFrame({"animal": ["golden retriever"]}), "animal")
  authority = DataframeAuthority("animal", LexicalExpander(None, None), authdata)

  anns = authority.annotate_input("my golden retriever has been limping")
  # text="golden retriever", start_pos=3, end_pos=19, ann_type="animal", auth_id=0
  ```

  Three separate breaks stood in the way, each fatal on its own:
  `add_annotations` declared two parameters where `Authority.annotate_input`
  declares and passes one; the call into `LexicalExpander.build_first_token`
  supplied an `input_id` keyword that method has never accepted; and
  `TokenMatch.matched_text` read `Token.input_text`, a member no `Token` has
  ever had. `RegexAuthority` — the other arm, reached through the same base
  class — was unaffected throughout and is unchanged here.

  **`DataframeAuthority.add_annotations` now takes one argument**, the
  `AnnotatedText`, matching its base class and its two sibling
  implementations. The text object supplies both halves the old signature
  named separately: it is a `Text`, so the tokenizer reads the id and label
  off it, and it owns the `Annotations` the matches are added to. No working
  caller changes, because neither the one-argument call the base class makes
  nor a direct two-argument call could complete before this release.

- **`DataframeAuthority.get_id_by_variation` answers on a freshly built
  authority.** The variation-to-term index it reads is populated as a side
  effect of expanding the authority's values, and nothing in the method did
  that, so until some other member happened to expand them it returned an
  empty set for every variation — the same answer it gives for a variation
  that genuinely is not declared, which is why the difference was invisible.
  It now materializes the expansion first.

- **An `anns_validator` is shown one match at a time, whichever arm found
  it.** The parameter is documented on `Authority`, `LexicalAuthority` and
  `AuthoritiesBundle` as judging "a single match or entity", and
  `RegexAuthority` honoured that while `DataframeAuthority` made a single
  call carrying every match in the document. Two harms followed from the one
  call: a validator written to the documented contract was handed a batch
  spanning unrelated entities, which it cannot judge; and rejecting any one
  match discarded every other match in the same text, returning an empty
  `Annotations` indistinguishable from a text carrying no declared form.

  ```python
  # vocabulary ["golden retriever", "beagle"], a validator that rejects beagles
  authority.annotate_input("my golden retriever met a beagle")
  # now: the validator is called twice and "golden retriever" survives
  ```

  The unit is no longer each implementation's to choose. `Authority` grew
  `add_valid_annotations(text_obj, found)`, which judges and adds one
  match's rows at a time, and both arms route through it — so a match with
  several rows, as a regex with named groups produces, is still judged and
  added as one unit.

  **The matches arrive in document order**, whichever arm found them:
  `RegexAuthority` iterates `re.finditer`, and `DataframeAuthority` walks the
  token chain, so a validator carrying state across the matches of one text
  sees them in the order the text reads. Two matches beginning at the same
  position — which a dictionary authority produces where one declared form
  prefixes another, as `"golden"` does `"golden retriever"` — are not ordered
  further, so a validator should not read anything into which of those comes
  first. `TokenAligner.matches` and `TokenAligner.annotations` hold that same
  order; a consumer reading `Annotations.df` is unaffected either way, since
  `Annotations` sorts by span on every add.

- **An `AuthoritiesBundle` consults the `anns_validator` it was given.** The
  composite declared the parameter in its own signature, documented it with
  the same "single match or entity" sentence as the base class, and stored
  it — and then never called it. A consumer who handed a bundle a validator
  got no exception, no warning, and a result identical to the one they would
  have got by passing nothing:

  ```python
  # a bundle over a dictionary arm and a date pattern, with a validator
  # that rejects beagles
  bundle.annotate_input("my golden retriever met a beagle on 07/04/1776")
  # was: the validator is called 0 times and the beagle survives
  # now: it is called once per match and "golden retriever", "07/04/1776" survive
  ```

  A bundle could not use the seam its arms use, because its members added
  their rows straight to the shared text object and the match boundaries were
  gone by the time the bundle could look. `Authority` grew `find_matches`
  (below) so a member can hand its matches back instead, and the bundle
  judges what its members found.

  **A member judges its own matches first**, so a match a member's validator
  rejects is never shown to the bundle — the two are consulted innermost
  first, which is the only order in which either can mean anything.
  **The members are merged into document order rather than chained**, so a
  bundle, like each arm, shows a validator the matches of one text in the
  order the text reads even when a later-added member matched earlier;
  results are unaffected either way, since `Annotations` sorts on every add.

  A bundle carrying **no** validator delegates exactly as it did before. The
  one new failure is a member that does not implement `find_matches` inside a
  bundle that **does** carry one: that raises `NotImplementedError` naming the
  class, rather than judging a document while that member's matches go
  unexamined.

- **A validator reads a match's rows through the authority that found them.**
  Every accessor on `AnnotationsValidator.AuthAnnotations` read the rows
  through the authority the validator was called for, which assumes the
  authority that *judges* a match is the one that *found* it. That holds for
  an authority judging what it found itself, and it is false for a bundle
  judging what its members found: the rows are a member's, written in the
  member's column vocabulary, while the authority is the bundle.

  ```python
  # a member built with DerivedFieldGroups(field_type_suffix="_part"), one
  # regex with three named groups, on "abc 07/04/1776 xyz"
  auth_annotations.attributes
  # judged by its own authority: {'day': '07', 'month': '04', 'year': '1776'}
  # was, judged by a bundle    : {None: '1776'}
  # now, judged by a bundle    : {'day': '07', 'month': '04', 'year': '1776'}
  ```

  Three fields became one entry under `None`, silently: the bundle's
  `field_type` column name is not among the member's columns, so every row's
  field type read as missing and the three collapsed onto the one key. Where
  a member's `text_col` differed instead, the same assumption raised
  `KeyError` from inside pandas.

  **`AuthAnnotations` carries both authorities.** `auth` is unchanged and
  still means what it is documented to mean — the authority proposing the
  annotations, whose validator is being consulted and whose decision stands.
  `finder` is the authority that found them, whose columns they are written
  in, and every accessor reads through that. The two are one object wherever
  an authority judges what it found itself. A validator given as a plain
  callable is offered the finder as a third argument if its signature accepts
  one, so a validator written to either documented form is called the way it
  was written: `fn(auth, ann_dicts)` keeps working everywhere, including as a
  bundle's own validator, and `fn(auth, ann_dicts, finder)` is given the
  finder everywhere, including on an authority that finds its own matches —
  where the finder is that authority. Asking the callable is what makes both
  ends work, since a leaf is always its own finder and a bundle is never its
  own; conditioning the call on the two authorities differing instead would
  make each form uncallable by one of them.

  The finder travels with the match: `Authority.find_matches_with_finders`
  (below) pairs each match with the authority that found it, and
  `add_valid_annotations` judges the pairs. A member that is itself a bundle
  names a finder further down rather than naming itself, so nesting does not
  reintroduce the divergence one level in; the merge reads each match's start
  position through its finder's column for the same reason.

- **`AuthAnnotations.get_text` reads the column its metadata names.** It asked
  the row accessor for a column *name* where that method takes a column
  *type*, and the two are the same word only for the default metadata. An
  authority built with any other `text_col` could not read its own matches
  back — the lookup fell through to the derived-column path and raised.

- **An unrecognized column type reads as the missing value.**
  `DerivedFieldGroups.get_col_value` documents that it returns the caller's
  missing value for an unknown or missing column, and raised
  `UnboundLocalError` instead: it left the column name unbound for any column
  type other than the three it derives.

- **A document longer than about a thousand words is annotated.**
  `TokenAligner` walked the token stream by recursing along `next_token`,
  one stack frame per token, so the longest text `DataframeAuthority` could
  annotate was fixed by `sys.getrecursionlimit()` rather than by anything
  about the text. At the interpreter default of 1000 the last length that
  survived was 980 whitespace tokens — this package's own changelog, at 2849
  words, was past the edge. The `RecursionError` surfaced from whichever
  frame happened to be on the stack when the limit was reached, usually a
  pandas internal and never `TokenAligner`, so the traceback did not name
  the cause.

  The walk now follows `next_token` iteratively, so document length is
  bounded by memory rather than by the interpreter.

  **The walk is now linear in the token count, where it was quadratic.** A
  token that matched nothing was never marked as seen, so the traversal
  re-queried the authority for it once per enclosing level of the walk, and
  unmatched tokens are the bulk of any real document. `RegexAuthority` was
  never affected by either half of this: it finds its matches with
  `re.finditer` and walks no tokens.

- **`DataframeAuthority.has_value` answers with a `bool`.** It returned
  `np.any(...)` unconverted, so what a caller got back was a `numpy.bool`
  wearing the annotation `bool`. A truth test cannot tell the two apart, which
  is why nothing noticed; an identity test — `is True`, a `bool` key in a
  dict, a comparison against a sentinel — silently disagreed with the
  signature.

- **A `LexicalExpander` keeps an expansion function that is falsy.** Either
  function may be `None`, and the constructor asked which it had been given
  with `if variations_fn`. A function object is never falsy, so the question
  read correctly for every argument anyone had passed — but it is the wrong
  question, and any other callable that defines `__len__` or `__bool__`
  answers it wrongly. Such a callable was replaced by the default without a
  word, so every term expanded to itself. Both functions are now tested with
  `is not None`.

- **`CorrelatedAuthorityData.sub_authority_names` says it is unimplemented.**
  It was the one member of that class that did not — its four siblings all
  raise `NotImplementedError` — and it returned `None` against an annotation
  of `List[str]`. A caller iterating the result got `TypeError: 'NoneType'
  object is not iterable` from its own frame; a caller testing it for truth
  got a silent "this data correlates no sub-authorities", which is the one
  thing a `CorrelatedAuthorityData` cannot be. Nothing in this package calls
  it or overrides it.

- **The only `AuthorityFactory` accepts the data its abstract declares.**
  `MultiAuthorityFactory.build_authority` opened with a lookup declared on
  `MultiAuthorityData` alone, so a consumer holding the abstract — which is
  what publishing an abstract factory invites — got `AttributeError:
  'AuthorityData' object has no attribute 'get_authority_data'`, raised from
  inside the factory, for every plain `AuthorityData` handed to it. Flat data
  now supplies the authority of its own name, so the call builds over the data
  it was given:

  ```python
  authdata = AuthorityData(pd.DataFrame({"animal": ["dog", "cat"]}), "animal")
  authority = MultiAuthorityFactory("animal").build_authority(
      "animal", AuthorityAnnotationsBuilder(), authdata
  )
  ```

  A `MultiAuthorityData` still resolves to its named "sub" authority, so
  nothing that already worked through the factory changes.

- **The factory passes on the annotations builder it is handed.**
  `auth_anns_builder` is the abstract factory's own second parameter, and the
  only implementation named it, documented it, and then built the authority
  without it — so every authority built through the factory used a fresh
  default builder, and the annotation metadata a caller's builder carries
  (column names, the id column, anything else) silently did not apply.

### Changed

- **`AuthorityFactory` is generic over the authority data it builds from**, and
  `MultiAuthorityFactory` declares itself `AuthorityFactory[AuthorityData]` —
  the base type, because every `AuthorityData` now supplies the data for the
  names it holds and a factory building one authority at a time needs nothing
  narrower. The parameter is there for a factory that does need more — a
  container whose correlations it reads across sub-authorities, say — which
  can declare that container in its own type instead of narrowing the method
  and contradicting the base it implements. **The third parameter of
  `MultiAuthorityFactory.build_authority` is renamed `multiauthdata` →
  `authdata`**, matching the base class it overrides; a positional call is
  unaffected.

- **The annotations say where `None` is accepted and where it is returned.**
  `DataframeAuthority.prev_aligner`, `Authority.parent`,
  `CorrelatedAuthorityData.auth_records_mask` and
  `CorrelatedAuthorityData.combine_masks` are each documented to answer `None`
  and now say so, as do the optional constructor arguments across `Authority`,
  `LexicalAuthority`, `DataframeAuthority`, `MultiAuthorityFactory` and
  `TextFeatures`. Nothing changes at runtime; what changes is that a consumer
  type-checking against this package is told about the `None` it could always
  have received, instead of finding it.

### Added

- **`Authority.find_matches(text_obj)`**, the counterpart to
  `add_valid_annotations`: a subclass finds its matches and hands them back
  rather than adding them, so something other than the authority that found a
  match can judge it. `RegexAuthority` and `DataframeAuthority` implement it,
  and `AuthoritiesBundle` implements it by merging its members'. It is **not
  abstract** — a subclass written before it existed still constructs, and
  keeps working wherever nothing needs its matches back.

- **`Authority.find_matches_with_finders(text_obj)`**, the same matches paired
  with the authority that found each one — the authority whose columns the
  match's rows are written in, which is what anything reading those rows back
  has to go through. An authority that finds its own matches is its own
  finder, so the default pairs each match with itself and a subclass
  implementing only `find_matches` needs nothing more. `AuthoritiesBundle`
  overrides this one rather than `find_matches`, so a member's vocabulary
  survives being judged by the composite and by any composite above that.
  `add_valid_annotations` takes these pairs.

- **`Authority.add_valid_annotations`**, the seam above: a subclass finds the
  matches and this decides how they are judged. Two supporting members come
  with it — `RegexAuthority.build_match_annotations(match)`, which builds one
  match's rows and is now an override point in its own right, and
  `TokenAligner.matches`, which records the aligner's rows grouped by match.
  `TokenAligner.annotations` is unchanged as a flat list of every row, now
  derived from `matches`.

- **`AuthorityData.get_authority_data(name)`**, answering with the data the
  named authority is built over. Flat data holds one authority's values and so
  answers for its own name, raising `KeyError` for any other;
  `MultiAuthorityData` overrides it to build and keep its named "sub"
  authorities exactly as it already did. It is the one question a factory asks
  of the data it is handed, which is what lets one factory take either.

- **`MultiAuthorityFactory` takes the field groups and annotations validator it
  builds with**, as `field_groups` and `anns_validator` constructor arguments
  read back through `get_field_groups(name)` and `get_anns_validator(name)` —
  the shape `get_lexical_expander(name)` already had, so a subclass can vary
  any of the three by authority without reimplementing the build. Both were
  hardcoded `None`, so an authority built through the factory could not carry
  a consumer's derived field groups and validated nothing, though
  `DataframeAuthority` accepts both. Unconfigured, the factory builds what it
  built before.

### Removed

- **`more-itertools` is no longer a dependency of this package.** Its one use
  in the whole workspace was `more_itertools.consume(self.v2t[v].add(term)
  for v in variations)` in `LexicalExpander.__call__` — a generator written to
  spell a loop as an expression, over a call that returns nothing. It is a
  loop now, so nothing under `packages/xization/` imports the library and the
  declaration stated a floor this package no longer has a reason to hold.

  Unlike `nltk` below, this one does not arrive by another route: no other
  package in the workspace depends on it, so `uv lock` drops it from the
  resolution entirely. A consumer that imports `more_itertools` itself should
  declare it, as it should have been doing already.

- **`nltk` is no longer a *declared* dependency of this package.** No module
  under `packages/xization/` has ever referenced `nltk` directly — not in the
  package's whole history — so the declaration duplicated a floor this package
  neither imports nor is in a position to justify.

  This does not mean `nltk` goes away. `dataknobs_utils/__init__.py` imports
  `resource_utils`, which imports `nltk` at module level, so any
  `from dataknobs_utils import ...` in this package still pulls `nltk` in at
  runtime. It arrives through `dataknobs-utils`, which declares
  `nltk>=3.10.3` and actually calls it (`nltk.download`, `nltk.data.path`,
  `nltk.corpus.wordnet`). Nothing changes at install time or import time; the
  floor is simply stated once, by the package that imports it, instead of
  being maintained in two places that had to be kept in step by hand.

  The `nltk.*` entry in this package's `ignore_missing_imports` mypy override
  is retained deliberately: mypy follows the import chain above into
  `resource_utils`, and `nltk` ships no stubs.

## v2.2.1 - 2026-09-02

### Changed

- **The documentation filenames are lower-hyphen**, matching the workspace's
  one-document-one-name rule: `HTML_CONVERSION.md`, `INGESTION.md`,
  `JSON_CHUNKING.md`, `MARKDOWN_CHUNKING.md`, `RAG_HEADING_ENRICHMENT.md` and
  `RAG_QUALITY_FILTERING.md` are now `html-conversion.md`, `ingestion.md`,
  `json-chunking.md`, `markdown-chunking.md`, `heading-enrichment.md` and
  `quality-filtering.md`. A shared name is what lets one page be served from
  both the package tree and the documentation site with its links correct in
  each; the links that pointed at the old spellings are updated, and the
  ingestion index now distinguishes the reference from the consumer-facing
  guide rather than describing both the same way.

  No source change and no consumer-visible change: this is a maintenance
  release, cut so the workspace carries one version set rather than because
  anything here behaves differently.

## v2.2.0 - 2026-08-26

### Changed

- **`format_heading_for_display` now raises `ValueError` when `headings` and
  `heading_levels` differ in length, instead of silently dropping the excess.**
  The function is public (`dataknobs_xization.markdown`) and both lists come
  from the caller, so a mismatch was a caller error that produced quietly
  truncated output rather than a complaint. Callers passing equal-length lists —
  every caller inside this package — are unaffected. The same tightening was
  applied to the two internal heading walks in `Chunk.to_text` and
  `MarkdownChunker`, where `ChunkMetadata.headings` and
  `ChunkMetadata.heading_levels` are built together and a mismatch would mean a
  construction bug rather than bad input.

  Behaviour is otherwise unchanged: the rewrites in `RegexAuthority` (an
  `enumerate(..., start=1)` replacing a manual increment) and `JSONChunker`
  (decoding into a separate name rather than rebinding the loop variable)
  produce identical results.

## v2.1.0 - 2026-08-19

### Security

- **A `DocumentFileRef` can no longer read outside a `LocalDocumentSource`'s
  root.** `read_bytes` and `read_streaming` composed `root / ref.path` and
  opened the result unchecked, so a caller-built ref carrying `..` read a file
  from outside the tree — and an absolute `ref.path` discarded the root
  outright, which is the wider of the two spellings rather than a narrower
  case. Both readers now raise `PathEscapeError` before any filesystem call,
  so an escaping ref fails identically whether or not it names something that
  exists.
  Nesting is untouched — `sub/a.md` and an interior `a/../b` still read, and
  so does an absolute ref that lands back inside the root, because
  containment is judged on where the ref lands rather than on how it is
  spelled. The `DocumentSource` protocol now states the rule, so a
  consumer-written implementation over a bounded store inherits it.

- **A glob pattern could enumerate outside a `LocalDocumentSource`'s root.**
  `iter_files` handed each pattern straight to `Path.glob`, which treats `..`
  as an ordinary literal segment and descends through it — so a pattern of
  `../secrets/*.env` yielded refs for files outside the tree, each carrying
  that file's real size and its resolved absolute `source_uri`, which travel
  onward into chunk metadata. Patterns arrive from `IngestionConfig`, the
  same config plane every other guard in this area exists for.

  The claim that no check was needed here — that the class "already treats
  the root as its boundary" because every ref is derived with
  `relative_to(root)` — does not hold: `relative_to` re-expresses a path
  lexically and enforces nothing, returning `../outside/x` rather than
  raising. A match outside the root is now skipped and logged, and a ref's
  path is recorded in its canonical spelling so one file has one ref path
  whichever pattern found it. **Breaking** for a configuration whose patterns
  deliberately reached outside the declared root — such an ingest previously
  enumerated those files and then failed on the first read, since reading was
  already bounded; it now consistently ignores them.

## v2.0.0 - 2026-08-11

### Changed

- **A `chunker:` or transform key written `module.path:Name` now resolves
  instead of failing as an unregistered plugin.** The gate deciding whether a
  key names a class to import or a registry entry tested only for `.`, so a
  path using the other separator this workspace accepts was never handed to the
  resolver at all — it fell through to a registry lookup and reported that no
  such plugin was registered, which is not what went wrong. Both separators are
  recognised now. Registry keys are plain identifiers (`markdown_tree`,
  `merge_small`) and contain neither, and a registered key wins in any case:
  a path is resolved only for a key the registry does not already hold, so
  widening the gate cannot shadow an existing entry.

- **Resolving one of those keys fails as a `ConfigurationError`.** This site
  raised `ImportError`, `AttributeError` or `TypeError` depending on how the
  path was wrong, so a caller wrapping chunker construction in
  `except ConfigurationError` — which catches every other dotted-path fault in
  the workspace — caught none of them. They are now `DottedPathError` (path
  unresolvable) and `DottedPathTypeError` (resolved, not a `Chunker` /
  `ChunkTransform` subclass), both `ConfigurationError` subclasses, raised by
  the shared `dataknobs_common.imports` resolver rather than a local copy of
  it. Catch the old three and you catch nothing; catch `ConfigurationError`.

### Security

- **A config file's resolved path and the parser's text no longer reach the
  error message.** `KnowledgeBaseConfig.load` reported a failure by relaying
  both, and a parser reports a syntax error by quoting the line it choked on —
  an unterminated quote on an `api_key` puts the key in the text. The message
  now names the file and the exception class, with the parser's own words on
  `__cause__`. Matches what `dataknobs-config` does for the same failure, so
  the two loaders cannot disagree about it.


### Fixed

- `MultiAuthorityData.get_unique_vals_df` raised `TypeError` for columns with
  a pandas extension dtype (nullable `Int64`/`Float64`/`boolean`, `category`,
  `string`). Integer detection moved from `np.issubdtype` to
  `pd.api.types.is_integer_dtype`, so these columns produce a unique-value
  frame like any other — integer columns indexed by value, others by position.

  **Behaviour change for `timedelta64` columns.** `np.timedelta64` subclasses
  `np.signedinteger`, so the old check treated a timedelta column as integer
  and used the raw timedelta values as row IDs. It is now treated as
  non-integer and gets a positional 0..n-1 index. This is the only column type
  whose branch changed without previously raising; `datetime64`, `bool`, and
  the signed/unsigned numpy integer and float dtypes are unaffected.
- Raised the `nltk` floor to `>=3.10.2`, excluding the broken 3.10.1
  release. 3.10.1 shipped an import-security hook (`nltk/inisec.py`) that
  blocked any module whose install path resolved under the current working
  directory, without excluding an in-tree virtualenv. With the venv inside
  the project — uv's default layout — every nltk-initiated `import regex`
  raised `ImportError: Blocked import of regex from current working
  directory`, which in this workspace made six of the nine workspace packages fail to import at all;
  running with `cwd=/` blocked the standard library. The `PYTHONSAFEPATH`
  workaround named in the error message does not help, because the check
  is path containment rather than `sys.path` membership. Upstream removed
  the module in 3.10.2.

## v1.3.14 - 2026-07-29

## v1.3.13 - 2026-07-20

## v1.3.12 - 2026-07-15

### Security

- Bumped minimum `nltk` requirement from `>=3.9.4` to `>=3.10.0` to exclude
  versions affected by GHSA-p4gq-832x-fm9v / PYSEC-2026-2078 / CVE-2026-54293
  (CVSS 7.5, path traversal in `nltk.data.find()` / `load()` via percent-encoded
  `..%2f` sequences that bypass the `../` regex check once `url2pathname()`
  decodes them), now fixed in 3.10.0 — previously acknowledged as unfixed in
  v1.3.11. Flagged at the floor resolve by the `dependency-update` workflow. The
  related PYSEC-2026-597 / CVE-2026-12243 (same path-traversal class) still has
  no upstream fix and remains accepted — not reachable from this codebase, which
  loads only fixed corpus names (wordnet/omw-1.4/wordnet_ic) and never passes
  caller-controlled strings into `nltk.data.find()`.

## v1.3.11 - 2026-07-07

### Security

- Acknowledged GHSA-p4gq-832x-fm9v and PYSEC-2026-597 / CVE-2026-12243
  (both CVSS 7.5, path traversal in `nltk.data.find()` / `load()` via
  percent-encoded `..%2f` sequences that bypass the `../` regex check
  once `url2pathname()` decodes them) against the `nltk>=3.9.4` floor,
  flagged at the floor resolve by the `dependency-update` workflow.
  Both affect all `nltk` versions through 3.9.4 with no upstream fix.
  Not reachable from this codebase: no `nltk.data.find()` / `load()`
  call site takes caller-controlled input. The inline floor comment in
  `pyproject.toml` records the rationale.

## v1.3.10 - 2026-06-29

## v1.3.9 - 2026-06-22

### Changed

- ruff's `ASYNC` lint family (`flake8-async`) is now enforced for this
  package, so blocking I/O on the event loop inside `async def` code is
  caught at lint time. See the `async-transport` authoring rule.

- **`LocalDocumentSource.iter_files` no longer blocks the event loop.**
  The `Path.glob` walk and per-path `stat` are blocking filesystem
  calls; they are now collected in a single worker-thread hop via
  `asyncio.to_thread`, matching the already-offloaded `read_bytes` /
  `read_streaming`. Only the lightweight `DocumentFileRef` list is
  materialized — file contents are still read lazily per file.
  Behavior and ordering are unchanged.
- **`DirectoryProcessor.process_async` no longer blocks the loop when
  ingesting YAML/CSV files.** The YAML/CSV → markdown conversion is
  offloaded via `asyncio.to_thread`: `ContentTransformer` decides
  path-vs-inline-content with `Path(content).exists()` (a blocking
  `stat`), which previously ran on the event loop even though the
  content was already read into memory through the offloaded source.
  Markdown and (non-streaming) JSON paths were already loop-safe. Output
  is unchanged.
- **`DirectoryProcessor.process_async` no longer blocks the loop when
  streaming large local JSON/JSONL files.** The streaming branch hands
  the on-disk path to a *lazy* synchronous chunker generator that
  `open`/`gzip.open`s the file and reads forward on every chunk pull —
  previously each pull ran on the event loop. The generator is now driven
  on a worker thread and its chunks are pumped to the async consumer
  across a bounded queue, so the file open, gzip decompression, and every
  read happen off the loop. Streaming is preserved (chunks are not
  buffered whole-file), backpressure keeps memory bounded, and abandoned
  iteration tears the worker thread down and releases the file handle.
  gzip handling and path/format dispatch are unchanged. The remote
  single-JSON-tree branch (whole-tree parse of an in-memory buffer) is
  driven through the same worker-thread primitive — that parse is
  CPU-bound rather than blocking I/O, but a large tree would still stall
  the loop, so it is offloaded too.

Together these make the local-filesystem async directory-ingest path
(`process_async`, and `RAGKnowledgeBase.load_from_directory` above it)
loop-friendly across markdown, YAML, CSV, and JSON corpora — including
large streamed JSON/JSONL.

## v1.3.8 - 2026-06-02

## v1.3.7 - 2026-05-20

## v1.3.6 - 2026-05-18

### Added

- **`BackendDocumentSource(file_filter=)`** — optional keyword-only
  `Callable[[KnowledgeFile], bool]` predicate. Evaluated in
  `iter_files` *after* the glob/pattern match (and applied even when
  `patterns` is empty), it restricts enumeration to a subset of the
  backend's files. `None` (default) enumerates every matching file —
  behavior-identical to prior releases, so no existing caller
  changes. This is the source-layer seam that lets a per-file delta
  re-ingest (`dataknobs-bots`
  `KnowledgeIngestionManager.ingest_changes`) re-embed only the
  changed files while reusing the full pattern/chunking pipeline.

## v1.3.5 - 2026-05-09

### Fixed

- **`MarkdownChunker._create_chunk` no longer lets caller-supplied
  node metadata overwrite the chunker-supplied `node_type`** in
  `ChunkMetadata.custom`. Defense-in-depth: the md_parser callers
  do not currently set `node_type` in node metadata, so the path
  is practically unreachable today. The safeguard becomes zero
  marginal cost once
  `dataknobs_common.metadata.enforce_immutable_keys` exists, and
  emits a `WARNING` if a colliding override is ever attempted.

- **`ChunkMetadata.to_dict()` no longer lets `custom` overwrite
  structured fields.** Pre-fix, `to_dict` ended with
  `**self.custom`, so a custom entry sharing a key with a
  structured field (`headings`, `chunk_index`, `chunk_size`,
  `line_number`, `content_length`, etc.) silently overwrote the
  structured value in the serialized dict — same vulnerability
  class as the `_create_chunk` `node_type` defense, but covering
  the entire system-field surface. Post-fix, `**self.custom` is
  unpacked FIRST so structured fields win.

### Security
- Bumped minimum `nltk` requirement from `>=3.9.1` to `>=3.9.4` to
  exclude versions affected by GHSA-rf74-v2fm-23pw, CVE-2026-33230,
  and CVE-2026-33231 (one DoS, two in the WordNet browser HTTP
  component).

### Changed
- `KnowledgeBaseConfig._load_file` raises `IngestionConfigError` for
  malformed or unreadable config files. `yaml.YAMLError`,
  `json.JSONDecodeError`, and `OSError` no longer escape; callers
  should catch `IngestionConfigError`.

### Internal
- `KnowledgeBaseConfig._load_file` uses
  `dataknobs_common.config_loading.load_yaml_or_json`. Surface is
  `IngestionConfigError`.

## v1.3.4 - 2026-05-06

## v1.3.3 - 2026-04-23

### Added
- `DocumentSource` async protocol plus `DocumentFileRef` dataclass
  and `LocalDocumentSource` / `BackendDocumentSource` implementations
  (`dataknobs_xization.ingestion.source`). Decouples ingestion from
  the local filesystem so the same pattern-based pipeline can drive
  any storage backend. `BackendDocumentSource` derives a literal
  prefix from configured patterns and passes it to
  `backend.list_files(prefix=...)`.
- `DirectoryProcessor` dispatches `.md`, `.markdown`, `.txt`,
  `.yaml`, `.yml`, `.csv`, `.json`, `.jsonl`, `.ndjson` (plus `.gz`
  variants for JSON). YAML and CSV are transformed to markdown via
  `ContentTransformer` before chunking.
- `DirectoryProcessor.files_skipped` counter exposing the number of
  config files, excluded paths, and unsupported-extension files
  skipped during iteration.
- `KnowledgeBaseConfig.load()` resolves a config file from the
  directory root (`knowledge_base.(yaml|yml|json)`) and, as a
  fallback, from a `_metadata/` subdirectory. Symmetric with
  `RAGKnowledgeBase.ingest_from_backend`'s backend-side lookup.
- `ProcessedDocument.source_path` — source-relative file path
  (stable across local and backend sources; suitable for metadata
  filtering).
- JSONL streaming from non-local `DocumentSource`s parses one
  object per line directly from the async byte iterator, without
  buffering the full file.

### Changed
- `DirectoryProcessor.process()` is a thin sync wrapper over
  `process_async()`. The sync API is unchanged for callers; async
  callers should prefer `process_async()` directly. `process()`
  cannot be called from inside a running event loop.
- `DirectoryProcessor` constructor accepts a `DocumentSource` in
  addition to `str | Path` for `root_dir`. When a `DocumentSource`
  is passed directly, `processor.root_dir` is `None`; use
  `processor.source` to access the underlying source.
- `JSONChunker.stream_chunks` and
  `dataknobs_utils.json_utils.stream_json_data` accept file-like
  objects (`TextIO` / `BinaryIO`) in addition to paths. Existing
  path-based callers are unaffected.
