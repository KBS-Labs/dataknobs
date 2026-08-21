# Changelog

All notable changes to the dataknobs-data package will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Fixed

- **`Not` is exported from `dataknobs_data.validation`.** The three logical
  combinators are `All`, `AnyOf` and `Not`; the first two were exported and
  the third was not, so the only one that negates was the only one
  unreachable by the path the other two use. It was reachable as `~other`
  and by importing the submodule directly, which is why the gap survived. A
  test now derives the expected export set from the constraints module, so
  a constraint added later is covered without anyone remembering to.

## v0.9.0 - 2026-08-19

### Added

- **`compose_scope_key()` shares the scope-composition decision consumers
  were re-deriving.** `_match_metadata_filter` documents the empty-list
  filter value as unsatisfiable on every backend and names the consumers
  that rely on it to express a deliberate cross-scope no-op, but the
  decision producing that value lived only inside
  `VectorStoreBase._effective_filter`. Consumers composing the same scope
  onto their own filters wrote it themselves, and the obvious spelling —
  overwrite the caller's key with the bound value — is wrong in the one
  direction that costs data: it turns a request for *another* scope into a
  request for *this* one, so a destructive call that should match no rows
  matches every row in the caller's own scope.

  The three-case decision (absent key, in-scope value, out-of-scope value)
  is now a module-level function beside the contract it depends on.
  `_effective_filter` is unchanged in behaviour and delegates to it.

- **`ChromaVectorStore` tracks `created_at` / `updated_at`**, exposed via
  `include_timestamps=True` on `get_vectors()` and `search()` — the same
  surface `MemoryVectorStore`, `FaissVectorStore` and `PgVectorStore`
  already carry. A `timestamps:` config block now takes effect on this
  backend; previously it was parsed onto the store and never read.

  A Chroma collection is the only per-row storage this backend has, so the
  values live in the collection metadata under reserved keys, stripped from
  every read path — `get_vectors()`, `search()`, `search_documents()`,
  `metadata_fields()`, and the residual metadata filter behind `count()` and
  `clear()`. The stored form is an epoch float regardless of the configured
  `format`, so a store whose format config changes still reads rows written
  under the old one. Rows in a collection written by an earlier version
  report `None` for both until their next write; nothing is backfilled.

  One consequence to plan for: a collection written by this version and read
  by an earlier one surfaces those reserved keys as ordinary metadata,
  because the earlier version has no strip. Read a collection with the
  version that wrote it, or later.

- **`get_available_backends()` and `is_backend_available()`** on
  `DatabaseFactory`, `AsyncDatabaseFactory` and `VectorStoreFactory`. Three
  documents described both as factory methods, so following any of them
  raised `AttributeError`; the reference page's own "Backend Information API"
  section showed all three calls while only `get_backend_info` existed.

  Registration probes the driver each backend declares in its new
  `requires_module` metadata, so *registered* means *installed*:
  `is_backend_available("postgres")` is the check to make before offering a
  backend, and `get_backend_info(...)["requires_install"]` is what to print
  when it is absent.

  The probe is what makes the answer trustworthy. Backends do not agree
  among themselves about when to fail — `postgres`, `duckdb` and async
  `sqlite` import their driver at module top level, while `faiss`, `chroma`,
  `pgvector`, `s3` and async `elasticsearch` catch their own `ImportError`
  and raise only on construction. Reading "did the module load?" would have
  answered honestly for the first group and optimistically for the second.

- **Backends stay described when their driver is missing.** A backend whose
  driver is absent is recorded as known-but-unavailable rather than left out,
  so `get_backend_info(...)["requires_install"]` answers in the one situation
  it exists for — nobody reads it while the backend is installed — and
  `create()` reports the missing driver instead of an unrecognised name.

  `get_available_backends()` reports canonical names with registration
  aliases collapsed — `postgres` once rather than `postgres`, `postgresql`
  and `pg` — so it is a list of backends rather than a list of spellings.
  The unknown-backend `ValueError` prints that same list, from the same call,
  so the two cannot drift apart.

- **`get_backend_info()` on `AsyncDatabaseFactory`**, which had none. The
  other two factories carried it, so the async one answered two of the three
  questions its siblings answer.

- **`dataknobs_data.backend_selection`** — `DEFAULT_BACKEND`,
  `select_backend()`, `available_backends()`, `backend_available()`,
  `backend_info()`, `normalize_backend()`, `register_backend()`,
  `build_backend()`, `module_installed()` and `is_default_backend()`, the
  answers the three factories previously each held their own copy of.
  All ten are re-exported
  from `dataknobs_data`, so a consumer with a `PluginRegistry` of their own
  backends gets the same provenance logging, the same alias collapsing and
  the same availability probing without reimplementing any of it —
  `module_installed` included, since `register_backend(installed=...)` takes
  the probe as a parameter and wrapping the default one should not mean
  reaching into a submodule. The module imports nothing from its own
  package, so it is safe to import from anywhere in it.

- **`AsyncDatabase.from_backend()` and `SyncDatabase.from_backend()` resolve
  the way the factories do.** They held a fourth copy of the same four steps
  and were not migrated with the other three, so a correctly spelled backend
  whose driver is absent came back as `Unknown backend: postgres` — the
  answer that sends the reader hunting for a typo in a name that is right.
  They now report what to install, list canonical names rather than every
  alias, and construct through `from_config()` like every other path.

- **`dataknobs_data.testing`** — deterministic vector draws, for this package's
  tests and for consumers testing their own `VectorStore` implementations.
  `vectors(count, dim, seed=0)` returns a `(count, dim)` float32 array,
  `vector(dim, seed=0)` its first row, and `text_embedding(text, dim=384)` a
  vector derived from the text, so the same text always yields the same one.

  Each call builds its own `numpy.random.Generator`. None of them reads or
  seeds the process-global RNG, so a draw in one test cannot shift what any
  later test draws.

  `chroma_embedding_function(dim=8)` returns a chromadb embedding function
  backed by `text_embedding`, for the document paths that embed text rather
  than accepting vectors. It exists so a test never falls through to
  chromadb's default, which downloads ~166 MB of model weights on first
  use — a cold runner fails on that rather than skipping, and no `skipif`
  can see a download coming. It must be passed on every open: measured on
  chromadb 1.5.9, a persistent collection reopened without it comes back
  holding the default embedding function, silently, whether or not the
  class is registered with chromadb's function table.

### Changed

- **`SyncFileDatabase` and `AsyncFileDatabase` leave a `<path>.lock`
  file beside their data file.** It is no longer removed when the lock
  is released, because removing it is what let two holders in: release
  hands the lock to a blocked waiter holding a now-nameless inode, and
  unlinking there lets the next acquirer create a fresh inode and lock
  that instead. Two instances of either backend in one process
  are also genuinely serialized now, which they were not before —
  `fcntl` record locks are owned by the process, so the second acquire
  used to be granted immediately.

- **`FileLock` is no longer reentrant.** One thread acquiring the same
  path twice now deadlocks where it previously succeeded. The old
  behaviour was not a feature: it worked only because `fcntl` grants the
  owning process a lock it already holds, which is precisely the defect
  the intra-process mutex above fixes — there was no way to keep it
  without keeping the hole. In-tree callers never nest, but the class is
  importable, and `from dataknobs_data.backends.file import FileLock`
  still resolves (to `dataknobs_common.locks.FileLock`, where it now
  lives so the vector stores can reach it too).

  Two other differences on that same surface. Holding the lock needs
  create-or-write permission on the directory even to read under it, so
  a caller on a read-only mount must degrade rather than fail. And
  `FileLock(path, timeout=...)` now bounds the wait — worth setting from
  a worker of the shared `asyncio.to_thread` executor, where an
  unbounded wait parks a pooled thread for as long as the holder runs.

- **A persisted vector-store file keeps the permissions it had.** The
  scratch-then-rename publish used to reset the mode to the umask
  default on every save, discarding any `chmod` a consumer had applied.
  It now carries the replaced file's mode across. A file the store
  creates for the first time is owner-only.

- **`ChromaVectorStore.update_metadata()` and `update_metadata_where()`
  return rows *matched*, not rows written.** Both previously returned
  the number of rows actually sent to chromadb, which now differs: a
  matched row whose requested state already holds produces an empty
  update payload, which chromadb rejects and which has nothing to send
  anyway, so it is counted but not written. `matched` is what the
  abstract contract asks for and what the other backends report.

- **`domain_id=""` scopes on every backend.** `PgVectorStore` guarded
  its column predicates on truthiness while the metadata-carrying
  backends tested `is None`, so a store configured with an empty-string
  domain isolated on three backends and ran completely unscoped on the
  fourth. `VectorStoreConfig.domain_id` is `str | None` and never
  required a non-empty value, so the config was reachable. One shared
  predicate now decides for all four — a tenant boundary that
  disappeared on the backend swap it exists to survive was the worst
  available failure mode.

- **A configured `domain_id` now scopes the id-keyed operations too.**
  `get_vectors()`, `delete_vectors()`, `update_metadata()`,
  `add_vectors()`, `add_documents()` and `metadata_fields()` address
  rows by id (or not at all) and so built no filter, which left the
  tenant scope binding only the surfaces that take one. The write verbs
  answer differently from the reads and are described under *Fixed*. A scoped store answered for any id in the collection, and
  `metadata_fields()` returned the union of every tenant's key names.
  All four are now confined to the configured domain on every backend,
  and an out-of-domain id is answered exactly as an absent one — so a
  caller cannot distinguish "not here" from "not yours".

  **This is a behaviour change.** Code that used a scoped store to reach
  rows outside its domain — knowingly or not — now gets `(None, None)`
  from `get_vectors()` and no effect from the other two. Use an
  unscoped store, or one scoped to the row's own domain, where that
  reach was intended. Unscoped stores are unaffected.

- **An empty `add_vectors()` batch is a no-op on every backend.** It
  writes nothing and returns `[]`. Previously the four disagreed, and
  one of them corrupted the store: `MemoryVectorStore` minted an id for
  a zero-dimension vector and grew by a row, `FaissVectorStore` raised a
  bare `AssertionError` with no message, and `ChromaVectorStore` raised
  `ValueError` or `IndexError` depending on whether the caller passed
  `[]` or `np.array([])`. An empty batch is something a caller produces
  rather than intends — a comprehension that filtered everything out —
  so the guard belongs here rather than at every call site.

- **`VectorStore.update_metadata()` documents its replace contract on
  the ABC.** It said only "New metadata for each vector", and that
  ambiguity is why one backend read it as a merge. The base class now
  states that the supplied dict becomes the row's metadata outright,
  that a configured `domain_id` survives the replacement, and that an
  out-of-scope id is not updated.

- **`ChromaVectorStore.update_metadata()` now replaces a row's metadata
  instead of merging into it.** A key omitted from the supplied dict is
  removed, matching `MemoryVectorStore`, `FaissVectorStore` and
  `PgVectorStore` and the "new metadata for each vector" the base class
  documents. chromadb's own `update` merges, and nothing compensated — so
  a consumer clearing a key by omitting it got the key removed on three
  backends and silently retained on the fourth, from identical code
  against a config-selectable store.

  Code relying on the merge to perform a partial update needs to supply
  the full replacement dict, or use `update_metadata_where()`, whose
  contract is a merge and is unchanged.

- **A file `persist_path` written by two overlapping instances now
  raises instead of silently discarding one of them.** `save()` — and
  the implicit one inside `close()` — serializes the instance's whole
  in-memory state over the file, so two stores holding one path with
  overlapping lifetimes each wrote a snapshot that had never seen the
  other's rows, and the earlier writer's rows were gone from disk
  entirely. A store now records the file's identity when it reads or
  writes it and raises
  `dataknobs_common.exceptions.ConcurrencyError` rather than overwrite
  a file that changed underneath it. Sequential lifetimes are
  unaffected and keep appending, and a single writer saving repeatedly
  is unaffected. Concurrent writers need a backend that supports them,
  such as `pgvector`.

  This covers **`FaissVectorStore` and `MemoryVectorStore` alike** —
  both persist the same way, so both had the same defect, and the guard
  lives on `VectorStoreBase` rather than in either of them. Three
  things follow that consumers will notice:

  * `close()` now persists only a store that was **mutated**. An
    instance opened to read writes nothing on teardown. This is load
    bearing rather than an optimization: such a write moves the file's
    identity, which would make the instance actually holding new rows
    refuse to save them.
  * `save(force=True)` overwrites deliberately, accepting the loss of
    whatever the other writer persisted. It is the way out of a
    refusal, which otherwise repeats on every subsequent save because
    what it compares against has not moved.
  * The check is explicitly **best-effort** — modification time, size
    and inode. Two writes inside one filesystem timestamp tick that
    produce the same size are indistinguishable, so this catches the
    common accident and is not a lock.

- **A post-filtered Chroma search escalates its fetch instead of
  settling for one over-fetch.** The multiplier is also now one shared
  policy: `ChromaVectorStore` held two hard-coded copies of `k * 4`, and
  both come from `VectorStoreBase._overfetch_sizes`. Where the caller
  can bound the search — Chroma's `collection.count()` is native and
  O(1) — the sequence doubles up to the whole collection rather than
  stopping at the first size, so the answer becomes exact rather than
  merely over-fetched. A sparse filter therefore costs several
  round-trips plus one `count()`; declaring the key in
  `scalar_metadata_keys` pushes the predicate down and avoids both.
  `dataknobs_bots`' knowledge-layer over-fetch is deliberately *not*
  merged into it — that one compensates for tombstone visibility rather
  than for scope, and coupling them would tie the knowledge layer's swap
  semantics to a store constant.

- **`UserStateStoreConfig.backend` now defaults to `None`, not `"memory"`.**
  The typed default was forwarded unconditionally, so a config that named
  no backend reached the factory as an explicit choice and the absence was
  consumed one frame above the only code positioned to report it — an
  unpersisted store, silently, for a config whose author may only have
  meant to leave the choice to the deployment. The key is now forwarded
  only when the config names one. Code reading `config.backend` and
  expecting a string must handle `None`, which means "not chosen here".


- **A config with no `backend` key now logs at WARNING.** It was INFO — the
  same line, naming the same backend, as an explicit `backend: memory` — and
  on `AsyncDatabaseFactory` there was no line at all. The two are not the
  same event. One is a deployment choosing an in-process store; the other is
  what is left when a config arrives empty, and an empty config handed to a
  factory does not fail. It produces a store that answers every query with
  zero results and loses everything when the process restarts.

  An explicit `backend: memory` is still INFO, which is what keeps the
  WARNING meaning something. The object built is identical either way — that
  identity is what made the difference invisible.

- **`get_backend_info()` answers for an alias.** An alias carries no registry
  metadata of its own, so asking about `pg` returned `{}` while every other
  question about it answered for postgres. It resolves to the key describing
  the same backend.

- **The unknown-backend `ValueError` lists canonical names**, where it
  previously listed every registered spelling. The three lead sentences are
  unchanged, including `AsyncDatabaseFactory`'s deliberately distinct one —
  an unrecognised name there usually means the backend exists without an
  async variant, which is worth saying differently from "you typed it wrong".

- **Backend-selection log records now come from
  `dataknobs_data.backend_selection`.** They were emitted by
  `dataknobs_data.factory` and `dataknobs_data.vector.stores.factory`, so a
  consumer routing or filtering by logger name needs to add the new one.

- **`VectorStoreFactory.create()` reports a missing driver differently.** It
  used to build the store, catch the `ImportError` the store raised from
  `_setup`, regex the text for a `pip install X` and re-emit it as
  `Faiss backend requires faiss-cpu` — degrading to `Backend 'faiss' has
  missing dependencies` whenever that pattern did not match. A store whose
  driver is absent is now refused before construction, with the same
  `ValueError` the database factories raise:
  `Backend 'faiss' is known but not available here. Install with: pip
  install faiss-cpu`. Code matching the old text needs updating; code
  catching `ValueError` does not.

- **A `vector_store` section naming an uninstalled backend still validates.**
  `StructuredConfig.validate()` resolves the section's config class through
  the same registry `create()` uses, so gating registration on the driver
  made a valid `backend: faiss` section resolve to nothing on a machine
  without `faiss-cpu` — reported as matching no known variant, which reads
  as a misspelled discriminator. Whether a config is well-formed does not
  depend on which optional drivers the machine reading it happens to have.
  A known backend that cannot be built here now skips validation of its
  section; only an unregistered name is still an error.

- **A `backend` key that is present but unusable says which way it is
  unusable.** `backend: null`, `backend: ""` and a non-string all rendered
  into `Unknown backend type: <value>`, which reads as a backend of that
  name and sends the reader looking for a spelling mistake. `backend: null`
  additionally raised `AttributeError` from the config-validation path,
  where the construction path raised `ValueError` — the two read the config
  through one function now, so they classify it identically. Names are
  stripped and lowercased on both paths.

- **A registered object that cannot be built from a config is reported as
  such.** `PluginRegistry` accepts any callable, while the database
  factories require the class form; a bare function reached
  `.from_config` and raised `AttributeError: 'function' object …` from
  inside the factory, naming nothing that would lead back to the
  registration. It now raises a `ValueError` naming both.

### Fixed

- **The scratch sweep read its target's name as a glob pattern.** A
  `persist_path` is a filename, so one containing `[`, `?` or `*` was
  interpreted rather than matched — the sweep stopped finding its own
  orphaned scratch files, reopening the unbounded leak it exists to
  close, and started matching a different file's. The name is now
  escaped.

  It also matched any target whose name merely *begins* with this one,
  so a store persisting to `idx` could unlink a live scratch file
  belonging to one persisting to `idx.pkl` — a different target, under
  a lock the sweeping store does not hold, whose writer is about to
  rename it. The scratch token is now required to be dot-free, which
  `mkstemp`'s alphabet guarantees for this target's own files.

  Sweeping moved from the save bracket to the publish itself, which is
  the only thing that creates scratch files and the only one that knows
  every path being written: a two-file store's side-car leftovers used
  to be reached, when they were reached at all, by the prefix looseness
  above.

- **The rename flush covered one directory, not each.** Both stores
  publish siblings, so flushing the first published path's directory was
  right by coincidence; a store publishing into two made only the first
  rename durable. Deduplicated by directory now, which is the same one
  flush in practice and the right one for a store inheriting the
  bracket.

- **`save(force=True)` called a file that never existed "unchanged".**
  True about the loss — there is none — but "unchanged" describes a
  file, and the WARNING is what an operator reads to find out what a
  destructive flag just did. Reported as no file at that path.

- **`fsync` before publishing was a silent no-op on Windows.** The
  staged file was opened `O_RDONLY` to flush it, and Windows implements
  `os.fsync` as `_commit`, which rejects a read-only descriptor — so
  the crash-durability guarantee was absent there, swallowed into a
  debug line. Opened `O_WRONLY` now; nothing is written through the
  handle.

- **A half-landed publish left a symlinked store refusing every save.**
  `FaissVectorStore` writes an index and a `.meta` side-car; if the
  second rename fails after the first has landed, the store has
  replaced the tracked file without reaching its identity stamp, so it
  refreshes the stamp on the way out — otherwise every later save
  raises `ConcurrencyError` naming a conflicting writer that does not
  exist, with `save(force=True)` the only escape.

  That recovery re-derived the tracked path from `persist_path`, which
  is not resolved, and compared it against the paths it had published,
  which are canonical. Through a symlink the two never matched, so the
  branch never ran — in exactly the layout resolving was introduced
  for, a stable name pointing at versioned storage. The publish is now
  told which path carries the stamp instead of deducing it.

- **A compressed file database locked a path nothing read or wrote.**
  `SyncFileDatabase` / `AsyncFileDatabase` built their `FileLock` before
  applying the `.gz` suffix a configured `compression` implies, so
  `{"path": "data.json", "compression": "gzip"}` wrote `data.json.gz`
  while locking `data.json.lock`. The same data file reached the other
  way — `{"path": "data.json.gz"}`, gzip auto-detected — locked
  `data.json.gz.lock`, so two instances over one file were serialized
  by nothing. `FileLock` resolves symlinks and keys its mutex by inode
  to guarantee one file gets one lock however it is spelled; this
  defeated all of it from the caller's side by naming the wrong file.

  Path, format, compression and handler are now resolved together, by
  one function both backends call, and the lock is built from the
  result. They are interdependent — a `.gz` suffix names the
  compression, the stem beneath it names the format, and a configured
  compression renames the file — and the two `_setup` bodies had
  derived them apart, identically, which is why the defect was in both.

  A temp database with `compression` set also leaked the name
  `tempfile` reserved: `close()` removed the compressed file and its
  lockfile but not the `.json` stub beneath, one `/tmp` entry per
  process. The stub is what keeps the compressed name unique, so it is
  held until `close()` rather than dropped at setup.

- **Two single-file vector stores over one `persist_path` could both
  write, with neither raising.** `FaissVectorStore` and
  `MemoryVectorStore` refuse to overwrite a file that changed since
  they read it — but the check and the write it guards ran on a worker
  thread with a whole serialization between them, so two instances both
  passed the check before either wrote. The second replaced the first's
  file with a snapshot that had never seen its rows, and the refusal
  that exists to prevent exactly that never fired.

  The check, the write and the stamp that follows are now held under
  `dataknobs_common.locks.FileLock` on a sibling `<persist_path>.lock`,
  so the second writer meets a file that has already moved and raises
  `ConcurrencyError` as documented. This closes the same-process case
  as well as the cross-process one: the per-instance `asyncio.Lock`
  could never see a second instance, and POSIX record locks alone do
  not exclude two threads of one interpreter.

  The `.lock` file is created on the first save or load and left in
  place; the lock is advisory and local-filesystem only, so a networked
  mount carries the same caveat the identity check already did.

- **`load()` could run inside the store's own `save()`.** The read ends
  by stamping two fields the save path owns, so a load landing mid-save
  declared the store in step with a file the save had not written yet —
  after which `close()` skipped persisting a mutation nobody wrote. For
  FAISS it was also a torn read: the index and its `.meta` side-car are
  published by two renames, and a reader between them paired a new
  index with a stale side-car. `load()` now takes the same locks the
  save does.

- **Concurrent publishes collided on one scratch file.** Every writer of
  a path staged to the same `<file>.tmp`, so one wrote over the other's
  bytes and the loser's cleanup could unlink a file the winner was about
  to rename — turning a silent clobber into a spurious
  `FileNotFoundError`. Each write now stages to a uniquely named scratch
  file in the target's own directory.

  Unique names have to be cleaned up rather than overwritten, so two
  things do that. A write that raises — an unpicklable value, a full
  disk — has its scratch file removed, where the cleanup previously ran
  over a list built only from writes that *succeeded* and left the
  partial snapshot behind. And a scratch file left by a process killed
  mid-save is swept by the next save of that target, under the lock that
  guarantees no live writer owns it.

- **A symlinked `persist_path` is written through, not replaced.**
  Publishing is `os.replace`, which replaces the *symlink* rather than
  following it: the first save turned a stable name pointing at
  versioned storage into a regular file holding that store's snapshot,
  while the versioned file it had pointed at kept the old one — two
  files silently disagreeing, with nothing to say which was live. The
  store now resolves `persist_path` once and derives everything from
  the result, so the index, FAISS's `.meta` side-car and the lockfile
  all land beside the resolved target rather than splitting across two
  directories.

  This is also what makes the lock hold. `FileLock` takes its lockfile
  beside the resolved target so two spellings of one file contend for
  it; a save that destroyed the symlink moved the lockfile with it, so
  two writers were serialized until the first write and not after.

- **A published file is flushed before the rename that publishes it.**
  `os.replace` is atomic against a concurrent reader but not against
  power loss: on a journalled filesystem the rename metadata can reach
  the disk while the data it names has not, leaving a truncated file
  that has already replaced a known-good one. The file and the directory
  entry are now both `fsync`ed, so staging protects against a power cut
  and not only against a crashed process.

- **A read-only `persist_path` directory no longer refuses to load.**
  Holding the file lock means creating or opening `<path>.lock`, so
  taking it on the read path made a *load* require write access that a
  load never needed — an index baked into a read-only image layer, or
  served from a read-only mount, stopped opening. `load()` now falls
  back to an unlocked read and logs at `WARNING`. Only the read path
  does: nothing can be published into a directory this process cannot
  write, so there is no concurrent writer to exclude, while `save()`
  keeps the hard lock because there the write *is* what needs excluding.

  That fallback is bounded by the directory being unwritable, not by
  the lock having failed. Any other reason it fails — a lockfile owned
  by another uid, `ENOLCK` from an NFS mount without `lockd`, `EMFILE`
  from descriptor exhaustion — leaves a directory that *is* writable,
  so a writer to exclude can exist and an unlocked read can return a
  half-published state. Those raise, with a message naming the lock and
  the umask that usually causes the first of them.

- **`save(force=True)` returned silently.** It is a deliberate
  destructive bypass of the staleness check, and it is now logged at
  `WARNING` every time, saying whether anything was actually discarded
  — the line an operator wants when asking where the rows went. A file
  that is *gone* is reported as such rather than as a discarded write:
  both fail the identity comparison, but only one of them had another
  writer's rows in it.

- **A temporary file database no longer removes its lockfile while an
  operation holds it.** `close()` unlinks the generated file and its
  `<path>.lock`, which is correct for a path belonging to one instance,
  but it ran outside the instance lock — so a `close()` concurrent with
  an in-flight write removed the lockfile that write was holding, which
  is the handover defect `FileLock` was fixed to stop causing. Both the
  sync and async backends now clean up under the lock every other
  operation takes, through one shared helper rather than two copies.

- **`update_vectors()` no longer resets a row's `created_at`, or destroys
  rows on a refused batch.** It was implemented as `delete_vectors()`
  followed by `add_vectors()`. The delete bought nothing — `add_vectors()`
  already replaces a row's metadata outright on every backend, which is
  the only thing the delete guaranteed — and it cost two things.

  It took the row's timestamp tracking with the row, so the re-add had
  nothing to preserve and stamped a fresh creation date. That breaks the
  documented rule that `created_at` survives every write to a tracked id,
  in the way the null-timestamp rationale warns about: a re-ingest sweep
  built on `update_vectors()` rewrote every row's creation date to the
  moment of the sweep, after which nothing could tell a fabricated date
  from a real one.

  And under a configured `domain_id` it lost data. A scoped
  `delete_vectors()` skips an out-of-domain id and deletes the rest,
  while `add_vectors()` refuses the batch outright — so a batch mixing
  the caller's own ids with one it does not own deleted the caller's rows
  and then declined to put them back, reporting an id the caller had not
  asked to lose. `update_vectors()` is now an alias for `add_vectors()`.

- **A scoped write no longer narrows a row that belongs to several
  domains.** Scope membership follows the four-quadrant rule, so a row
  tagged `["t1", "t2"]` belongs to both — and the write guard, which
  resolves membership the same way, admits a `t1`-scoped store's write to
  it. But the write-path default re-applied the configured scope as a
  *scalar*, so the admitted write replaced `["t1", "t2"]` with `"t1"` and
  the co-owner silently lost the row. Reachable through `add_vectors()`,
  `add_documents()` and `update_metadata()` on Memory, FAISS and Chroma;
  `PgVectorStore` keeps `domain_id` in a scalar column and cannot hold
  the shape. A write that does not mention `domain_id` now preserves the
  row's own value rather than re-stamping the configured one.

- **Declaring the configured scope key in `scalar_metadata_keys` no
  longer splits `ChromaVectorStore` against itself.** `scalar_metadata_keys`
  is a promise about stored values that the write path cannot keep for
  `domain_id`: the configured scope is a default rather than an override,
  so a caller can store a list there through the ordinary API. A list is
  stored sentinel-encoded, so the native `$eq` the declaration enables
  matched nothing — `count()` and `search()` went blind to a row that
  `get_vectors()` still returned and `clear()` could not remove. The
  scope key now stays in the post-filter however it is declared.

- **A `PgVectorStore` batch write is now atomic.** `add_vectors()` inserts
  row-by-row over a pooled connection, and asyncpg gives a bare `execute`
  its own implicit transaction — so any mid-batch failure (a bad
  `chunk_index`, a dimension mismatch, a serialization error) committed
  every row before it and left the caller retrying on top of a
  half-applied write. The loop now runs inside one transaction.

- **A malformed UUID id names itself on a scoped `PgVectorStore`.** The
  ownership probe a scoped store runs before its inserts binds the whole
  id array, and Postgres answers a malformed element with a message the
  guided-error wrapper rendered by interpolating that entire array. Only
  scoped stores were affected; `delete_vectors()` had validated
  client-side for exactly this reason, and `add_vectors()` now does too.

- **A consumer value under a reserved timestamp key can no longer become
  a `ChromaVectorStore` row's creation date.** Reserved keys were kept out
  of storage only by the stamping step overwriting them, and stamping
  returns early for a row the store does not yet track — so on a
  collection written before this backend tracked timestamps, a numeric
  value under the reserved key was stored and read back as that row's
  real `created_at`. The keys are now dropped where every write path
  already funnels, at the encoding boundary.

- **Re-adding an id on `ChromaVectorStore` now replaces that row's
  metadata instead of merging into it.** `add_vectors()` and
  `add_documents()` upsert on id conflict, and chromadb's `upsert`
  merges the metadata it is given into what is already stored — a key
  the caller omits survives. So re-adding `id="x"` with `{"rev": 2}`
  over a stored `{"tenant": "A", "rev": 1}` left `tenant` behind on this
  backend while Memory, FAISS and pgvector replaced the row outright,
  and re-adding with no metadata at all kept the entire prior dict. A
  consumer correcting a row's metadata got the correction on three
  backends and a silent merge on the fourth, with the stale keys still
  answering filters. Both write paths now name the departing keys with
  a `None` value to delete them, the same mechanism `update_metadata()`
  already used.

- **A scoped store can no longer capture another domain's row by
  writing its id.** `add_vectors()` and `add_documents()` are id-keyed
  like the read verbs, but were not scoped: because the row they write
  carries the configured `domain_id`, writing an id another domain owned
  destroyed the original and relabelled the replacement into the
  caller's domain — the victim's `count()` dropped by one and nothing
  recorded that it happened. On pgvector the capture was explicit in the
  SQL, whose `ON CONFLICT` clause assigns `domain_id` from the incoming
  row; on Chroma and Memory the stolen row also inherited the victim's
  `created_at`.

  All four backends now raise `VectorDomainScopeError` (new, exported
  from `dataknobs_data.vector`; subclasses `ValueError`) before writing
  anything, so a rejected batch leaves no partial state. A row carrying
  no domain at all is refused on the same grounds — every scoped read
  already treats it as absent. Unscoped stores are unaffected and remain
  the way to address a whole collection deliberately.

  **This is a behaviour change** for a scoped store that was writing
  ids outside its domain; that write is now an error rather than a
  silent capture.

- **A row belonging to several domains is no longer visible to half its
  own store.** Scope membership follows the same four-quadrant rule as
  any other metadata key, so a row whose `domain_id` is a list belongs
  to every domain in it — which the filter-keyed surfaces always
  honoured. The id-keyed check compared with `==` instead, so `count()`
  reported a row that `get_vectors()` called absent, `delete_vectors()`
  refused, and `clear()` then removed anyway. Both halves now resolve
  the scope through one evaluator. pgvector is unaffected: its
  `domain_id` is a scalar column and cannot hold the shape.

- **`add_documents([])` is a no-op on `ChromaVectorStore`, and the
  id-keyed verbs accept an empty id list.** `add_documents` never got
  the empty-batch guard its `add_vectors` sibling has, and
  `get_vectors([])`, `delete_vectors([])` and `update_metadata([], [])`
  reached chromadb's id validator, which rejects an empty list. All four
  raised `ValueError` on Chroma alone while the other three backends
  returned the empty answer, so a consumer whose code was correct
  everywhere else crashed after a backend swap.

- **`update_metadata()` no longer pushes a row out of its own domain.**
  On Memory, FAISS and Chroma the configured `domain_id` lives *in* the
  metadata dict, and `update_metadata()` replaces that dict wholesale —
  so a caller updating one field, without restating a scope key it has
  no reason to know about, silently unscoped the row. The row then
  vanished from `count()`, `search()` and `update_metadata_where()`, and
  could not even be deleted: a scoped `clear()` resolves to
  `{"domain_id": <configured>}`, and an absent key never matches a
  filter. It became an orphan that only an unscoped store could still
  see. The write-path default that `add_vectors()` applies is now
  applied here too. pgvector was never affected — its `domain_id` is a
  column the metadata write does not touch.

- **`ChromaVectorStore` no longer invents a `created_at` for a row that
  predates timestamp tracking.** An update stamped the current time into
  `created_at` whenever the row had none, so a single
  `update_metadata_where(None, ...)` migration sweep would record every
  legacy row as created at the moment of the sweep, with nothing left to
  distinguish a fabricated date from a real one. A write establishes
  tracking; an update no longer does — matching Memory and FAISS, which
  guard on the row having a side-car entry, and pgvector, which leaves a
  `NULL` `created_at` alone.

- **`ChromaVectorStore.add_vectors()` no longer discards a write to an id
  the store already holds.** It reached chromadb's `add`, which drops a
  duplicate id silently — no exception, no warning, the original vector
  and metadata retained. Re-adding an existing id is an upsert on every
  other backend, so a consumer correcting an embedding got the correction
  on three backends and kept the stale vector on the fourth, with nothing
  to indicate which had happened. `add_documents()` had the same defect
  and is fixed with it.

- **`ChromaVectorStore.add_documents()` now applies the configured
  `domain_id` to the rows it writes.** Its sibling `add_vectors()`
  defaults the configured domain into every row; this path did not, so a
  store scoped by `domain_id` wrote rows carrying none — and every scoped
  read then filtered them back out. `count()` omitted a document the store
  had just written, and `search_documents()` for that document's own text
  returned some other row.

- **`ChromaVectorStore.search(filter=...)` under-returned for the same
  reason, at a wider window.** A filter that cannot be pushed down is
  applied in Python *after* Chroma has truncated to `n_results`, and
  over-fetching a fixed `k * 4` only moves the threshold: a filter
  matching fewer than one candidate in four still lost rows, and a
  sparse one returned nothing at all while `count(filter=...)` reported
  many matches. The fetch now escalates to the collection size, so a
  filtered search returns a full `k` whenever `k` rows match. Note that
  a filter is post-filtered *unless* its key is declared in
  `scalar_metadata_keys`, which defaults to empty — so this was the
  default path, not an unusual one.

- **`ChromaVectorStore.search_documents()` scored every store as though
  it were cosine.** The collection is created with `hnsw:space` from the
  configured metric, so a store configured `euclidean`, `l2`,
  `dot_product` or `inner_product` receives distances in that metric —
  but this method applied `1 - distance` unconditionally, reporting
  *negative* scores for any L2 distance above 1. `search()` had the
  correct per-metric conversion all along; the two now share one, so
  they cannot disagree again.

- **`ChromaVectorStore.search()` and `get_vectors()` raised `TypeError`
  on `include_timestamps`.** The argument is on the `VectorStore` ABC
  and every other backend accepted it, so passing it broke exactly the
  runtime backend swap the filter-semantics doc promises. Both accept it
  now, and answer from values this backend really tracks — see the
  timestamp-tracking entry under *Added*.

- **Consumer metadata is no longer shared between a store and its
  caller, in either direction.** On Memory and FAISS a caller could edit
  a stored row without calling a mutator — `search()` and `get_vectors()`
  handed back the live `metadata_store` entry, and `update_metadata()`
  kept the dict it was given — while Chroma and pgvector, which
  serialize at their boundary, were unaffected by identical calling
  code. Both directions are now copied, and the copy is **deep**: a
  shallow one leaves `result["tags"].append(...)` reaching the store,
  which is the same defect one level down and the level at which it is
  actually hit. This covers the ranked read, the id-keyed read,
  `add_vectors()`, `update_metadata()` and `update_metadata_where()` —
  the last of which merged one `set_` into every matched row, so a
  nested value inside it was shared by the caller *and* by every row the
  filter selected.

  Two things worth knowing. Copies are taken per stored row and per
  *returned* row, never per scored candidate, so a filtered search over
  a large corpus does not pay for the rows it discards. And a caller
  that was relying on the old aliasing to mutate a store in place must
  now call a mutator — that path was never the contract on all four
  backends, but it did work on two of them.

- **`VectorStore.get_vectors()` is annotated to return what it returns.**
  The ABC declared `list[tuple[np.ndarray, dict | None]]`, but every
  backend yields `(None, None)` for an id it does not hold — that is the
  documented behaviour, and it is what keeps the result positionally
  aligned with `ids`. `FaissVectorStore` and `PgVectorStore` had each
  widened it locally; the ABC and `MemoryVectorStore` now agree with
  them. Type-checked consumers unpacking the vector may need a `None`
  branch they always needed at runtime.

- **A failed `.meta` write left a FAISS store unloadable.** The index
  file and its side-car were each written directly over their targets,
  so a `.meta` that failed to serialize — an unpicklable value in
  consumer metadata, a full disk — had already consumed the index write,
  and left behind a truncated side-car that the next reader could not
  load at all (`EOFError`). It also stranded the instance's identity
  stamp on a file it had itself replaced, so every later `save()` raised
  `ConcurrencyError` about a conflict that never happened. Both files
  are now written to scratch siblings and renamed into place only once
  every write has succeeded, so a failure leaves the previous state
  intact. `MemoryVectorStore` writes the same way, for the same reason.

  Renaming is atomic per file but not across the pair, so the second
  rename can still fail once the first has landed — leaving this
  instance as the file's last writer with no stamp to show for it, and
  every later `save()` refusing over its own write. A store now
  re-reads the file's identity on the way out of a failed publish while
  staying dirty, so the next `save()` or `close()` retries instead of
  demanding `save(force=True)` — a call that exists to discard another
  writer's rows, and no way to recover from a failure you caused
  yourself.

- **Overlapping `save()` calls on a *single* store could raise
  `ConcurrencyError` against themselves.** The staleness check and the
  write it guards are two operations on a worker thread, so an autosave
  overlapping `close()` — or a bare `asyncio.gather` — could have one
  save stat the other's half-written file. Saves are now serialized per
  instance; cross-instance conflict, which is what the check exists to
  detect, is unaffected.

- **`FaissVectorStore.search(filter=...)` applied the filter after the
  index had already truncated to `k`.** A filtered search returned only
  the matching rows that happened to fall inside the global top-`k`
  window — frequently none of them — while `count(filter=...)` reported
  the full number the store held, so a populated store simply retrieved
  nothing and said nothing about it. It now selects the matching rows
  from `metadata_store` and scores them directly from the vector
  side-car, returning a full `k` whenever `k` rows match, in the order
  and with the scores an unfiltered search would have given them. This
  matches `MemoryVectorStore` and `PgVectorStore`, which is what the
  cross-backend semantics doc already promised.

  Most visible on a store configured with a `domain_id`, where the
  scope is composed into every call and so every search was a filtered
  one, and worst where the configured tenant is small relative to its
  co-tenants. But no scoping was needed to reach it: any
  caller-supplied `filter=` whose matches sat outside the unfiltered
  top-`k` was affected.

  Filtered search is exact on every index type as a result, including
  `hnsw` and `ivfflat`, and no longer uses the index — see
  `VECTOR_FILTER_SEMANTICS.md` for what that costs and when to prefer
  `pgvector`. Unfiltered search is unchanged.

- **`is_default_backend()` reads the default's aliases when given a
  registry.** Without one it compares names, so a config spelling the
  default as `mem` read as naming something else and its caller took the
  non-default branch — for a backend the factory resolves to the same
  class. Pass the registry you would have built through wherever the
  answer selects between code paths rather than between log lines; alias
  identity is then the registry's answer rather than a second list here.

- **`backend_info()` called a metadata-less backend unrecognised.** It read
  metadata and treated a falsy result as "never heard of it", which is the
  wrong answer for a backend declared unavailable without any —
  `declare_unavailable` accepts `metadata=None`. The one state this
  function exists to describe was reported as a typo. It now asks the
  registry whether the name is known.

- **A `vector_store` section for an uninstalled backend is checked against
  its real schema where that schema is reachable.** The resolver returned
  `SKIP_VALIDATION` for every backend whose driver was absent, which made
  *which checks a config gets* a property of the machine reading it. Every
  optional store here guards its driver behind a module-level flag and so
  imports without it; the schema is read off that class, through the same
  loader the construction path uses. `SKIP_VALIDATION` is now reserved for
  a backend whose module genuinely cannot be imported. (No config changes
  verdict today — `from_dict` is permissive enough that both answers accept
  the same sections — so this pins the answer rather than tightening it.)


- **The async `sqlite` backend recorded `requires_install: False`.** It is on
  `aiosqlite`, which ships in the `sqlite` extra; only the sync variant is on
  stdlib `sqlite3`. The two shared a metadata block that described the sync
  one, so the async backend reported needing no installation while having a
  driver that can be absent.

- **`SyncPostgresDatabase.close()` closed nothing.** The body set
  `_connected = False` and carried a comment giving the reason — "PostgresDB
  manages its own connections via context managers". That is false: psycopg2's
  `with conn` is a transaction scope, not a close, and the `PostgresDB` it
  wraps had no `close()` to delegate to. Nothing surfaced as exhausted
  connections because CPython reclaimed each connection when the frame exited,
  which left the method's own contract unmet rather than satisfied. It now
  closes the `PostgresDB` it owns, through `close_if_owned_sync`.

  The same change removes a full TCP+auth handshake from every CRUD operation:
  `PostgresDB` now reuses one connection, so a read and a write share a
  backend instead of each opening their own.

- **Attributes assigned `None` in `__init__` were typed as `None`.**
  `SyncPostgresDatabase.db` and `.query_builder` are both set in `connect()`,
  so mypy read each as `None`-typed and every use of them — `self.db.query(...)`,
  `self.query_builder.build_search_query(...)` — as an error against `None`,
  with the surrounding bodies written off as unreachable and therefore never
  checked at all. Both now declare the type they hold.

- **`vector_search` passed a pandas `Series` where a row `dict` was declared.**
  Its three sibling loops convert with `.to_dict()` first; this one did not.
  It survived because a `Series` answers `.get`/`in`/`[]` the way a mapping
  does, and would have stopped surviving the moment the shared serializer used
  anything a `Series` implements differently. All four sites now go through one
  conversion helper.

- **Connection parameters lost their types at a dict boundary.** The five
  values passed to `PostgresDB` are individually typed on the config class, but
  collecting them into an unannotated dict joined them to `object` — which
  neither `PostgresDB` nor `validate_database_name` accepts. The dict is now
  declared per key, and read by key rather than through `.get(key, default)`:
  the declaration is total and every key is populated, so each default was
  unreachable while reading as though it still applied.

- **`connect()` replaced its `PostgresDB` without closing the first one.** When
  the initial connection fails because the target database does not exist and
  `ensure_database` is enabled, the database is created and the connection
  reopened — which rebound the attribute and left the previous object to the
  garbage collector. It holds no live connection on that path, so nothing
  leaked in practice, but it was the one construction path not applying the
  ownership rule the rest of the class follows.

## v0.8.0 - 2026-08-11

### Added

- **`dataknobs-data[chroma]`, `[faiss]`, and `[pgvector]` extras** splitting
  the former all-or-nothing `vector` extra, so a consumer can install just
  the vector store it uses. Previously reaching FAISS or pgvector also
  pulled chromadb, which carries an unfixed pre-authentication code
  injection advisory (GHSA-f4j7-r4q5-qw2c / PYSEC-2026-311, CVSS 9.3, no
  upstream fix as of 2026-08-05 and no release above the affected range);
  `dataknobs-data[faiss]` and `dataknobs-data[pgvector]` avoid it
  entirely. `[pgvector]` carries `asyncpg`, the transport `PgVectorStore`
  lazy-imports.

  `dataknobs-data[vector]` is retained as a roll-up of all three and
  resolves to exactly the same distributions as before, so existing
  installs are unaffected; `dataknobs-data[all]` likewise continues to
  include every backend, chromadb among them. For the full install minus
  Chroma, use
  `dataknobs-data[postgres,sqlite,duckdb,s3,elasticsearch,parquet,faiss,pgvector]`.

  This package owns the vector stores, so these extras are the single
  source of their dependency floors — `dataknobs-fsm` and
  `dataknobs-bots` now forward to them instead of re-pinning the drivers.

### Security

- **Postgres now maps a non-duplicate constraint violation at all.** It caught
  `UniqueViolation` by type and nothing else, so a `NOT NULL`, `CHECK`, or
  `FOREIGN KEY` failure was never mapped: the raw driver exception propagated,
  a caller's invalid write came back as a 500 rather than the 422 it is, and
  psycopg2's text for a `CHECK` violation carries a `DETAIL:` line quoting
  **the failing row** — so the rejected value travelled in an exception a
  library caller is likely to log.

  Both drivers expose the distinction as an exception type, which is more
  precise than the text matching the other backends need, so the split is
  `UniqueViolation` / `UniqueViolationError` first and then the
  `IntegrityError` / `IntegrityConstraintViolationError` base — catching the
  base rather than naming each constraint kind, so one not listed still maps.
  Six sites, sync and async.

  The structural guard did not catch this because it checks that no backend
  *constructs* `RecordValidationError` itself, and postgres constructed
  nothing. Elasticsearch, the other backend that never calls the factory, is
  correct as-is: it has no `NOT NULL` or `CHECK` to violate, so a version
  conflict is the only write rejection it can produce.

- **A constraint violation no longer answers with the schema that rejected
  it.** All eight SQL sites that map a non-duplicate constraint error built
  their message from `str(driver_exception)`, and a driver names what it
  enforced: `NOT NULL constraint failed: records.tenant_secret`.
  `RecordValidationError` is a `dataknobs_common.exceptions.ValidationError`,
  which the `dataknobs-bots` API layer returns to the caller as a 422 with its
  message shown — so a rejected write published a piece of the storage schema.

  The eight sites now raise through one `constraint_violation_error()` factory
  in `sql_base.py`, beside the `is_duplicate_key_error()` predicate they
  already share: eight independent copies is why one wrong message was wrong
  in eight places. The message names the record where the caller knows which
  one it was (a batch write does not — the driver reports the constraint, not
  the row), and the driver's text stays on `__cause__`, in the traceback a
  library caller sees and in the line the API handler logs.

- Re-verified the accepted GHSA-f4j7-r4q5-qw2c / PYSEC-2026-311
  (CVSS 9.3, pre-authentication code injection via the
  `/api/v2/.../collections` endpoint) against the `chromadb>=1.0.0`
  floor (extra: `vector`), flagged again at the floor resolve. As of
  2026-08-05 the advisory still affects 1.0.0–1.5.9 with no upstream
  fix, and 1.5.9 remains the latest release, so no floor bump can
  clear it. Risk accepted unchanged: `ChromaVectorStore` uses
  `chromadb.Client` / `chromadb.PersistentClient` only — never
  `HttpClient` or server mode — so the vulnerable endpoint is not
  reachable from this package. The inline floor comment records the
  refreshed verification date.

### Changed

- **`ConversionOptions.merge_metadata` documents its list behavior, and no
  longer carries its own merge.** The behavior is unchanged — the
  implementation was structurally identical to `dataknobs_config.deep_merge`,
  which it now calls — but the contract was only ever implied by the code.
  Nested dicts merge key by key; every other value, **lists included**, is
  replaced by the second argument's rather than extended.

  It was one of five independent copies of the same function in the
  workspace, one of which had drifted to extending lists and produced merged
  configurations that neither input described. Consolidating the copies is
  what keeps that from happening again; `tests/test_deep_merge_agreement.py`
  fails if any entry point stops agreeing.

  Non-mutation is unchanged and remains a *top-level* guarantee — the copy is
  shallow, so the result shares nested values with both inputs. See
  `dataknobs_config.deep_merge` for the exact contract.

## v0.7.0 - 2026-07-29

### Added

- **Consent-gated access for per-user state sections.** A section declaring a
  `consent_scope` now refuses reads and writes until the user grants that scope.
  `UserStateStore` / `AsyncUserStateStore` gain `grant_consent` /
  `revoke_consent` / `has_consent`; a direct `get_document` / `query` / write to
  an ungranted consent-scoped section raises
  `dataknobs_common.exceptions.ConsentRequiredError`, and `snapshot()` omits the
  section rather than raising. Grants are per `(user, scope)` — one grant unlocks
  every section sharing the scope — and are stored in a reserved,
  coordinator-managed `consent` document section, so `consent` is now a reserved
  section name (declaring it raises `ConfigurationError`). The reserved section
  is unreachable through the content API (`get_document` / `put_document` /
  `query` / `add_record` on `"consent"` raise `ConfigurationError`), so grants
  cannot be forged or the ledger clobbered by writing it directly — they flow
  only through `grant_consent` / `revoke_consent`. Revocation is block-only (the
  stored data is left in place for a later re-grant); erasure via `clear()` is
  never consent-gated. Sections that declare no `consent_scope` are unaffected.
- **Retention pruning for per-user state sections.** A **collection** section
  declaring a `retention_days` window ages out records whose `_written_at` stamp
  is older than the window. `UserStateStore` / `AsyncUserStateStore` gain
  `prune(user_id, section=None)` (deletes the expired records and returns the
  count — with `section=None`, across every windowed collection section) that a
  consumer schedules on its own cadence. A new `prune_on_query` config flag
  additionally prunes a windowed section's expired records for the queried user
  before a `query` returns (off by default). Retention time is measured against
  an injectable `now` clock (a `Callable[[], datetime]`, defaulting to
  wall-clock UTC). A `retention_days` on a **document** section — which holds one
  evolving record per user and never expires — is rejected at config load with
  `ConfigurationError`. A non-positive `retention_days` (zero or negative) is
  likewise rejected at config load — such a window would mark live records as
  already expired and delete them, so a mis-signed window is caught at the
  boundary rather than silently destroying data. Retention expiry is fail-safe:
  a record whose `_written_at` stamp is missing, unparseable, or not comparable
  to `now` (an aware/naive timezone mismatch between the stamp and the injected
  clock) is treated as not-expired and left in place rather than crashing the
  prune. Pruning is data minimization and, like `clear()`, is never
  consent-gated.
- **Deletion and erasure of per-user state now emit a metadata-only delta
  event** (`user_state:section_deleted`), a sibling of the existing
  `user_state:section_written` topic. `delete_record`, `prune`, and `clear` each
  fire one `op`-discriminated event (`op` = `"delete_record"` / `"prune"` /
  `"clear"`) when they actually remove data, so a consumer can build a deletion
  or erasure audit trail from the event stream. The payload is metadata-only by
  construction — deletes are by id, so no section value is ever emitted (a
  `SENSITIVE` section is safe) — and a whole-user `clear` reports
  `section = None`. Single deletes carry the `record_id`; bulk deletes carry a
  `count`. A section-less `prune(user_id)` (which ages out several windowed
  collections at once) additionally carries a `sections` map — the
  `{section_name: removed_count}` split — so an erasure-audit consumer can
  attribute the deletions while `count` stays the total. Nothing fires when
  nothing was removed. The event rides the same
  callback registry and optional `EventBus` fan-out as the write topic (the sync
  store still rejects an injected `event_bus`; in-process callbacks fire on both
  variants).
- **Persisted append-only audit log for per-user state.** Setting
  `enable_event_log` registers a reserved `events` collection section and appends
  one metadata-only record to it after every data write (`put_document` /
  `add_record` / `update_record`) and scoped deletion (`delete_record` / `prune`).
  `UserStateStore` / `AsyncUserStateStore` gain `query_events(user_id, query=None)`
  to read the trail; the reserved section is walled off from the content API
  (`events` is now a reserved section name — declaring it raises
  `ConfigurationError` — and reading or writing it through `get_document` /
  `query` / `put_document` / `add_record` raises), so audit entries cannot be
  forged or clobbered and are appended only by the coordinator. Each record
  stamps the operation metadata under `op` / `op_section` / `op_record_id` /
  `op_count` / `op_sections` keys — never a section value, so a `SENSITIVE`
  section's contents cannot leak into the log — with the record's own
  `_written_at` as the audit timestamp. A consent-refused write logs nothing (the
  gate raises before the write); whole-user `clear` (right-to-erasure) appends
  nothing (the log is erased with the user, and re-materialising a record would
  defeat the erasure — the ephemeral `user_state:section_deleted` event still
  fires). A new `event_log_retention_days` config field bounds the log through
  the ordinary section-less `prune` sweep (unbounded until `clear` when unset;
  positive-only, rejected at config load otherwise). The log is disabled by
  default; a `query_events` on a store without `enable_event_log` raises
  `ConfigurationError`.
- **Schema versioning with lazy on-read migration for per-user state.** Each
  section carries a schema `version` (stamped onto every record as
  `_section_version`); when a read surfaces a record behind that version, the
  section's registered upgrader chain rewrites its payload forward in memory
  before returning it (`get_document` / `query` / `snapshot`). Consumers register
  pure per-version upgraders through a new `register_section_migrator(section,
  from_version, fn)` / `section_migrators` registry (module
  `dataknobs_data.user.migration`; `SectionMigrator`, `register_section_migrator`,
  and `section_migrators` are re-exported from `dataknobs_data`). Migration is
  in-memory by default and preserves each record's `_written_at`, so a read never
  resets the retention clock; a new `persist_migrations` config flag writes the
  upgrade back once under a compare-and-set guard, skipping the write (and
  returning the in-memory upgrade) when a concurrent write won the guard. A record
  stamped newer than the running section version passes through un-migrated with a
  warning (rollback fail-open); a missing step in the upgrade chain raises
  `ConfigurationError` at read. An upgrader is a pure `Callable[[Mapping],
  Mapping]` that sees the consumer payload only: the coordinator's scope stamps
  are stripped from both the upgrader's input and its output, so an upgrader can
  neither read nor forge `_section_version` / `_written_at` / `tenant_id`. A
  section `version` must be a positive integer (a zero or negative version is
  rejected at config load). The reserved `consent` and `events` sections are
  never migrated.

### Changed

- **BREAKING: `create()` resolves the storage id uniformly with
  `create_batch()` across all backends.** A caller-supplied `record.id` is
  honored as the storage key (a fresh UUID is minted only when the record
  carries no id), and a colliding id fails closed with `DuplicateRecordError`.
  Previously the SQLite (async) and DuckDB (sync + async) `create()` paths
  minted a fresh UUID and ignored a payload `id` / `record_id` data field,
  diverging from their own `create_batch()` and from the other backends. A
  record whose data carries an `id` (or `record_id`) field is now keyed under
  that value on every backend and both write methods. If you relied on `create()`
  minting a fresh id while a payload `id` field stayed pure business data, store
  that identifier under a non-`id` field name (recommended — it is also
  queryable) or set `record.storage_id` explicitly; use `upsert` for
  insert-or-overwrite. See the Record ID Architecture guide.

- **Minor: `upsert(Record(id=""))` now mints a fresh storage id instead of
  keying the record under `""`.** A falsy (empty-string) record id is treated as
  absent and minted via `_generate_id()` on the single-`upsert` path — the same
  rule `create()` and `upsert_batch()` already applied — so single upsert no
  longer diverges from batch upsert on an empty-string id (and honors a
  `_generate_id()` override). A non-empty id is unaffected; if you deliberately
  relied on `""` as a storage key, set `record.storage_id` explicitly.

- **Minor: `upsert()` no longer mutates the caller's record.** The
  resolved/minted storage id is now stamped onto a copy (matching `create`,
  `create_batch`, and `upsert_batch`, which were already copy-first), so no write
  method mutates the record object the caller passes in — including the buffered
  transaction's `tx.upsert(id, record)` staging. Read the resolved id from
  `upsert()`'s return value (`new_id = db.upsert(record)`); a caller that
  previously relied on `record.storage_id` being stamped in place after the call
  must use the return value instead.

### Added

- **Overridable `_generate_id()` storage-id mint hook.** When a record carries
  no caller id, `create()` / `create_batch()` **and** `upsert()` /
  `upsert_batch()` mint the storage id through a single overridable hook —
  `_generate_id()`, defined once on the shared `RecordStorageMixin` that both
  `SyncDatabase` and `AsyncDatabase` inherit, and routed through by every mint
  fallback (the base write-keying helper, the shared single-`upsert`
  id-resolution preamble, the SQL create/upsert query builders, and the Postgres
  / Elasticsearch create/upsert paths). Override it on a backend subclass to
  supply a custom storage-id scheme (ULID, Snowflake, monotonic/deterministic,
  tenant-prefixed) uniformly across every create and upsert path, instead of
  patching each backend. `update()` / `update_batch()` never mint (every
  `update` takes an explicit id). The default remains a random UUID4, so existing
  behavior is unchanged; a caller-supplied `record.id` is always honored and
  never routes through the hook.

- **`UserStateStore` / `AsyncUserStateStore` — per-user cross-session state
  coordinator.** A config-driven, backend-agnostic coordinator for a user's
  state across sessions, built on the `SyncDatabase` / `AsyncDatabase`
  compare-and-set surface. It scopes an injected database by `(namespace,
  tenant, user_id, section)` over user-defined `document` sections (one record
  per user, addressed by a derived deterministic id) and `collection` sections
  (many records per user, read by filter). Writes are optimistic-concurrency
  aware (`expected_version` compare-and-set, advertising
  `Capability.CONDITIONAL_WRITE`), tenant-scoped when a `BoundTenantContext` is
  injected (`Capability.TENANT_SCOPED_STATE`, with explicit-filter-wins admin
  reads), and emit metadata-only delta events through an in-process callback
  registry; the async variant can additionally fan them out to an `EventBus`
  (the sync variant rejects an injected `event_bus`, since `EventBus.publish`
  cannot be driven safely from the sync fire path). `snapshot()` returns a
  whole-user view (omitting `SENSITIVE` sections by default) and `clear()`
  erases a user's state across all sections via a single batched delete. Record
  identity is coordinator-owned — document ids derive from the scope tuple,
  collection ids are backend-generated, and a payload carrying a storage-identity
  key (`id`, `storage_id`, `_id`, `record_id`) is rejected. `record_version()` is
  scope-checked (an out-of-scope id returns `None`). Declared sections are
  validated at config-load time (unique, non-empty names). The opaque `user_id`
  is only ever a hash input or a filter value, never split into a delimited key,
  so ids containing `/` or `://` are structurally safe. Sync and async variants
  share the same scoping helpers; a config-built backing database is owned and
  closed, an injected one is caller-owned and left open. See the User State
  guide.

## v0.6.2 - 2026-07-20

### Added

- Storing a record whose data carries a top-level field named `id` now emits a
  one-time diagnostic signal. That name is reserved for the record's storage key,
  so a `Filter`/`SortSpec` on `id` resolves to the storage key and the stored
  value is unreachable by query — a silent footgun with no error and zero
  matching rows. Every record-persisting verb of every backend — single
  (`create`, `upsert`, `update`) and bulk (`create_batch`, `upsert_batch`,
  `update_batch`) — passes through the signal, so it fires regardless of which
  verb or backend performs the write. The signal is `DEBUG` by default (silent
  under normal configuration, visible when a consumer raises verbosity to
  investigate an empty result) and promotes to `WARNING` when the environment
  variable `DK_WARN_SHADOWED_ID` is set to `true` (case-insensitive; any other
  value keeps it at `DEBUG`). It fires at most once per process (thread-safe
  under concurrent writers). Rename the field to an entity-qualified name such
  as `entity_id` to keep it queryable.
- `RESERVED_KEY_FIELD` and `is_storage_key_field(field)` exported from
  `dataknobs_data`: the single source of truth for the reserved query/sort field
  name (`id`) that every backend routes to a record's storage key. Every
  backend's filter and sort translation now consults the predicate instead of
  comparing `field == "id"` inline, so all backends agree on the reserved name by
  construction. Code that generates field names can assert against
  `is_storage_key_field()` to avoid the storage-key shadowing footgun.
  Consolidating the flat-`Query` and native-SQL/Elasticsearch translation sites
  is behavior-preserving; the in-memory boolean-query path is additionally
  corrected — see Fixed.

### Fixed

- Boolean `ComplexQuery` (OR / NOT / nested logic) resolved on the shared
  in-memory scan path — the memory backend, and any backend without native
  boolean-query translation — now routes the reserved `id` field to the record's
  storage key for both filtering and sorting, matching the flat-`Query` path and
  the native-SQL/Elasticsearch translations. Previously a boolean query such as
  `Filter("id", EQ, key) OR ...` (and `SortSpec("id", ...)` on such a query)
  resolved `id` to a shadowed `data["id"]` value on this path, silently diverging
  from the SQL backends that resolved it to the storage key — the exact
  cross-backend drift the reserved-name consolidation exists to prevent.

## v0.6.1 - 2026-07-18

### Changed

- `allocate` / `allocate_sync` now retry on a colliding id raised **anywhere in
  the read-compute-create cycle** — including from inside the caller's `build`
  callable — not only from the final `create()`. A `build` that probes-and-creates
  and raises `DuplicateRecordError` triggers a re-build and retry, so a collision
  is handled wherever it surfaces. The public signatures, the `max_attempts`
  bound, and the fail-closed exhaustion behavior are unchanged.

## v0.6.0 - 2026-07-15

### Added

- `allocate` / `allocate_sync`: create a record under a caller-computed monotonic
  key (a version, a sequence number, any derived key), retrying on a colliding id
  so concurrent allocators each land a distinct next key. A caller-supplied
  `build` callable does the fresh read, next-key computation, and record
  construction; the helper re-runs it on `DuplicateRecordError` and retries. A
  bounded create-on-conflict loop over the atomic `create()` — a single
  uncontended allocation makes exactly one attempt, and after `max_attempts`
  (default 16) collisions it re-raises the last collision, fail-closed. Key-agnostic
  (never mints or mutates ids) and backend-agnostic (composes over the shared
  `create()` contract, no backend-specific path).

- `Operator.STARTS_WITH`: a literal, case-sensitive, escape-safe prefix
  predicate — a `_` or `%` in the prefix matches literally, unlike `LIKE`. It
  pushes down to the backend query engine where available (a SQL range or
  `LIKE ... ESCAPE`, an Elasticsearch `prefix` query) and scans in memory
  otherwise. Record identifiers are first-class query targets: `Filter("id", ...)`
  resolves to the storage key — equality, membership, literal prefix, and
  sorting — uniformly across all backends, including records whose id the
  backend minted at write time, so a store that encodes hierarchy into its keys
  can scan a subtree with one filter instead of fetching coarsely and filtering
  in Python.

### Changed

- The string-matching operators (`LIKE`, `NOT_LIKE`, `REGEX`, `STARTS_WITH`)
  match **only string values**, consistently across the SQL and in-memory
  backends. On the SQL backends (SQLite, PostgreSQL, DuckDB) a `data`/`metadata`
  JSON field whose stored value is not a string (a number, boolean, …) no longer
  matches these operators via text coercion — the push-down carries a
  JSON-string-type guard so it agrees with the in-memory backends'
  (`memory`/`file`/`s3`) `isinstance(str)` contract. The `id` storage key is a
  real string column and is unaffected. The async S3
  backend's `LIKE` now uses the same anchored, case-insensitive SQL-wildcard
  semantics as the sync S3 backend and every other backend (previously a
  case-sensitive substring test), and its non-`LIKE` operators — `BETWEEN`,
  `EXISTS`, `REGEX`, `STARTS_WITH` — are now honored rather than silently
  dropped.

- The `Migrator`'s batched write path (`migrate()`) now writes each batch
  through the target's native bulk verbs — `create_batch` for `insert`
  (fail-closed on a colliding id) and `upsert_batch` for `upsert` — with a
  graceful per-record fallback that preserves per-id progress accounting and
  `on_error` semantics when a batch write fails. This matches the throughput of
  the streaming write path (`migrate_stream()` / `migrate_parallel()` /
  `migrate_async()`), which already rode the bulk verbs, so a from-empty
  `insert` or a `make-target-match-source` `upsert` migration issues one bulk
  call per batch instead of a round trip per record. The `insert` bulk
  fast-path is used only where `create_batch` is atomic on a collision (memory,
  SQLite, PostgreSQL, DuckDB, file); on a backend whose bulk create is
  non-atomic (Elasticsearch's per-item bulk API, the S3 per-record loop),
  `insert` routes per-record — mirroring what those backends' own `stream_write`
  already does — because riding the bulk verb there would let the per-record
  fallback re-write the rows a failed bulk call had already durably written and
  count them as spurious duplicate failures. The `skip` policy still writes per
  record (a whole-batch verb cannot skip one duplicate while inserting the
  rest). Conflict-policy behavior and `MigrationProgress` accounting are
  identical to the prior per-record path on every backend.

- `AsyncDatabase.transaction()` / `begin_transaction()` now commit a buffered
  transaction of **any** composition all-or-nothing on a transactional backend
  (SQLite, PostgreSQL, DuckDB). `BufferedTransaction.commit()` coalesces
  consecutive same-kind operations into a single `create_batch` / `upsert_batch`
  / `delete_batch` call and runs every coalesced batch inside **one** native
  transaction, so a commit is atomic whether the buffer is single-kind (all
  creates, all upserts, or all deletes) or spans several kinds (e.g. mixed
  create + delete, or create + upsert) — a mid-flush failure rolls the whole
  commit back rather than partially persisting. `BufferedTransaction.is_atomic`
  reports `True` for any composition on a transactional backend. Non-transactional
  backends (memory/file/s3/elasticsearch) are unchanged — a multi-kind commit
  there stays best-effort per batch and `is_atomic` is `False`.

- `create()` is now a defined atomic insert across every backend: a colliding
  id raises `DuplicateRecordError` instead of silently overwriting the existing
  record (memory, file, S3, Elasticsearch) or raising a bare `ValueError`
  (SQLite, DuckDB). This removes the racy `exists()`-then-`create()` workaround
  consumers needed for collision-safe inserts. `DuplicateRecordError` subclasses
  `ValueError`, so existing callers that caught the former `ValueError` on a
  duplicate id are unaffected. On S3 the guarantee is enforced with a
  conditional PUT (`If-None-Match`) and therefore holds against any S3
  implementation that honors conditional writes (real AWS S3, recent
  LocalStack); both a pre-existing key (412) and a concurrent conditional-write
  race (409) fail closed as `DuplicateRecordError`, while older stores that
  ignore the header degrade to last-writer-wins. `create_batch()` honors the same
  contract uniformly across every backend — see the `create_batch()` and
  streaming entries below.
- On the SQL backends (SQLite, DuckDB), `create()` now distinguishes a
  duplicate-id collision from other column-constraint violations: only a
  primary-key collision raises `DuplicateRecordError`, while a `NOT NULL` or
  `CHECK` violation on the stored data surfaces as `RecordValidationError`
  rather than being mislabeled as a duplicate id.

### Added

- `DuplicateRecordError` (exported from `dataknobs_data`), raised by `create()`
  on a duplicate id. Subclasses both the data-layer `ConcurrencyError` and
  `ValueError`; carries the colliding id in `.id` and `context={"id": ...}`.
- `ConcurrencyError.__init__` accepts an optional keyword-only `context` mapping
  (backward-compatible), so concurrency conflicts can carry structured detail.
- `get_version(id) -> str | None` on `AsyncDatabase` / `SyncDatabase` and every
  backend: returns an opaque, backend-local optimistic-concurrency token for a
  stored record (or `None` if the id does not exist). The token is native where
  the store provides one — an in-memory per-instance monotonic sequence,
  PostgreSQL `xmin`, Elasticsearch `_seq_no`/`_primary_term`, S3 `ETag` — and a
  deterministic content hash of the stored record on the file, SQLite, and
  DuckDB backends. The in-memory token is ABA-safe on every path, including a
  delete→recreate at the same id (the sequence value is never reused). Treat it
  as opaque; it is not comparable across backends.
- An opt-in, keyword-only `expected_version` parameter on `update()`,
  `upsert()`, and `delete()` across both base contracts and all 14 backends.
  Passing a token read from `get_version()` turns the write into a
  compare-and-set: it proceeds only if the record's current token still matches,
  otherwise it raises `ConcurrencyError` (carrying `id` / `expected_version` /
  `actual_version` in `.context`) instead of last-writer-wins. The
  compare-and-set is enforced atomically where the store supports it —
  PostgreSQL `WHERE ... AND xmin = …`, Elasticsearch `if_seq_no`/`if_primary_term`,
  S3 `If-Match` (on both the conditional PUT and the conditional DELETE) — and
  the in-process content-hash backends serialize the check within a single
  connection/instance. A conditional `update()` never inserts (a missing record
  returns `False`); a conditional `delete()` never conflicts on an absent id (a
  missing record returns `False`); a conditional `upsert()` never inserts (a
  missing record is itself a conflict and raises). Omitting `expected_version`
  leaves all three operations byte-identical to prior behavior (unconditional
  last-writer-wins). `upsert()` applies the update through the backend's own
  atomic guard and acts on its result, so a concurrent delete cannot make it
  report success without writing.
- The database backends advertise their optional consistency features through
  the `CapabilityContract` surface: `AsyncDatabase` / `SyncDatabase` and every
  concrete backend report `Capability.CONDITIONAL_WRITE`. A consumer can query
  `db.supports(Capability.CONDITIONAL_WRITE)` (or use `require_capability`)
  before relying on `expected_version` compare-and-set, instead of knowing the
  backend matrix out-of-band. The advertisement is uniform because every
  backend enforces the contract; the ABA nuance of the content-hash backends
  is documented, not encoded as a separate capability.
- `Migrator` gains an `on_conflict` policy (`insert` / `upsert` / `skip`) for
  idempotent re-runs into a populated target. `insert` (the default) fails
  closed on a colliding id as before; `upsert` overwrites the target row;
  `skip` leaves the existing row and counts the id as skipped. The policy is
  threaded through all four migrate methods — `migrate()` and
  `migrate_parallel()` take it directly, `migrate_stream()` / `migrate_async()`
  read it from `StreamConfig`. Default behavior is unchanged.
- `ConflictPolicy` enum and `StreamConfig.on_conflict` field (exported from
  `dataknobs_data` and `dataknobs_data.migration`) carry the policy on the
  streaming path; every backend's `stream_write` honors it. `StreamResult`
  gains a `skipped` counter. The `insert` fast-path uses the backend's native
  batch write; `upsert` uses the native `upsert_batch` bulk verb (see below)
  with a per-record `upsert` fallback; `skip` writes one record at a time (a
  whole-batch verb cannot skip individual dupes while inserting the rest). An
  unknown `on_conflict` value is rejected when the `StreamConfig` is built.
- `upsert_batch(records)` on `AsyncDatabase` / `SyncDatabase` and every backend
  — the batch sibling of `create_batch`, with upsert (insert-or-overwrite)
  semantics: it honors a caller-supplied `record.id` (minting one only when
  absent), overwrites a colliding id (never raised, never skipped), returns ids
  in input order, and carries no version check (a whole batch cannot carry one
  optimistic-concurrency token). Native bulk fast-paths where the store has one
  — a single `INSERT ... ON CONFLICT (id) DO UPDATE` on SQLite, DuckDB, and
  PostgreSQL; a bulk index-by-id on Elasticsearch; a single file-rewrite (file)
  / single-lock pass (memory) — and the per-record ABC-default loop (per-key
  PUT) on S3, which has no cheaper bulk verb. The streaming `upsert` policy and
  the FSM `DatabaseResource.commit_batch` identity path both adopt it for batch
  throughput. `BufferedTransaction` gains a matching `upsert_batch` staging
  method; on commit a consecutive upsert run is coalesced into a single
  `upsert_batch` (atomic on transactional backends — see the Changed entry).
- `create_batch()` now fails closed on a colliding id across **every backend**,
  matching single `create()`: a colliding id — against an existing record or a
  duplicate within the same batch — raises `DuplicateRecordError`, and a
  caller-supplied `record.id` is honored (minted only when absent). The SQL
  backends (SQLite, DuckDB, PostgreSQL) previously minted a fresh id per record
  and ignored `record.id`; Elasticsearch overwrote (sync) or used server-assigned
  ids (async); memory/sync-file overwrote and async-file minted. On the
  transactional SQL backends the batch is atomic (a collision rolls back the
  whole INSERT — nothing written); on Elasticsearch the bulk API is per-item, so
  — exactly like a `create()` loop — non-colliding rows in the same batch may be
  written before the conflict is raised.
- The **streaming INSERT** path (`migrate_stream` / `migrate_async` and every
  backend's `stream_write` under the default `ConflictPolicy.INSERT`) now fails
  closed on a colliding id across **every backend** — recording it as a failure
  and preserving the source id, rather than writing a fresh-id row. A re-run into
  a populated target records the colliding ids as failures, matching the batched
  `migrate()` path. SQLite/DuckDB reach this through the tightened bulk
  `create_batch` plus a per-record `create()` fallback; PostgreSQL through its
  atomic `_write_batch` fast-path; S3, Elasticsearch-sync, and async-Elasticsearch
  through per-record `create()` (their non-transactional bulk write would
  otherwise double-write the already-written rows under the fallback). `upsert` /
  `skip`
  remain available for idempotent re-runs.

### Fixed

- The async Elasticsearch backend honors the full operator set — `REGEX`,
  `EXISTS`, `NOT_EXISTS`, `NOT_LIKE`, and the negations of `IN`/`BETWEEN` — in
  `search()`, `count()`, and `stream_read()`, matching the sync backend and the
  other backends. It previously translated only a subset inline, so those
  operators were silently dropped and the query fell back to matching every
  document. The async `search()` also accepts a `ComplexQuery` (AND/OR/NOT).
- Elasticsearch `LIKE`/`NOT_LIKE` uses SQL-wildcard (`%`→any, `_`→one),
  case-insensitive matching, consistent with the in-memory and SQL backends
  (the async backend previously did a case-sensitive substring match). Only `%`
  and `_` are wildcards: a literal Lucene metacharacter in the pattern (`*`,
  `?`, `\`) is now escaped so it matches verbatim, rather than leaking through
  as an Elasticsearch wildcard (an unescaped `*` previously matched anything).
  The case-insensitive `wildcard` form requires Elasticsearch ≥ 7.10.
  **Behavior change / migration:** a consumer that relied on the old
  Elasticsearch-only passthrough — passing a raw `*` or `?` in a `LIKE` pattern
  and expecting it to act as an Elasticsearch wildcard — must switch to the
  portable SQL wildcards `%` and `_`. For example `LIKE "*name*"` (which matched
  any value containing `name` on Elasticsearch, but matched a literal asterisk
  or errored on the other backends) becomes `LIKE "%name%"`, which now behaves
  identically on every backend.
- Elasticsearch `REGEX` matches the **full field value** (via the `.keyword`
  sub-field, case-sensitive), consistent with the in-memory (`re.search`) and
  SQL backends. It previously ran against the analyzed `data.<field>` path,
  where `regexp` matches a single lowercased token — so a pattern spanning a
  word boundary (e.g. `alice.*smith` against `"alice smith"`) matched nothing.
  `Filter("id", REGEX, ...)` was already full-value; data fields now agree.
  (Elasticsearch `regexp` is anchored and uses Lucene RegExp syntax, which
  differs from Python `re` — no `^`/`$` anchors, no look-around.)
- Elasticsearch raises `ValueError` for an operator its translator cannot
  express, rather than silently falling back to `match_all` (a dropped filter
  returning every document). This is the fail-loud counterpart to the operator
  convergence above; a caller that relied on the silent everything-match now
  gets an explicit error.
- `Filter("id", ...)` on Elasticsearch accepts the full operator set — prefix,
  range, membership, wildcard, regex — against the record's storage key, which
  is carried as a queryable `id` keyword field (a metafield-`_id` filter did not
  support range/prefix/wildcard). **Migration note:** because `id` filtering now
  targets the stamped top-level `id` field rather than the `_id` metafield, it
  only sees documents written with that field present. Every write path stamps
  it now, but a record indexed by an older version that did not stamp a *minted*
  `id` (one the backend generated because the record had none) is invisible to
  an `id` filter until the index is reindexed. `read(id)` is unaffected — it
  still fetches by `_id`.
- Elasticsearch vector and hybrid search pre-filters honor the full operator set
  and resolve `Filter("id", ...)` to the record's storage key; equality is exact
  (`term` on the keyword sub-field) rather than an analyzed match.
- `upsert(id, record)` now honors the explicit `id` when the record carries a
  *different* pre-set `storage_id` and the id does not yet exist. The base
  create-fallback previously wrote the new row under the record's own
  `storage_id` (and returned that id), silently discarding the explicit `id`
  argument; it now stamps the resolved id onto the record before the create so
  the explicit id is authoritative. Affects the backends that use the base
  `upsert` (SQLite, DuckDB, sync S3); backends overriding `upsert` (memory,
  PostgreSQL, Elasticsearch, file, async S3) already behaved correctly. The
  common paths —
  `upsert(record)`, and `upsert(id, record)` with a matching or absent
  `storage_id` — are unchanged.
- The async Elasticsearch backend's `create_batch()` and `upsert_batch()` now
  reconcile the bulk response per item, so a record whose bulk operation failed
  (e.g. a mapping error or version conflict) is no longer reported as written. A
  partial bulk failure previously returned every input id as successful — the id
  list is now filtered to the operations that actually succeeded, matching the
  sync backend's `_execute_bulk_index` reconciliation (extracted here into a
  shared `_extract_bulk_index_ids` helper used by both async bulk paths).
- Streaming now accounts a partial-batch failure honestly. When a batch write
  verb confirms fewer ids than the batch it was given — which a bulk backend
  can do (Elasticsearch reports per-item errors, so its `create_batch` /
  `upsert_batch` return only the ids that succeeded) — the unconfirmed records
  are counted as `failed` instead of silently vanishing from the tally, so
  `StreamResult.total_processed == successful + failed + skipped` holds. The
  shortfall is routed through `StreamConfig.on_error` once as an aggregate
  error (with a `None` record, since per-item identity is not available at this
  layer), placing the batch path on the same stop/continue contract as the
  per-record fallback: a configured handler decides whether to continue, and
  with no handler the stream quits on the first failing batch — the same
  fail-stop default a per-record failure already gets.
- Corrected the buffered-transaction documentation (the
  `dataknobs_data.transactions` module docstring and the Transactions guide) to
  stop directing consumers to a non-existent "backend-native transaction"
  primitive for cross-operation atomicity and connection-scoped isolation — the
  public API exposes no connection-scoped transaction beyond the buffered
  `db.transaction()` form. The docs now state the actual options: stage a single
  operation kind per transaction for all-or-nothing on a transactional backend,
  and use optimistic concurrency (`update` / `upsert` with `expected_version`)
  for a read-modify-write invariant.
- `Filter("id", ...)` on the async S3 and async Elasticsearch backends now
  resolves to the record's storage key, matching every other backend; it
  previously matched a data field named `id` (so an id filter returned the wrong
  rows or none). The async S3 backend now also honors `BETWEEN` / `NOT_BETWEEN` /
  `EXISTS` / `NOT_EXISTS` / `REGEX` filters it previously ignored — its filter
  matching now delegates to the shared `Filter.matches` matcher rather than a
  narrower inline operator switch.
- The sync Elasticsearch backend's `count()` now honors `BETWEEN` / `NOT_BETWEEN`
  filters. Its query translation is now shared with `search()` (both route
  through one per-filter translator); previously `count()` carried a separate
  translation that omitted these operators, so a `BETWEEN`-only count fell back
  to matching everything and returned the total instead of the filtered count.

### Notes

- The file/SQLite/DuckDB content-hash token is subject to the classic ABA
  limitation: an A→B→A mutation cycle yields the original token, so a stale
  conditional write in that exact scenario is not detected. The backends with a
  native monotonic version (memory counter, PostgreSQL `xmin`, Elasticsearch
  `_seq_no`, S3 `ETag`) are ABA-safe. The in-process content-hash backends
  enforce the compare-and-set within a single connection/instance; conditional
  writes are not hardened across separate processes/connections.

## v0.5.5 - 2026-07-07

### Changed

- The AWS session helper is now AWS-generic rather than S3-named, and has
  been relocated to `dataknobs-common` so every AWS consumer across the
  stack shares one implementation. The normalized session config is
  `AwsSessionConfig` (region, credentials, endpoint, retry/pool tuning)
  and now lives in `dataknobs_common.aws` alongside
  `create_aioboto3_session` and `clear_aioboto3_session_cache`. The
  S3-specific surface — `S3PoolConfig`, `create_boto3_s3_client`,
  `validate_s3_session` — stays in `dataknobs_data.pooling.s3`, which
  re-exports the generic names for import stability.
  `create_aioboto3_session` gains a keyword-only `warm_service` parameter
  (default `"s3"`) selecting which service's botocore data files are
  warmed off the event loop; the process-wide warmed-session cache is
  keyed by session kwargs **and** `warm_service`, so distinct services key
  to distinct warmed sessions.

### Deprecated

- `S3SessionConfig` is a deprecated alias for `AwsSessionConfig`.
  Importing it from `dataknobs_data.pooling.s3` emits a
  `DeprecationWarning`; the `dataknobs_data.pooling` package-root alias
  resolves without one (a permanent compatibility alias). Import
  `AwsSessionConfig` from `dataknobs_common.aws` (or the
  `dataknobs_data.pooling` package root) instead.

### Security

- Acknowledged GHSA-f4j7-r4q5-qw2c / PYSEC-2026-311 (CVSS 9.3, pre-auth
  code injection via the `/api/v2/...` endpoint) against the
  `chromadb>=1.0.0` floor (optional Chroma backend), flagged at the
  floor resolve by the `dependency-update` workflow. Affects chromadb
  1.0.0–1.5.9 with no upstream fix. Not exposed in this package:
  `ChromaVectorStore` uses `chromadb.Client` / `chromadb.PersistentClient`
  only (embedded/persistent modes), never `HttpClient` or server mode.
  The inline floor comment in `pyproject.toml` records the rationale.

## v0.5.4 - 2026-06-29

### Added

- **`AsyncDatabase` gains a buffered transaction capability.** A new
  `async with db.transaction(policy="strict"|"emulate") as tx:` context manager
  (and an explicit `await db.begin_transaction(...)` / `tx.commit()` /
  `tx.rollback()` form for staging writes across call sites) defers every write
  (`tx.create` / `tx.create_batch` / `tx.upsert` / `tx.delete`) until commit.
  Two guarantees: an exception before commit persists nothing on *any* backend
  (universal rollback), and on backends whose batch operations run inside a
  backend transaction — SQLite, Postgres, DuckDB — a commit whose staged ops
  reduce to a single coalesced same-kind batch (all creates, or all deletes,
  with no upserts) is all-or-nothing. A **mixed** create/delete or
  **upsert**-containing buffer commits as a *sequence* of independent batches
  and can partially persist if one fails mid-flush; the `tx.is_atomic` property
  reports — from the currently staged ops — which case applies, so a consumer
  needing cross-operation atomicity can branch on it. A new
  `db.supports_transactions()` flag reports which backends wrap a coalesced
  batch in a backend transaction; the three transactional backends return
  `True`, the rest (`memory`, `file`, `s3`, `elasticsearch`) return `False`.
  The `policy` argument chooses what happens on a non-transactional backend:
  `"strict"` (default, fail-closed) raises `CapabilityNotSupportedError`;
  `"emulate"` proceeds with best-effort buffer-and-flush. The handle does
  **not** provide in-transaction isolation or read-your-writes — buffered
  writes are invisible to reads until commit; consumers needing connection-
  scoped isolation should branch on `supports_transactions()` and use a
  backend-native transaction. The new `BufferedTransaction` type and the
  `VALID_TRANSACTION_POLICIES` tuple are exported from `dataknobs_data`.

## v0.5.3 - 2026-06-23

### Fixed

- **`ConnectionPoolManager` now reference-counts pools shared by DSN
  across instances on an event loop.** A new async `release_pool(config)`
  closes and evicts a pool only when its last holder releases.
  `AsyncPostgresDatabase`, `AsyncElasticsearchDatabase`, and
  `AsyncS3Database` `close()` now *release* their shared resource instead
  of either hard-closing it (Postgres — which closed the pool out from
  under sibling instances on the same DSN, so a sibling's `close()` broke
  the others' live connections) or never reclaiming it (Elasticsearch —
  whose pooled client was leaked until process exit under instance churn).
  Concurrent first-time connects on a cold key are serialized by a
  per-event-loop create lock so exactly one pool is created and the holder
  count stays sound under concurrency. Single-holder teardown is
  unchanged; the public `close()` signatures are unchanged.

## v0.5.2 - 2026-06-22

### Added

- **`PgVectorStore` can run against an externally supplied connection
  pool.** Build the store with
  `PgVectorStore.from_components(config, pool=shared_pool)` to hand it an
  asyncpg pool you manage. In that mode `initialize()` runs only the
  schema/table setup against the pool (it does not create one), and
  `close()` leaves the pool open for you to manage and retains the store's
  reference to it — so one pool can back several stores that are opened,
  closed, and reopened independently. Pool ownership is fixed at
  construction: re-initializing a self-owned store after `close()` rebuilds
  its pool, while an injected-pool store reuses the same caller-owned pool
  on reopen (it never fabricates one). The config / connection-string /
  `VectorStoreFactory` path is unchanged: it builds and owns its own pool
  and closes + drops it on `close()`.

### Fixed

- **The aioboto3 session-warm now pre-loads botocore's paginator model so
  `AsyncS3Database.stream_read` (and any aioboto3 paginator consumer) never
  stats for `paginators-1.json` on the event loop on first use.** Client
  creation loads the service model but not the paginator model, so the warm
  also builds a throwaway `list_objects_v2` paginator on its private
  worker-thread loop — the consumer's first real paginator build then reuses
  the session's loader cache instead of blocking the loop. The same pre-load
  also covers the knowledge S3 backend's paginating paths: every aioboto3
  client is built from one cached session (`_SESSION_CACHE`), so warming that
  session's loader once covers all of its paginator consumers.
- **`AsyncSQLiteDatabase.connect` and `AsyncDuckDBDatabase.connect` no
  longer block the event loop creating the database directory.** For a
  file-based database each created the parent directory with a synchronous
  `mkdir` on the running loop; the `mkdir` is now offloaded via
  `asyncio.to_thread`. Behavior is unchanged.
- **`SyncS3Database` and `AsyncS3Database` now sort search results
  correctly when a sort field holds a falsy value such as a numeric `0`.**
  The async backend's inline sort key coerced any falsy value (`0`,
  `False`, `""`) to an empty string, so sorting a numeric field whose
  values included `0` raised
  `TypeError: '<' not supported between instances of 'str' and 'int'`.
  Both S3 backends now apply sorting, `offset`/`limit` (including
  `limit=0`), and field projection through the shared
  `process_search_results` helper, so result ordering is consistent with
  every other backend and the duplicated per-backend logic is gone.

- **The async file database, the in-memory, Chroma, and FAISS vector
  stores, and the shared aioboto3 session factory perform their I/O
  without blocking the event loop.** Each held a synchronous, blocking
  transport behind an `async def`, stalling the loop for the duration of
  the call: `AsyncFileDatabase` ran its locked file load/save (including
  the inter-process `FileLock` acquire) on the loop on every CRUD
  operation, plus its temp-file cleanup on `close()`;
  `MemoryVectorStore.save`/`load` ran their `pickle` disk I/O on the loop
  (and `initialize` an `os.path.exists` stat before loading);
  `ChromaVectorStore` drove the synchronous chromadb client/collection
  directly; `FaissVectorStore.save`/`load` did blocking `faiss` index +
  pickle disk I/O; and `create_aioboto3_session` blocked on session
  construction plus aiobotocore's first-client botocore-data load. All
  now offload their blocking work via `asyncio.to_thread`, and the
  aioboto3 factory additionally warms the session's botocore caches
  off-loop so the first client creation by any consumer
  (`AsyncS3Database`, the SQS event bus, S3-backed knowledge storage) is a
  cache hit. Warmed sessions are cached process-wide by config, so
  consumers that build a session per instance rather than once at startup
  (e.g. a multi-tenant registry loading several runtime configs against
  the same bucket) warm once instead of once per instance. The async and
  sync file backends now share a single synchronous load/save
  implementation. FAISS in-memory `add`/`search`
  remain on the loop — they are CPU-bound and release the GIL internally,
  so offloading them buys nothing. No public signatures changed and no new
  runtime dependency was added (`asyncio.to_thread` is stdlib).

- **`MemoryVectorStore.save` and `FaissVectorStore.save` persist a
  consistent snapshot when a write runs concurrently with the save.**
  Because the disk write is offloaded to a worker thread, each `save()`
  now copies its in-memory state — the vectors / metadata / timestamp
  dicts, plus a clone of the FAISS index — on the event loop *before*
  handing off, so a `save()` that overlaps an `add_vectors` /
  `delete_vectors` records the state as of the `save()` call rather than a
  partially-mutated mix observed mid-serialization. `MemoryVectorStore.save`
  additionally handles a `persist_path` with no directory component (a bare
  filename), which previously failed with `FileNotFoundError`.

### Changed

- The `AsyncDatabase` and `VectorStore` base classes now document an
  async-transport contract — implementations use an async transport or
  offload blocking calls off the event loop (`asyncio.to_thread` /
  `aiter_sync_in_thread`), never blocking `open()` / `os` disk I/O behind
  an `async def`. ruff's `ASYNC` lint family now enforces this for the
  package.
- **`VectorStore.close()` now documents a backing-resource ownership
  contract.** A store that built its own backing resource (connection
  pool, client, session) closes it; a store handed an externally supplied
  resource leaves it open and releases only per-store state. Stores that
  build their backing resource internally (in-memory, FAISS, Chroma)
  satisfy this trivially; `PgVectorStore` honors the contract for its
  caller-supplied connection pool. No behavior change for stores that
  build their own resources.

## v0.5.1 - 2026-06-08

## v0.5.0 - 2026-05-26

### Changed

- **`StreamConfig` is now a frozen `StructuredConfig`.** It gains
  `from_dict()` / `to_dict()` and round-tripping; its existing
  `__post_init__` validation (`batch_size > 0`, `prefetch >= 0`,
  positive `timeout`) is preserved and now also fires on the
  `from_dict()` path. All `StreamConfig(...)` constructors are
  unchanged, but instances are immutable — construct a modified copy
  with `dataclasses.replace(...)` instead of assigning fields.
  `StreamResult` (runtime data) is unaffected.
- **All four vector stores now construct through typed configuration
  dataclasses.** `MemoryVectorStore`, `FaissVectorStore`,
  `ChromaVectorStore`, and `PgVectorStore` each grow a
  `<Backend>VectorStoreConfig` frozen dataclass (a
  `dataknobs_common.structured_config.StructuredConfig` subclass, in
  `dataknobs_data.vector.stores.config`) and are built via the
  `StructuredConfigConsumer` mixin. As a result, **`store.config` is now
  the typed config object, not a dict** — read fields as attributes
  (`store.config.dimensions`) rather than dict lookups. Every existing
  construction shape is preserved: `Backend(config_dict)`,
  `Backend.from_config(config_dict)`, and the `VectorStoreFactory` all
  continue to accept the same dict keys (projected onto the typed
  config), and a typed config may now be passed directly. The common
  keys (dimensions, metric, persistence, batch size, parameter
  sub-dicts, `domain_id`, and a nested `timestamps` config) live on the
  shared `VectorStoreConfig` base; each backend's leaf config adds only
  its own keys. Per-field validation (`id_type`, `index_type`,
  identifier shape, timestamp format) and pgvector connection
  resolution (`connection_string` / `DATABASE_URL` / `POSTGRES_*`)
  surface at construction exactly as before. Mixing a typed `config=`
  with loose keyword arguments raises `TypeError`.
- **The empty-list filter contract is now documented and enforced
  across backends.** An empty-list filter value (`{key: []}`) is an
  unsatisfiable predicate — it matches no record on any vector-store
  backend. This was already true (it backs the deliberate no-op
  `VectorMemory.clear()` uses for tenant isolation) but rested on four
  independent implementations with no shared test; a parametrized
  cross-backend conformance test now guards it so a regression in any
  one backend's filter translation is caught.
- **All 14 database backends now construct through typed configuration
  dataclasses.** Every `SyncDatabase` / `AsyncDatabase` backend (memory,
  sqlite, postgres, elasticsearch, s3, duckdb, file — sync and async)
  grows a `<Backend>DatabaseConfig` frozen dataclass (a
  `dataknobs_common.structured_config.StructuredConfig` subclass) and is
  built via the `StructuredConfigConsumer` mixin. As a result,
  **`db.config` is now the typed config object, not a dict** — read
  fields as attributes (`db.config.table`) rather than dict lookups
  (`db.config["table"]`). Every existing construction shape is preserved:
  `Backend(config_dict)`, `Backend.from_config(config_dict)`, and the
  `database_factory` / `async_database_factory` registries all continue
  to accept the same dict keys (projected onto the typed config), and a
  typed config may now be passed directly. Mixing a typed `config=` with
  loose keyword arguments raises `TypeError`.
- **The sync and async Postgres backends now share one configuration**
  (`PostgresDatabaseConfig`), the union of their parameters. This
  corrects prior drift where only the async backend honored `ssl`
  (see Fixed). `command_timeout` and the pool-size knobs
  (`min_pool_size` / `max_pool_size`) remain async-only — psycopg2 has
  no connect-time equivalent.
- **The sync and async S3 backends now emit a single bucket-required
  error message** (`"S3 backend requires 'bucket' in configuration"`);
  the sync backend previously raised a different string. Both report the
  same message now that bucket validation lives in the shared config.
- **Credential fields are redacted from config `repr`.** Building on the
  `StructuredConfig._SENSITIVE_FIELDS` mechanism in `dataknobs-common`,
  the backend and vector-store configs mask their credentials as `'***'`
  in `repr(config)` (and therefore in logs, tracebacks, and pytest
  failure output): `PostgresDatabaseConfig.password`,
  `AsyncElasticsearchDatabaseConfig.api_key` / `.basic_auth`,
  `S3DatabaseConfigBase.aws_access_key_id` / `.aws_secret_access_key` /
  `.aws_session_token` (inherited by both S3 backend configs),
  `PgVectorStoreConfig.connection_string`, and
  `ChromaVectorStoreConfig.openai_api_key`. `to_dict()` is never redacted,
  so round-trip construction is unaffected.

### Fixed

- **Sync Postgres backend now honors `ssl` configuration.** Previously
  only `AsyncPostgresDatabase` applied `ssl`; `SyncPostgresDatabase`
  silently ignored it. The sync backend now translates the asyncpg-native
  `ssl` value to a psycopg2 `sslmode` (`str` → that mode, `True` →
  `"require"`, `False` → `"disable"`); an unsupported value such as an
  `ssl.SSLContext` raises `ConfigurationError` rather than silently
  connecting without TLS. (Requires `dataknobs-utils` with the new
  `sslmode` connector parameter.)

### Security

- Bumped minimum `pyarrow` requirement (extra: `parquet`) from
  `>=17.0.0` to `>=23.0.1` to exclude PYSEC-2026-113 (CVSS 7.0),
  flagged at the floor resolve by the `dependency-update` workflow.
  The bump preserves the prior sweep of PYSEC-2023-238 (CVSS 9.8) and
  PYSEC-2024-161 (both fixed by 17.0.0).

## v0.4.20 - 2026-05-20

### Fixed
- **`TestS3Backend` LocalStack bucket provisioning** —
  `tests/examples/test_vector_multi_backend.py::TestS3Backend` no
  longer assumes `test-bucket` pre-exists on the LocalStack volume.
  Both `test_s3_sync_backend` and `test_s3_async_backend` now depend
  on the shared `make_localstack_s3_bucket` fixture from
  `dataknobs_common.testing.localstack_fixtures`, which idempotently
  creates the bucket on session entry. Inlined `localstack_host`
  detection blocks removed in favour of the resolved
  `endpoint_url` the fixture provides. Only affects opt-in
  (`TEST_S3=true`) test runs.

## v0.4.19 - 2026-05-18

### Added

- **`FaissVectorStore` timestamp exposure** — `FaissVectorStore` now
  tracks `created_at`/`updated_at` per vector and accepts
  `include_timestamps=True` on `get_vectors()` and `search()`, at
  parity with `MemoryVectorStore` and `PgVectorStore`. Timestamps are
  carried across upserts (created preserved, updated refreshed),
  evicted with the row on delete/`clear`, and persisted in the FAISS
  sidecar pickle (legacy indexes without the side-car load empty and
  surface `None` until the next write — same pre-migration semantics
  as the other backends). Only `ChromaVectorStore` remains deferred.

- **`VectorStore.update_metadata_where(filter, set_) -> int`** — the
  filter-keyed sibling of the id-keyed `update_metadata`. Bulk-*merges*
  `set_` into the metadata of every vector matching `filter` (same
  four-quadrant filter shape as `clear` / `count` / `search`; `None`
  matches all), preserving unrelated metadata keys, and returns the
  number of rows affected. Implemented on **all four in-tree stores**:
  `MemoryVectorStore`, `FaissVectorStore` (side-car merge — FAISS
  filtering is post-retrieval, there is no index to invalidate),
  `PgVectorStore` (`metadata = metadata || $::jsonb`), and
  `ChromaVectorStore` (fetch-merge-`update`). The ABC default raises
  `NotImplementedError` — the contract for **out-of-tree**
  implementers only, so an unported backend fails loudly rather than
  silently mis-applying a zero-downtime swap; it is never reached by
  a backend DataKnobs ships. This is the store-layer primitive behind
  `dataknobs-bots`' `IngestSwapMode.TOMBSTONE` re-ingest.

- **`AsyncS3Database.region`** — public attribute exposing the resolved
  region (`None` when the config relies on the boto default chain), at
  parity with the long-standing `SyncS3Database.region`. Lets callers
  inspect region resolution without reaching into the internal pool
  config.

### Changed

- **Review before upgrade.** `PgVectorStore` now validates the
  existing `embedding` column's vector dimensionality at
  initialization (when the table already exists and
  `auto_create_table=True`). A mismatch between the stored
  `vector(N)` and the configured `dimensions` now raises
  `ConfigurationError` at `initialize()` — naming both dimensions —
  instead of deferring to an opaque `asyncpg.DataError` at the first
  insert. The guard is read-only (it reads
  `pg_attribute.atttypmod`; no schema is altered or dropped).
  Consumers that (incorrectly) relied on the silent
  `CREATE TABLE IF NOT EXISTS` dimension shadow must drop/migrate the
  mismatched table or reconfigure `dimensions`.

### Fixed

- **`ChromaVectorStore` works against chromadb 1.x and no longer
  corrupts non-scalar metadata.** chromadb's metadata contract is
  scalar-only: it rejects an empty/`None` metadata dict, and — the
  dangerous case — *silently accepts* a list/dict-valued metadata
  value then corrupts it, bleeding the value positionally across
  unrelated collections that share chromadb's process-wide in-memory
  `System`. Every list/dict value (including `[]`) is now encoded to a
  reversible JSON sentinel at the Chroma boundary and restored on read
  (the legacy empty-list sentinel still decodes), so chromadb only ever
  stores scalars and the metadata round-trip — `{"k": []}`,
  `{"k": [...]}`, nested dicts — matches `MemoryVectorStore`/
  `FaissVectorStore` with no cross-store contamination. chromadb result
  fields (now numpy arrays) are coerced before truthiness/indexing,
  fixing `get_vectors`/`search` silently returning no rows. List
  filter values are post-filtered (chromadb's where-engine returns
  zero rows for any predicate against list-valued metadata) unless the
  key is declared in `scalar_metadata_keys`; four-quadrant results are
  unchanged. The `chromadb` floor is now `>=1.0.0`.

- **`MemoryVectorStore`/`FaissVectorStore` now own ingested
  metadata** (copy-on-ingest, parity with `PgVectorStore`/
  `ChromaVectorStore` which already serialize on write). Callers may
  safely reuse or mutate the dict they passed to `add_vectors`
  without corrupting store state, and store-internal keys (`_stale`,
  injected timestamps) no longer leak onto the caller's dict.
  (Behavior already in effect since the config-level `domain_id`
  symmetry change via `VectorStoreBase._apply_domain_default`; this
  entry documents the guarantee and adds a cross-backend conformance
  test.)

- **`FaissVectorStore.get_vectors()` returns stored vectors and
  metadata for every index type.** Previously it returned
  `(None, None)` for all ids on `ivfflat`/`ivfpq` indexes
  (auto-selected for embedding dimensions ≥ 100 — the 384/768/1024
  production case); `flat`/`hnsw` were unaffected. The store now keeps
  the authoritative vectors in an internal side-car (same key space
  as its metadata/timestamp stores) and serves `get_vectors` from
  there instead of FAISS reconstruct-by-id, which is not usable for
  IVF without a maintained direct map that this faiss build refuses
  to combine with `remove_ids`. The FAISS index is retained for
  similarity `search`; `get_vectors`, `delete_vectors`, upsert,
  `clear`, and save/reload stay correct for IVF across re-ingest and
  clear/repopulate cycles. A resolved id whose internal id has no
  stored vector (post-delete reuse race) is logged at WARNING rather
  than being silently indistinguishable from an absent id.
  **Migration:** an index persisted by an earlier `dataknobs-data`
  has no stored-vector side-car, so `get_vectors` returns `None` (and
  empty timestamps) for its ids until rebuilt — re-add the vectors
  (or re-ingest) once; `search` is unaffected, and new indexes need
  no action.
- **`FaissVectorStore` no longer crashes when an IVF store's first
  batch is smaller than `nlist`.** Previously a sub-`nlist` first
  `add_vectors` on an `ivfflat`/`ivfpq` store raised
  `RuntimeError: ... 'is_trained' failed` (the train-skip path fell
  through to `add_with_ids` on an untrained IVF index). The store now
  serves a temporary flat index until the corpus reaches `nlist`,
  then trains the real IVF and migrates to it from the side-car —
  search and `get_vectors` stay correct throughout. The deferred
  state is persisted, so a save/reload before the threshold resumes
  correctly.
- **`FaissVectorStore` IVF search now honors the configured
  `nprobe`.** The index is wrapped in `IndexIDMap2`, which does not
  proxy `nprobe`, so the setting never reached the underlying IVF and
  every `ivfflat`/`ivfpq` search ran at FAISS's default `nprobe=1`
  regardless of `search_params.nprobe` — silently degrading recall.
  `search()` now unwraps the inner index and applies `nprobe` there.

### Changed

- **`PgVectorStore` default `schema` changed from `"edubot"` to
  `"public"`** (the PostgreSQL default). **Review before upgrade:**
  deployments that relied on the implicit default were writing to a
  schema named after an unrelated project; after upgrade they will
  use `public`. To retain prior behavior, set `schema="edubot"`
  explicitly in the store config. No in-tree consumer relied on the
  implicit default.

- **`MemoryVectorStore`, `FaissVectorStore`, and `ChromaVectorStore`
  now honor a config-level `domain_id`** (matching `PgVectorStore`).
  A store constructed with `{"domain_id": "x", ...}` defaults
  `domain_id="x"` into the metadata of vectors added without one and
  AND-composes `domain_id="x"` into the effective filter for
  `search()`, `count()`, `clear()`, and `update_metadata_where()`.
  `clear()` (no filter) on a tenant-scoped store now deletes only
  that tenant's rows rather than wiping the whole collection, and an
  out-of-scope explicit `domain_id` filter resolves to a no-match.
  `PgVectorStore` behavior is unchanged (its SQL predicate already
  enforced this). **Review before upgrade:** consumers that
  previously set `domain_id` in the store config on Memory/FAISS/
  Chroma (where it was silently a no-op) now get real tenant
  isolation — `count()`/`search()`/`clear()` will scope to that
  tenant. One residual cross-backend divergence remains and is
  documented in `VECTOR_FILTER_SEMANTICS.md`: an *explicit*
  `filter={"domain_id": "x"}` is a metadata-key match on Memory/
  FAISS/Chroma but a JSONB-containment probe on PgVector (which
  stores the configured tenant in a column, not in JSONB) — rely on
  config-level scoping, not explicit `domain_id` filters, for
  backend-portable isolation.

## v0.4.18 - 2026-05-13

### Added

- **`AsyncKeyedRecordStore[T]` / `SyncKeyedRecordStore[T]`** — generic
  id-keyed persistence over `AsyncDatabase` / `SyncDatabase` for
  registry / pointer-table use cases.  Encapsulates the `Record`
  two-column (`data`, `metadata`) shape *by construction*: the
  serializer signature is ``(T) -> tuple[dict, dict]`` rather than
  ``(T) -> Record``, so the metadata channel is part of the function's
  type and cannot be silently dropped.  Surface: `put`, `get`,
  `exists`, `delete`, `put_batch`, `get_batch`, `delete_batch`,
  `list`, `count`, `stream`, `search`.  Filter channels —
  `filter_data` and `filter_metadata` — both routed through the
  existing `metadata.X` field-path convention so JSONB pushdown
  works on Postgres / SQLite / DuckDB and `Record.get_value`
  traversal works on memory / file backends.  Exported from
  `dataknobs_data` package root.  Composed by
  `DataKnobsRegistryAdapter`, `ArtifactRegistry`, `RubricRegistry`,
  and `GeneratorRegistry` in `dataknobs-bots`, and by
  `UnifiedDatabaseStorage.save_step` in `dataknobs-fsm`, as the
  single Record-construction site for those registries.

### Changed

- **`limit=0` now produces an empty result across every backend**,
  consistent with Python slice semantics (``limit=None`` →
  unlimited, ``limit=0`` → empty).  Previously the pagination paths
  used truthy-checks (``if query.limit:`` / ``if query.offset:``),
  so ``limit=0`` was silently treated as "no limit".  ``offset=0``
  is now also documented as a no-op rather than a slice that copies
  the full list.

  **Migration:** Audit consumers that pass ``limit=0`` explicitly.
  Any caller that relied on the truthy-check to silently mean
  "unlimited" will now receive an empty result; pass
  ``limit=None`` (or omit the argument) for unlimited semantics.

## v0.4.17 - 2026-05-09

### Added

- **`VectorStore.clear(filter=...)`** — filter-aware clear, now
  supported across all four backend implementations
  (`MemoryVectorStore`, `FaissVectorStore`, `ChromaVectorStore`,
  `PgVectorStore`). When `filter` is `None` (default), behavior is
  unchanged — all vectors are removed. When provided, only vectors
  whose metadata matches the filter are removed; non-matching
  vectors are preserved. The filter shape matches `search()` and
  `count()`; each backend reuses its existing filter-translation
  infrastructure (`_match_metadata_filter` for memory/FAISS,
  `_partition_filter_for_chroma` for Chroma,
  `_build_jsonb_filter_sql` for pgvector).

  This closes a long-standing gap where multi-tenant shared stores
  could not perform per-tenant cleanup without scanning IDs in the
  consumer. `KnowledgeIngestionManager` (in `dataknobs-bots`) now
  uses this to scope its automatic clear-before-reingest by
  `domain_id`.

- **`ChromaVectorStore` accepts `scalar_metadata_keys`** — opt-in
  declaration of metadata keys whose stored values are guaranteed
  scalar (never list-valued). For declared keys with scalar filter
  values, `_partition_filter_for_chroma` pushes a Chroma-native
  `$eq` predicate instead of post-filtering in Python.
  `count(filter=...)` then fetches IDs only (no metadata
  materialization) when the filter pushes down fully — eliminating
  the memory-bound trade-off documented in
  `VECTOR_FILTER_SEMANTICS.md` for the common multi-tenant scoping
  pattern (e.g. `{"domain_id": "x"}`). Backward compat preserved:
  keys not declared keep the conservative post-filter behavior.

- **`VECTOR_FILTER_SEMANTICS.md` documents the pgvector
  config-level `domain_id` swap asymmetry** — when runtime-swapping
  between vector-store backends, `PgVectorStore`'s config-level
  `domain_id` scopes `clear()` automatically while the other three
  backends do not. The doc gives explicit guidance for swap-safe
  consumers.

### Fixed

- **`FaissVectorStore.add_vectors` no longer leaks orphan metadata
  on upsert.** Pre-fix, re-adding an external ID overwrote
  `id_map[ext_id]` without removing the prior internal ID's entries
  from the FAISS index or `metadata_store`, leaving silent residuals
  that filtered `clear()` could not reach (it walks `id_map`).
  Post-fix, the prior internal ID is evicted from FAISS and
  `metadata_store` before the new mapping is assigned.

### Migration

- **No source-compat break.** `await store.clear()` continues to
  work and continues to remove all vectors.
- **Backend-specific note (FAISS).** FAISS has no native filtered
  delete; filtered clear iterates `metadata_store` to collect
  matching IDs and delegates to `delete_vectors(ids)`. This is O(N)
  over stored vectors — acceptable for typical KB sizes, but
  workloads at scale where filtered clear is hot should prefer
  pgvector or Chroma where filtered delete is native.

### Security
- Bumped minimum `duckdb` requirement (extra: `duckdb`) from `>=0.9.0`
  to `>=1.1.0` to exclude versions affected by PYSEC-2024-25 (CVSS
  9.8) and PYSEC-2024-203. This is a major-version bump (0.x → 1.x);
  the public DuckDB API used by `SyncDuckDBDatabase` /
  `AsyncDuckDBDatabase` (connection management, `execute`, `query`,
  Arrow result conversion) is stable across this range.
- Bumped minimum `pyarrow` requirement (extra: `parquet`) from
  `>=14.0.0` to `>=17.0.0` to exclude versions affected by
  PYSEC-2023-238 (CVSS 9.8) and PYSEC-2024-161.

### Changed

- **`validate_database_name()` now raises `ConfigurationError`
  instead of `ValueError`** for consistent exception typing across
  the postgres identifier-validation surface (the new
  `validate_pg_identifier` already raised `ConfigurationError`,
  and config-shape errors belong to the
  `dataknobs_common.exceptions` hierarchy).  External callers that
  catch `ValueError` specifically must update to catch
  `ConfigurationError` (or its base `DataknobsError`).
  `validate_database_name` is internal infrastructure with no
  publicly documented `ValueError` contract, so this is a small
  behavior change rather than a breaking API change.

### Fixed

- **`PostgresBaseConfig._parse_postgres_config` now raises
  `ConfigurationError` when the `table` or `schema` config key is
  not a valid string identifier**, instead of silently propagating
  non-string values through `quote_ident()` and producing broken
  SQL at first query.  Defense-in-depth — the canonical fix for
  the FSM-side `schema`-key collision lives in `dataknobs-fsm`
  (Item 117).  This validator catches misuse from any future
  consumer that accidentally injects a non-identifier value via
  either key.  The same identifier shape (`^[a-zA-Z_][a-zA-Z0-9_]*$`)
  used by `validate_database_name` is enforced for both keys
  through the public `validate_pg_identifier` helper in
  `dataknobs_data.backends.postgres_mixins`.
- **`PgVectorStore` now validates `schema` and `table_name`
  identifiers at construction**, closing the third Postgres
  consumer's parallel hazard.  `PgVectorStore._parse_backend_config`
  reads these keys directly (it does not flow through
  `_parse_postgres_config`), so the records-backend fix above did
  not cover it.  Both consumers now use the same
  `validate_pg_identifier` helper, so a malformed identifier is
  caught with a clear `ConfigurationError` at construction
  regardless of which Postgres consumer the application uses.
- **`AsyncPostgresDatabase.update_batch()` no longer raises
  `PostgresSyntaxError` from a duplicated `RETURNING id` clause**.
  The query was built by `SQLQueryBuilder.build_batch_update_query`,
  which already appends ` RETURNING id` for the postgres dialect
  (sql_base.py:559-561), and then `update_batch` appended
  ` RETURNING id` a second time at postgres.py:1484 — producing
  invalid SQL ending in `RETURNING id RETURNING id`. asyncpg
  rejected the query before any row was updated. Pre-existing
  latent bug uncovered while reviewing PR #303 — no prior test
  exercised `AsyncPostgresDatabase.update_batch` against a real
  Postgres (only sqlite/duckdb async `update_batch` had coverage).
  Fix removes the second `RETURNING id` append; the builder's
  output is already postgres-ready. Sync `SyncPostgresDatabase.
  update_batch` was already correct (its comment "query now
  includes RETURNING clause" was accurate).
- **`AsyncPostgresDatabase.stream_read()` no longer raises
  `TypeError: 'async for' requires an object with __aiter__ method,
  got Cursor`**. The cursor was constructed via
  ``cursor = await conn.cursor(sql, *params)``, which returns an
  asyncpg ``Cursor`` (intended for the explicit-fetch API
  ``await cur.fetch(n)``) and is not an async iterator; the
  subsequent ``async for row in cursor`` then failed before yielding
  the first row. Pre-existing bug uncovered by the new
  ``test_async_stream_read_preserves_record_id`` parity test added
  for the Item 114 fix above (no prior test exercised
  ``AsyncPostgresDatabase.stream_read`` against a real Postgres). Fix
  iterates the ``CursorFactory`` returned by
  ``conn.cursor(sql, *params)`` directly — matching the asyncpg
  pattern used elsewhere in this file. ``stream_read`` now actually
  streams rows from Postgres for the first time.
- **`AsyncPostgresDatabase.read()`, `search()`, `vector_search()`,
  `stream_read()`, and `_text_search_for_hybrid()` now return records
  with populated `record.id` / `record.storage_id`**, matching the
  sync `SyncPostgresDatabase` behavior. The async `_row_to_record`
  previously copy-pasted the sync serializer body but dropped the
  `ensure_record_id` step, so `await db.read(id)` returned records
  where `record.id` / `record.storage_id` were whatever was in the
  JSON payload (typically `None`) — silently differing from
  `db.read(id)` on the sync backend for the same on-disk row.
  `search()` was the only async call site that compensated explicitly
  (`record.storage_id = str(row['id'])` after `_row_to_record`); the
  other four were silently broken. The fix delegates the async
  `_row_to_record` to the shared
  `SQLRecordSerializer.row_to_record(dict(row))` static helper that
  the sync sibling already uses, so all five async call sites now
  populate the id uniformly. Strictly information-additive — the
  sync backend has always returned the populated id, so consumers
  working against both backends already handle the populated case.

### Changed

- **`SQLRecordSerializer.record_to_row(record, id=None)`** added as
  the outbound counterpart to the existing
  `SQLRecordSerializer.row_to_record(row)` static. Centralizes the
  `id` / `data` / `metadata` row shape so sync and async SQL
  backends do not duplicate the body and silently drift — the same
  shape that produced the inbound `_row_to_record` divergence.
- `SyncPostgresDatabase._record_to_row` and
  `AsyncPostgresDatabase._record_to_row` are now one-line delegations
  to `SQLRecordSerializer.record_to_row`. Behavior is unchanged
  (both bodies were functionally identical pre-consolidation); the
  consolidation eliminates the parallel-implementation drift surface.
- **All four redundant `record.storage_id = str(row['id'])`
  assignments in SQL backend `search()` paths have been removed.**
  After the `_row_to_record`-delegation fix above, every SQL
  backend's search path now goes through
  `SQLRecordSerializer.row_to_record` (directly, or via
  `SQLQueryBuilder.row_to_record`), which calls `ensure_record_id`
  before returning — so the post-call explicit assignments were
  no-ops. Cleanups: `AsyncPostgresDatabase.search()` (postgres.py:
  1362-1363), `SyncPostgresDatabase.search()` (postgres.py:419-420),
  `AsyncSQLiteDatabase.search()` (sqlite_async.py:271-272),
  `AsyncDuckDBDatabase.search()` (duckdb.py:404-405), and
  `SyncDuckDBDatabase.search()` (duckdb.py:952). Behavior unchanged
  (each was a redundant double-write of the same value); the
  cleanup eliminates the future-confusion surface ("why does this
  set storage_id when `_row_to_record` already does?").
- New unit-test module `tests/test_backends/test_sql_record_
  serializer.py` covers `SQLRecordSerializer.row_to_record` and
  `SQLRecordSerializer.record_to_row` directly (round-trip,
  id-population, metadata serialization edge cases). The new
  helpers were previously covered only transitively via integration
  tests requiring a live Postgres.
- `packages/data/docs/RECORD_SERIALIZATION.md` documents the new
  `record_to_row` static and the inbound/outbound boundary
  contract, with a forward-reference to the Item 114 cautionary
  tale.

## v0.4.16 - 2026-04-29

### Security

- **`get_vector_extraction_sql`, `_build_text_field_concat`, and both `stream_read` implementations now validate field names** against `[A-Za-z_][A-Za-z0-9_]*` before embedding them in SQL string literals (JSONB key positions). All four functions previously accepted input like `"field'name"` or `"'; DROP TABLE;--"`, which breaks SQL syntax or enables injection in the string-literal position where `quote_ident()` does not apply. In `stream_read`, validation fires before the connection check so it raises `ValueError` on first iteration without requiring a live database. Invalid names raise `ValueError`; valid names are unchanged. The shared `validate_field_name(field)` helper in `sql_base` centralises the check and error message; `postgres.py` calls it via the public function rather than reaching into the private `_FIELD_NAME_RE` regex.

### Changed

- **Identifier quoting in all SQL backends**: `SyncPostgresDatabase`, `AsyncPostgresDatabase`, `SyncSQLiteDatabase`, `AsyncSQLiteDatabase`, `SyncDuckDBDatabase`, `AsyncDuckDBDatabase`, `PgVectorStore`, and `PostgresTableManager` now internally quote schema and table names using `quote_ident()` from `dataknobs_utils`. Any valid SQL identifier (mixed-case, reserved words, etc.) is now accepted without pre-quoting. Existing consumers using plain `[a-z_][a-z0-9_]*` names see no behavior change. Vector column names in `AsyncPostgresDatabase` are also fully quoted (`_ensure_vector_column` ALTER TABLE, `vector_search`, `hybrid_search`). `AsyncPostgresDatabase.stream_write()` uses asyncpg's `schema_name=` keyword so the table name is not double-quoted. `postgres_vector.py` helper functions (`build_vector_index_sql`, `get_vector_count_sql`) now accept pre-quoted identifier arguments.

- **`PostgresTableManager.get_table_exists_sql()`** added as a new static method returning `(sql, params)` tuple with `$1`/`$2` parameter binding.

### Added
- `auto_create_table` config option on all SQL-style relational database
  backends — `Sync/AsyncPostgresDatabase`, `Sync/AsyncSQLiteDatabase`,
  `Sync/AsyncDuckDBDatabase`. Default is `True` (no behaviour change for
  existing consumers). When `False`, `connect()` verifies the records table
  exists and raises `RuntimeError` if it doesn't, enabling
  Alembic/Flyway/Sqitch-managed schemas with DML-only application roles.
  Mirrors the existing `PgVectorStore.auto_create_table` contract.
- `SQLTableManager.get_table_exists_sql()` — dialect-aware parameterized
  table-existence query supporting qmark (`?`), numeric (`$1`/`$2`), and
  pyformat (`%(name)s`) placeholder styles. Used internally by all SQL
  backends; both Postgres backends now delegate to this shared helper
  (`SyncPostgresDatabase` with `param_style="pyformat"`,
  `AsyncPostgresDatabase` with `param_style="numeric"`) replacing the
  separate `PostgresTableManager.get_table_exists_sql()` static method.
- `SQLTableManager.coerce_bool()` — public shared helper for coercing
  YAML/env string values (`"false"`, `"0"`, `"no"`) to Python `bool`.
  `None` returns the `default` parameter (``True`` by default). Replaces
  per-backend inline coercion logic for consistent edge-case handling.
  **Behaviour change for `ensure_database`:** the previous inline coercion
  used an allowlist (`"true"`, `"1"`, `"yes"` → `True`; all other strings
  → `False`). `coerce_bool` uses a blocklist (`"false"`, `"0"`, `"no"`, `""`
  → `False`; all other strings → `True`). Unrecognised strings such as
  `"on"` or `"enabled"` now correctly enable the feature rather than
  silently disabling it.
- `SQLTableManager.__init__` now accepts a `param_style` keyword argument
  (`"qmark"` default, `"numeric"` for asyncpg, `"pyformat"` for psycopg2)
  controlling which placeholder style `get_table_exists_sql()` emits.

## v0.4.15

### Breaking
- Individual keys now override the same field from a
  `connection_string` (restoring the historical
  `_parse_postgres_config` precedence). A caller passing
  `{"connection_string": "postgresql://.../dbA", "database": "dbB"}`
  now connects to `dbB`, not `dbA`. Pre-Unreleased releases had
  briefly inverted this.

### Added
- `S3SessionConfig` and `create_boto3_s3_client` in
  `dataknobs_data.pooling.s3` — single canonical layer for boto3 /
  aioboto3 S3 client construction. Used by `SyncS3Database`,
  `AsyncS3Database` (via `S3PoolConfig.to_session_config()`), and
  `S3KnowledgeBackend`. `S3SessionConfig.from_dict` accepts both
  `region`/`region_name` and both legacy (`access_key_id`,
  `max_workers`, `max_retries`) and canonical (`aws_access_key_id`,
  `max_pool_connections`, `max_attempts`) key shapes so one config
  dict feeds every S3 construct.
- **`PgVectorStore` now tracks `updated_at`** on each row. The schema
  gains `updated_at TIMESTAMP DEFAULT NOW()`, refreshed to `NOW()`
  on every upsert (same-ID `add_vectors`) and on `update_metadata`;
  `created_at` is preserved on upsert. Pre-existing tables gain the
  column via idempotent `ALTER TABLE ADD COLUMN IF NOT EXISTS`
  during `initialize()` when `auto_create_table=True`; pre-existing
  rows keep `updated_at IS NULL` until re-ingested — treat `NULL` as
  "not re-ingested since the column was added." Consumers with
  `auto_create_table=False` must apply the ALTER manually (SQL in
  the `pgvector-backend.md` doc). Memory (in-process) tracks the
  same (created, updated) tuple and preserves it through the pickle
  round-trip.
- **`VectorStore` timestamp exposure** — `include_timestamps=True`
  on `get_vectors()` and `search()` returns `_created_at` /
  `_updated_at` in the metadata dict. Format (`iso` / `epoch` /
  `datetime`) and key names are configurable via a new `timestamps`
  config block on every `VectorStoreBase` subclass. Supported by
  `MemoryVectorStore` and `PgVectorStore`;
  `FaissVectorStore` and `ChromaVectorStore` do **not** yet accept
  the `include_timestamps` kwarg (calling it raises `TypeError`) —
  deferred per Item 36 follow-ups. Collision policy: consumer
  metadata values for a configured timestamp key always win; a
  WARNING is logged once per process per colliding key. See
  `packages/data/docs/vector-timestamps.md` for the full contract.

### Changed
- **Behavior change (Defect A):** `PgVectorStore` `id_type` default
  changed from `"uuid"` to `"text"`. RAG consumers passing chunk ids
  such as `"01-fundamentals_0"` now work out-of-the-box. **No data
  migration is required** — `CREATE TABLE IF NOT EXISTS` is a no-op
  on existing tables. **A config update IS required** for pre-flip
  consumers whose tables use a UUID `id` column: add
  `id_type: "uuid"` to the store config, otherwise inserts and
  lookups will fail with a guided `ValueError` pointing at the fix.
- `PgVectorStore`, `PostgresPoolConfig`, `AsyncPostgresDatabase`, and
  `SyncPostgresDatabase` now accept individual `host`/`port`/
  `database`/`user`/`password` keys plus `POSTGRES_*` env-var
  fallbacks. `DATABASE_URL` env fallback now works uniformly across
  all postgres-using constructs (previously only PgVectorStore).
- `SyncPostgresDatabase._open_connection` no longer uses
  `DotenvPostgresConnector` directly; the connection path goes through
  `normalize_postgres_connection_config`, which reads `.env` /
  `.project_vars` files as an additional env fallback layer
  (preserving the retired connector's auto-loading behavior for
  developers who keep secrets in those files).
- **Behavior change:** `SyncS3Database` and `S3KnowledgeBackend` no
  longer default `region` to `"us-east-1"`. Both now defer to
  boto's resolution chain (`AWS_DEFAULT_REGION` env →
  `~/.aws/config` → IMDS → `us-east-1` terminal fallback) when no
  region is configured. Consumers who set `AWS_DEFAULT_REGION`
  previously had it silently overridden — it is now honored.
  Consumers explicitly passing `region: "us-east-1"` see no change.
  Consumers with no AWS config anywhere still terminate at
  `us-east-1` (boto's fallback), preserving existing behavior.
- `S3PoolConfig.from_dict` now accepts `region` (in addition to
  `region_name`), so the same config dict feeds sync and async S3
  paths without rename.
- `SyncS3Database._ensure_bucket_exists` now resolves the effective
  region from `client.meta.region_name` when `region` is unset, so
  bucket creation correctly applies `LocationConstraint` for
  env-derived regions.
- `S3SessionConfig.to_client_kwargs()` and `validate_s3_session`
  automatically add `use_ssl=False` when `endpoint_url` starts with
  `http://` (LocalStack, MinIO, dev S3-compatible servers),
  preserving the previous `SyncS3Database` behavior. `https://`
  endpoints leave `use_ssl` unset so boto's default (`True`) applies
  — a slight tightening of the prior code, which set `use_ssl=False`
  for *all* custom endpoints regardless of scheme. Callers can
  override either case via
  `extra_client_kwargs={"use_ssl": ...}`.

### Fixed
- **Defect C:** `asyncpg.DataError` raised by a `PgVectorStore` when
  the configured `id_type` disagrees with the actual id value or
  column type is now wrapped as a guided `ValueError`. Both
  directions are covered: `id_type="uuid"` + non-UUID id, and
  `id_type="text"` + UUID-typed column (the common post-Defect-A
  migration case). The message names the offending id, the table, and
  the exact config or schema change required.
- `PgVectorStore.delete_vectors` now validates ids client-side when
  `id_type="uuid"` so a bulk delete containing one malformed id
  surfaces that specific id in the error instead of dumping the
  full list.
- `validate_s3_session` no longer passes an empty `endpoint_url`
  kwarg to boto when none is configured.
