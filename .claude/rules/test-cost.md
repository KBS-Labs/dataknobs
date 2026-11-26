# Test Cost — a fresh service resource is for a test that writes

## The rule

> **A test takes a resource of its own only when it writes to it. A test that
> only reads shares one seeded resource.**

Per-test provisioning is the choice an author makes without thinking, because it
is the safe one: nothing another test does can reach your index. It is also where
the gate's time goes. When this was measured, Elasticsearch setup (create an
index, seed it, refresh it) was most of the setup time across the suites that
use a service, and most of the tests paying it only searched what they had seeded.

So this is not left to memory. The cheap shape is the one the shared fixtures
make easy, and the remaining risk is a count the gate holds under a ceiling.

## The patterns

| Pattern | Do | Don't |
|---|---|---|
| **Seed once, read many** | `make_elasticsearch_corpus(name, seed)` for tests that only search, count or read: seeded once per module and name, shared by every case and both twins | give each read-only case an index of its own to seed and refresh |
| **A fresh resource only for a writer** | `make_elasticsearch_test_index` in a test that creates, updates, deletes, closes, blocks or remaps | take one for a test that only searches, counts or reads |
| **Seed in a batch** | `create_batch(records)`, so one refresh covers the corpus | `create()` per record with `refresh=True`: one refresh per record |
| **Never re-ensure a session resource** | ask `make_localstack_s3_bucket` for a bucket; it is made once per session and the factory remembers it | ensure or recreate a shared bucket per test; isolate with a per-test key prefix instead |
| **Take names from the plugin** | `config["index"]`, or a name derived from it (`config["index"] + "_down"`) | spell an index name in a test, as a literal or an f-string |

A test that seeds in **setup** and then only reads is the shape the count looks
for. Seeding inside the test body is a write, so such a test is not counted, but
it pays for the index all the same; prefer the shared shape for it too.

`seed_elasticsearch_records(records, **config)` (`dataknobs_data.testing`) is
the usual seed: one `create_batch` through the backend, so the corpus has the
backend's own mapping. A test of an index made some other way (a foreign
mapping, a bare index) passes its own seed callable. In a parity suite, only the
Elasticsearch branch of the backend helper takes the corpus; skip the shared
seeding loop for that branch, since the corpus is already seeded.

## What holds it

### The provisioning count

When the gate runs a suite it sets `DK_PROVISIONING_RECORD`, and the plugin
(`dataknobs_common.testing.provisioning`) fingerprints every index a test took
as its call phase begins and ends. A test whose index **existed when the call
began and was unchanged when it ended** is counted: setup paid for an index the
test never needed to own. A name taken and never created costs nothing and is not
counted.

The count is per file, summed per package, and held in
`.dataknobs/quality-contract.json` under the `provisioning` tool, with one cell
per package test directory. It is a ratchet like every other cell:

- **Above its ceiling, the gate fails.** The message names each file and its
  count; the suite's `provisioning-<stage>-<package>.json` under the gate's
  output directory names each test. Move the test onto a corpus
  (`make_elasticsearch_corpus`), or, if it should write, let it.
- **Below its ceiling, the gate also fails,** and prints the command that lowers
  it. Lowering it is how the progress is recorded:

  ```bash
  uv run python bin/quality-contract.py update-baseline --tool provisioning \
      --cell packages/<package>/tests --records <gate output dir> --tested <package>
  ```

- **A package the run did not test is not measured,** and its ceiling stands.
- **Tests skipped because Elasticsearch was unreachable fail the check.** Those
  tests took nothing, so the count would fall and ask for a ceiling the tree has
  not earned. Start the services (`bin/dk up`) and rerun.

The count only exists once the tests have run, so it is not part of the bare
`quality-contract.py check` (which the gate runs before any test). It runs after
the integration stage. `charge --tool provisioning <path> --records <dir> --tested <package>`
answers per file from the same records.

### The corpus check

Sharing is safe only while nobody writes. When `make_elasticsearch_corpus`
tears down, each corpus the test took is fingerprinted against the reading
taken after seeding. A test that wrote errors in its own teardown, naming
`make_elasticsearch_test_index`; every later test asking for that corpus errors
at setup, naming the writer. This runs on every session, recorded or not.

When it fires, the test it names writes: move it onto a fresh index. Do not
re-seed or work around it in the later tests, which did nothing wrong.

### The census guard

`tests/test_tests_take_index_names_from_the_plugin.py` reads every test file that
references `requires_real_elasticsearch` and refuses an index name spelled in it.
A name the plugin did not hand out is shared by every run that uses it, survives
teardown, can be removed by another checkout's session sweep while in use, and is
never fingerprinted, so it would go round the count.

## Not covered

- **S3.** The plugin hands out a shared bucket, and each test's isolation is a
  key prefix it chooses itself, so the plugin cannot see what a test was given;
  measured S3 setup is small since the bucket stopped being re-ensured per test.
- **Postgres.** Its statistics counters lag across connections, so a fingerprint
  taken at the end of a call can miss the call's own writes.

Both would need a different mechanism before they could be counted. Until then
the patterns above still apply to them.

## Related

- [`touched-file-cleanup.md`](./touched-file-cleanup.md) — the same ratchet,
  for type-checker findings
- `packages/common/docs/testing.md` — the fixtures, the corpus, the count and
  the record
- `docs/development/quality-checks.md` — where the check runs in the gate
