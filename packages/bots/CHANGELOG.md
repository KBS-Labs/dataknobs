# Changelog

All notable changes to the dataknobs-bots package will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Added

- **Wizard arcs carry a name, and a transition record says which one fired.**
  A stage may declare several transitions to the same target — different
  conditions, one destination. The compiled FSM arcs were anonymous, so
  `Arc.name` fell back to `"<source>-><target>"` and both siblings reported the
  same string. The loader now names every arc it compiles:
  `"<source>-><target>#<index>"`, or the `name` an author sets in the
  transition's `metadata`, which takes precedence and already reached `Arc.name`
  before this.

  The derived form extends the FSM's own fallback rather than replacing it, so a
  reader who knows the old string still reads the prefix and only the
  discriminator is new. It is reported as `StepResult.transition`, recorded in
  the stage metadata beside `target` and `condition`, and persisted on
  `TransitionRecord` as a new **`transition_name`** field — which
  `transition_record_to_execution_record` now carries into `ExecutionRecord`,
  where the field has always existed with nothing to put in it, and
  `execution_record_to_transition_record` carries back.

  The field is defaulted and appended, so a record persisted by an earlier
  version restores with `transition_name=None`. A consumer asserting an exact
  key set on a serialised record will see one more.

  **The reported string changes for wizard arcs.** Anything keyed on the old
  endpoint-derived form — an FSM trace filter, an `arc_name` selector passed to
  `execute_step_*` — now sees `"<source>-><target>#<index>"` for an arc compiled
  from a wizard transition. That string was never unique between siblings, which
  is the defect below; the prefix is unchanged, so a match on it still works.

  `dataknobs_bots.reasoning.wizard_loader.arc_identity(source_stage, transition,
  idx)` is exported as the single derivation of an arc's target and name, so a
  consumer correlating its own config against `StepResult.transition` need not
  re-derive the rule.

  **A name identifies an arc only while it is unique.** The derived form carries
  the index and cannot collide; an authored `metadata: {name: ...}` can be
  repeated, and two arcs answering to one string are as unidentifiable as the
  anonymous arcs this replaces. The loader reports a collision at load time,
  naming both transition indices, and the readers treat a duplicated name as the
  ambiguous case — recording nothing rather than the first sibling's condition.
  A name is also matched only among the transitions leading to the stage
  actually reached, so one reused on a route elsewhere cannot answer for a move
  it did not cause.

  **The derived index is a position, not an identity.** Reordering a stage's
  transitions, or inserting one above an existing one, changes the name the same
  route compiles to from then on. Records written earlier keep the name they
  were written with and their own `condition_evaluated`, so each stays
  internally consistent; correlating an old name against an edited config does
  not survive. An explicit `name` is how to get one that does.

- **`DynaBot.get_wizard_transitions(conversation_id)`** returns the transition
  records recorded for a conversation. `get_wizard_state()` returns the
  *normalized* state, whose canonical schema carries the stage, the collected
  data and the history but not the records — so `condition_evaluated` and
  `transition_name`, the fields that say which route carried the wizard
  forward, had no supported reader. Same two-path lookup as `get_wizard_state`:
  the in-memory manager first, then persisted storage, so an evicted
  conversation answers from what was saved.

- **`BotTestHarness.get_transitions()`** returns the same records for the
  harness's conversation, delegating to `DynaBot.get_wizard_transitions()` so a
  test asserting on transitions exercises the shipped reader.

- **`WizardConfigBuilder.transition(..., metadata=...)`** (the testing builder)
  passes arc metadata through, so a test can name an arc without falling back to
  an inline config dict.

- **`WizardStateSnapshot` carries the current stage's configuration, and
  converts to the state a tool is handed.** The snapshot gains
  `stage_metadata` — the declared config of `current_stage`: prompt, schema,
  `can_skip`, and whatever else the stage declares — and a
  `to_tool_view()` method returning the `ToolWizardState` a
  `ContextAwareTool` receives.

  The two go together. `ToolWizardState` has five fields and `stage_metadata`
  was the only one with no snapshot counterpart, so a conversion written
  before this could not have populated it: every tool handed a converted
  snapshot would have seen an empty dict, with nothing to say the value had
  been dropped rather than genuinely absent.

  **Only the live constructor supplies the new field.** Stage metadata is not
  written into the persisted `fsm_state`, so `get_state_snapshot()` reads it
  from the FSM that owns the stage — the subflow's, inside a push — while
  `snapshot_from_metadata()` has nothing to read and reports `{}`. That is the
  same answer, for the same reason, that
  `ToolWizardState.from_manager_metadata()` already gives. `stage_definitions`
  is not a substitute: it is the *main* flow's declarations, so inside a push
  it describes a different stage than `current_stage` does, and it is optional
  besides.

  The conversion **copies** its payloads. A *published* `ToolWizardState`
  holds `collected_data` by reference on purpose — that is the live channel,
  and a tool's writes are meant to land in wizard state. A snapshot is not
  that channel, so writes to a converted view go nowhere and the copy makes
  that structural rather than a matter of documentation.

  **`to_dict()` gains a key.** The change is otherwise additive — the field is
  defaulted and appended, and `from_dict()` tolerates its absence, so
  snapshots serialised by an earlier version still load — but a consumer
  asserting an exact key set on a serialised snapshot will see one more.

### Changed

- **`DynaBotConfigSchema.validate` applies the `$resource` marker rule to the
  whole config file, not to its `bot:` section.** The rule is a property of the
  config format rather than of any schema, and the component loop beside it
  visits registered names only — so a reference block under any other key is one
  nothing else looks at. That was already the argument for checking below the
  top level; it holds identically for a section *beside* `bot:` as for one
  beneath it, and the check stopped at the `bot:` boundary anyway.

  **A config that lints clean today can now report a finding.** Everything newly
  reported already failed at resolution with the same sentence — a `$requred`
  under a top-level `domain:` or `educational:` raised `ConfigurationError` when
  the bot was built. What moves is who finds out and when: an authoring-time
  verdict instead of a deploy-time crash. Nothing that shipped was silently
  degraded.

  The surface is bounded to the rule's two halves, so a `$`-prefixed key is not
  enough on its own: an unknown marker is reported only inside a block that
  carries `$resource`, and a stranded marker only for the literal keys
  `$requires` and `$required`. JSON Schema's `$ref`, `$defs`, `$schema` and `$id`
  are invisible to both, so a stage's `schema:` block is unaffected.

  **A finding now names the path the reader has open.** It is rooted at the file
  rather than inside `bot:`, so it reads `bot.knowledge_base.vector_store` where
  it used to read `knowledge_base.vector_store` — a path locating nothing in a
  wrapped config. This falls out of walking the file: a config written without
  the `bot:` wrapper is its own root and keeps unprefixed paths.
  `validate_component` is unchanged, still rooted at the component it was handed.

### Fixed

- **The wizard no longer reports the wrong arc when two transitions share a
  target.** Both readers of a completed step — the DEBUG step log and
  `WizardFSM.get_transition_condition`, whose answer is *persisted* as a
  transition record's `condition_evaluated` — scanned the stage's declared
  transitions for the first one naming the target it landed on. With two arcs to
  one target that names the first whichever fired, and nothing in the output
  indicated a guess had been made. Both now match on the arc name the step
  reports, through one shared matcher rather than two identical scans.

  `get_transition_condition` gains a keyword-only **`arc_name`**. Without it, a
  target reached by more than one transition now returns `None` — the value is
  persisted, where a plausible-but-wrong expression is worse than an absent one,
  and the method's own comment already argued that "nothing recorded" is the
  honest answer when the value cannot be trusted. A target reached by a single
  transition answers exactly as before.

  When the arc cannot be identified, the DEBUG line now says how many arcs lead
  to the target instead of naming one of them. It also renders an unconditional
  transition as `unconditional` rather than `None`: the stage metadata always
  carries a `condition` key, so the `.get()` default the line was written with
  could never fire. A *declared but empty* condition reads as
  `empty (never fires)` rather than being folded in with the unconditional case
  — the loader builds a condition function for any present `condition` key, so
  `condition: ""` compiles to an arc that can never fire rather than one that
  always does.

- **`TransitionRecord.from_dict` no longer raises on a key it does not
  recognise.** These records are persisted into conversation metadata and read
  back by whichever build is running, so a record written after a field is added
  raised `TypeError` on any build predating it — a downgrade, or a rolling
  deploy where two versions read one store. The forward direction was already
  safe; unknown keys are now dropped, and logged at DEBUG so a misspelled one
  stays findable.

- **Stage metadata records an arc's compiled target** as `arc_target`, beside
  the `target` the author declared. The two differ for a subflow transition,
  whose `_subflow` sentinel compiles to a self-loop, and re-deriving that rule
  at each reader is the mirroring `arc_identity` exists to end.

- **A subflow arc's target was derived in two places.** `_translate_transition`
  and `_register_inline_conditions` each computed the self-loop rule, the second
  under a comment instructing its reader to keep it in step with the first. A
  drift between them would have silently unregistered a condition function — the
  arc's `FunctionReference` simply would not resolve. Both now call
  `arc_identity`.

- **A snapshot's payloads are copied in depth, so writing through one cannot
  reach what it was taken from.** `WizardStateSnapshot` is documented
  read-only and its payloads were copied one level. A shallow copy isolates
  only the top, so writing through `snapshot.stage_metadata["schema"]` reached
  the FSM's live stage configuration and reconfigured the running wizard for
  every later turn — silently, and for the life of the process.

  Three sites now copy through `dataknobs_common.copy_structure`:
  `get_state_snapshot`'s `stage_metadata`, `snapshot_from_metadata`'s `data`
  (which read the persisted dict directly, so a nested collected value aliased
  conversation metadata), and `to_tool_view()`, which hands both to a tool.

  `WizardFSM.stages`, `current_metadata` and `stage_metadata_for` are
  **unchanged**: they still return the live stage dict, deliberately and on
  the per-turn path, where a deep copy would charge every turn for a guarantee
  no caller there asks for. `stages` already documented that a caller
  intending to edit a stage it read must copy it; the snapshot is that caller,
  and now does.

  `tasks` is left as it was — it is rebuilt from `WizardTaskList` on every
  call. `get_state_snapshot`'s `data` needed no change either:
  `_get_wizard_state` already deep-copies it out of the metadata, so what the
  snapshot copies is a per-call transient.

- **A snapshot's `suggestions`, roadmap and recorded transition data no longer
  alias the conversation they were read from.** Three payloads reachable from
  the same read-only object were still the live ones, and each is fixed where
  the alias was created rather than where it surfaced.

  `normalize_wizard_state` now copies the containers it hands out. It read
  `data`, `history`, `suggestions` and `stages` straight off the metadata and
  put the same objects in its result — so `snapshot_from_metadata`'s
  `suggestions` was the persisted list itself, with no copy at any level, and
  its `stages` was copied only at the outer list, leaving every roadmap entry
  shared. Appending a quick reply to a snapshot, or marking a stage complete
  through one, rewrote conversation metadata that the next turn reads back.

  `WizardStateSnapshot.from_dict` now copies the containers it reads. It is
  the type's third constructor and the only one that bound its payloads to
  the caller's dict, so a snapshot restored from persisted JSON shared `data`,
  `stage_metadata`, `suggestions`, `stages`, `history`, `tasks` and
  `available_task_ids` with whatever had been loaded. "Read-only" is a claim
  about the object, so it has to hold however the object was built.

  `TransitionRecord.from_dict` now copies `data_snapshot`. It bound the field
  by reference from a dict the caller does not own, and all four of its call
  sites deserialize one — three of them straight out of `manager.metadata`.
  This exposed **both** snapshot constructors, the live one included, because
  `_get_wizard_state` restores transitions through the same classmethod.

- **`DynaBot.get_wizard_state()` returns a dict the caller owns.** It is the
  second consumer of `normalize_wizard_state`, and on the in-memory fast path
  what it normalized was the live metadata — so writing into the returned
  `data`, or into a roadmap entry in the returned `stages`, rewrote persisted
  conversation state through a public read API. Fixing the reader rather than
  either caller is what closes this and the snapshot together.

## v0.12.0 - 2026-08-26

### Changed

- **`preview_config` now reports whether the config it renders is valid.** The
  preview returns `valid`, `errors` and `warnings` beside whatever it rendered —
  in all three formats — using the same keys `validate_config` returns, so the
  two are comparable directly. Wired to the same `builder_factory`, the two
  tools now cannot disagree: the verdict comes from `builder.validate()`, the
  same validator `build()` and `build_portable()` run.

  Before this, `preview_config` rendered a config and said nothing about it,
  while `validate_config` refused the same wizard data in the same turn and
  `build()` raised — so there was no final config, and the tool's own
  description promised the model *"what the final config will look like."* That
  sentence is now *"Reports whether it is valid, and shows it either way"*; the
  catalog entry, which never made the promise, is unchanged.

  It reports **and** renders. A config with errors is still the thing being
  built, and seeing it is how an author works out what the errors are about — so
  the verdict is carried alongside the render, never in place of it. A validator
  that raises is reported as the verdict rather than costing the render.

  **Consumers reading the preview's output should expect the three new keys.**
  A caller that treated the result as exactly its rendered keys will see
  `valid`, `errors` and `warnings` arrive alongside them. The keys accompany a
  render, so the two paths with no config to render — no wizard data, and a
  `builder_factory` that raises — are unchanged: they still return `{"error":
  ...}` and carry no verdict, because on those nothing was validated.
  `validate_config` reports its own failures as `valid: False`, so the two
  tools' *failure* shapes still differ even though their verdicts no longer can.

- **Config validation now enforces the `$resource` marker rule at every depth,
  so more configs are reported invalid than before.** This is stricter, not a
  narrowing: a config that validated yesterday and carries a marker typo below
  the top level of a component is now reported invalid, which is the only way a
  consumer notices the gap is closed. Everything newly reported already failed
  at resolution — it simply failed later, in whichever deployment lacked the
  resource, instead of at config-lint time.

  `ConfigValidator` carried its own transcription of one clause of the rule,
  applied to one mapping. It agreed with the resolver about a reference section
  handed to it directly and disagreed about everything else: a reference nested
  inside one (`knowledge_base.vector_store` and every sibling), a misspelled
  `$resource` selector, which leaves `$required` stranded on an ordinary dict
  that then reaches a factory with its markers attached, and any section no
  schema is registered for — the component loop visits registered names only,
  so a `$resource` block under any other key was read by nothing at all.

  The transcription is gone. `DynaBotConfigSchema.validate()` calls
  `dataknobs-config`'s `collect_marker_violations()` once over the whole `bot`
  block, and `validate_component()` calls it on the subtree, rooted at the
  component name so a finding names a path the reader can find. Messages are the
  resolver's own sentences: one defect described one way, whether it surfaces as
  a lint or as a failed build. `marker_violations_result()` is exported for a
  consumer composing a validator pipeline of its own.

  `_validate_against_schema`'s docstring claimed "nested property validation"
  it has never performed, which is the whole explanation for how a rule applied
  there came to be a rule applied at depth 1 of a tree; it now says what it does.

### Fixed

- **`WizardConfigBuilder.add_subflow_network()` now produces a subflow the
  loader can read.** Each value under `subflows:` is a whole wizard config --
  `{name: ..., stages: [...]}` -- which is what `WizardConfigLoader` hands to
  `load_from_dict` and what the subflow guide documents. The builder collected
  a bare list of stages, so neither direction worked: `to_dict()`, whose
  docstring promises loader compatibility, emitted a shape `load_from_dict`
  refuses with `Wizard config must have 'stages' field`, and `from_dict()`
  iterated a documented `subflows:` section as though it were a list and got
  its keys, raising `dictionary update sequence element #0 has length 1`. So a
  wizard built with the builder could not declare a subflow, and a wizard YAML
  that declared one could not be read back into the builder -- which
  `from_file()` is the documented way to do.

  Callers still pass stages alone; the wrapping happens once, on the way in.
  The `WizardConfig.subflows` field is now typed as the configs it holds. The
  method had no caller and no test anywhere in the tree, which is the whole
  explanation for how a public method that round-trips through neither
  direction survived; the round trip is now pinned in both.

- **A subflow's `is_end` stage now renders its `response_template`.** The
  stage was entered and left inside one turn -- reaching it is what makes the
  subflow poppable -- so the pop ran in the same step and the parent's return
  stage rendered instead. The end stage's template was parsed, validated, and
  on screen nowhere.

  The cost is not the usual missing line. A subflow that can fail ends on a
  stage whose whole job is to say *nothing was saved, and here is why*, and
  that refusal was the one message that never appeared: the flow discarded
  the work and reported success. A completion message is the natural thing to
  put on an end stage, and it silently was not one.

  The template renders **before** the pop, against the subflow's own data and
  under its own stage name, and is prepended to the turn ahead of whatever the
  parent's return stage renders. A value that exists only inside the subflow
  interpolates correctly; the pop then replaces the data with the parent's, so
  fields the parent needs still travel through `result_mapping` as before.

  Both pop paths render, not just one -- the post-transition step and the
  auto-advance loop pop through the same method now, so an end stage reached
  by `auto_advance` says the same thing as one reached by an ordinary
  transition. Unchanged: `auto_advance: true` on an end stage still does
  nothing, the auto-advance loop excluding end stages by design; and `prompt`
  is not among the templates a departing stage can offer -- only
  `greeting_template`, `response_template` and `clarification_template` are --
  so an end stage carrying only a `prompt` is still silent.

  A template that fails to render no longer takes the departure with it.
  Putting a render in front of the pop put it in front of a structural step
  that never had one, and the render is reached for the first time by this
  change: an end-stage template that raises -- `{{ data.x }}` does, the
  render context exposing collected values as top-level names and defining no
  `data` -- would have escaped before the pop, leaving the subflow unable to
  exit on that turn or any later one. The message is decoration and the
  departure is structural, so a failed render is logged and contributes
  nothing. The same guard covers the auto-advance loop, which collects a
  stage's message before stepping past it.

- **`skip_default` no longer has to overwrite a value the user set.** The
  block was applied with a bare `dict.update`, which cannot be asked to do
  anything else: a key the user set five turns ago was replaced exactly as
  readily as one never touched, with no log line and nothing left to say the
  value had ever been different. Every downstream reader -- conditions,
  transforms, emission, templates -- then saw the stage's default as though
  the user had chosen it.

  A stage now declares `skip_default_mode: fill` to write only where a key is
  unset, and a key may state its own mode with `{value: ..., mode: fill}`
  where the block's is wrong for it alone. Both directions are needed in one
  block: an option the user configured must survive the skip that saves it,
  while a flag guarding an unconfigured branch must be cleared by that same
  skip or the user is pushed back into the branch they were leaving.
  `overwrite` remains the default, so a block that names no mode behaves
  exactly as it did.

  "Unset" is the reading the rest of the package already uses -- a key is set
  when its value is not `None`, which is what `has()`, the confidence gate and
  schema-default application all ask. A key extraction left holding `None` is
  one `fill` writes.

  A mapping is an annotation **only when it names exactly `value` and `mode`**,
  so a nested default keeps meaning what it reads as: `{provider: "x"}` names
  no `value`, `{value: "", label: "Email"}` names one but would lose `label`,
  and `{value: 3}` names nothing an annotation needs, since an entry declaring
  no mode takes the block's anyway. A mapping that names one of the two modes
  without being an annotation is reported and then written as the value it
  reads as, because `{values: false, mode: fill}` is a typo whose silent
  reading puts a *truthy* mapping where the author wrote `false`. Keys whose
  value is actually replaced are logged at DEBUG; a default equal to what was
  already there has replaced nothing and is not reported. Values are copied on
  the way in, so a transform editing a nested default cannot reach the loaded
  config the next conversation starts from.

  **One config shape changes meaning:** a `skip_default` key whose value is a
  mapping naming *exactly* `value` and `mode` was a nested default and is now
  read as an annotation. That collision is irreducible -- the two are the same
  text -- so wrap it to say otherwise:
  `knob: {value: {value: 3, mode: "off"}, mode: overwrite}`.

  Two things this makes explicit rather than incidental. The skip marker
  `_skipped_<stage>` is written **before** any default lands, and that
  ordering is now a documented guarantee -- it is what lets anything running
  on the skip turn tell the user's own value from the stage's. And a
  `skip_default` of the wrong shape is reported instead of dropped: the
  `isinstance(..., dict)` guard has silently discarded scalars since the field
  was introduced, while the config builder declared the parameter `bool | None`
  and the package's own documentation showed a string -- so an author following
  either got a stage that quietly did nothing on skip.

  `SkipDefaults`, `SkipDefaultEntry` and the mode constants are exported from
  `dataknobs_bots.reasoning`, so a consumer can name what `get_skip_defaults()`
  returns. `SkipDefaults.from_stage()` is the constructor for an authored
  block; `from_dict()` takes the projected `{"entries": ...}` shape and now
  rejects an authored one rather than yielding an empty block that applies
  nothing.

- **A stage field left unset is no longer reported as ill-typed.**
  `WizardFSM._stage_field` replaces a wrong-typed value with the field's
  documented default and warns; an *absent* field reaches it as `None`, which
  is not a wrong type but the registry's own marker for "not declared". Any
  accessor whose default is not `None` therefore accused every config leaving
  the field out. No accessor could reach the *absent* case before -- every
  shipped one asks for a field the registry already defaults -- so nothing
  warned in practice, but it made `_stage_field` unusable for exactly the
  fields most worth reading through it. A field authored as an explicit
  `null` did reach it, and is now read as unset rather than reported as
  ill-typed, which is what a YAML `null` says.

- **Stage-dependent state resolves against the FSM that owns the stage.**
  `WizardNavigator` holds both the main FSM and the subflow manager, and each
  of its methods picked one by hand; five picked the main FSM, which inside a
  push does not have the current stage. Asking it returned an empty metadata
  dict, indistinguishable from a stage that declared nothing -- so a subflow
  stage declaring `can_skip: true` was told it was required, its own
  `navigation.skip.keywords` were never found (the wizard-level defaults
  applied instead), back landed on the right stage and then rendered one with
  no prompt, schema or template, and an amendment jump to a subflow stage
  found nothing. `WizardResponder` picked by hand too: a custom
  `settings.context_template` rendering `can_skip` or `can_go_back` was handed
  the main FSM's answer, so the system prompt told the model a skippable
  subflow stage was required. The same stage config was correct standalone and
  a dead end when pushed, with nothing in the config to say so.

  Every site now asks `SubflowManager.fsm_for_state()`, which derives the
  active FSM from the wizard state's subflow stack -- one rule, in the class
  that owns the stack, correct both during a turn and outside one.
  `WizardFSM` grew `stage_metadata_for()`, `has_stage()` and
  `find_stage_owner()` so callers stop indexing another class's private
  attribute. **Note for existing configs:** stage-level keywords replace the
  wizard-level ones per command, as they always have outside a subflow, so a
  subflow stage declaring `skip.keywords` now answers to those words and no
  longer to the default `skip` -- keep `skip` in the list to have both.

- **Amendments resolve a section against the whole flow, and unwind to reach
  it.** The section-to-stage table confirms the wizard actually has the stage
  it maps to, and asked that of one frame -- so the main flow never found a
  stage living in a subflow, and once a subflow was pushed it stopped finding
  main-flow stages, which is reachable because `complete_wizard` sets
  completion with no subflow guard. Membership is now asked of the whole flow
  tree (`WizardFSM.find_stage_owner`), because "is this a stage of this wizard"
  is a property of the config and not of where the user stands. Acting on the
  answer is separate and does read the stack: an amendment whose target lives
  in the main flow while a subflow is open unwinds the subflow first, rather
  than restoring the subflow's FSM to a stage it does not have; one naming a
  stage inside some *other* subflow is declined and logged, since entering a
  subflow needs a parent stage and a data mapping an amendment does not have.

- **Restart inside a subflow leaves the subflow.** `restart_cleanup` reset the
  main FSM and cleared data, history and banks, but left the subflow stack
  loaded and the active subflow FSM set. The wizard then reported the main
  flow's start stage while rendering the subflow stage's prompt, schema and
  template -- and could not recover: `should_push` declines while already in a
  subflow, and `should_pop` needs an end stage of the subflow, which the main
  flow's start stage is not. Restart, the escape hatch of last resort, was
  what wedged the wizard. It now unwinds the stack before restarting, through
  `SubflowManager.unwind_all()` -- which also **records a `subflow_pop` for
  each frame it tears down**, so the transition trail no longer holds a
  `subflow_push` that nothing closes. A consumer pairing those records, or
  reconstructing depth from them, was wrong in exactly the case this made
  reachable.

  Two further pieces of state survived the reset, because `replace_data({})`
  empties `data` alone: **task completion** (`tasks` round-trips through
  `fsm_state`, so a restarted wizard reported the previous run's completed
  tasks -- a checklist that starts full on a flow the user just asked to start
  over) and **`transient`** (merged into the metadata a UI reads). The task
  *list* is rebuilt from the config rather than emptied: the wizard still has
  the same tasks to do, none of them done.

- **The read-only state snapshot describes the subflow stage it is standing
  on.** `WizardReasoning.get_state_snapshot()` -- the documented way a UI reads
  wizard state -- asked the **main** FSM for the current stage's metadata,
  skippability and back-navigability, and inside a push the main FSM does not
  have that stage. A skippable subflow stage therefore reported `can_skip:
  False` and its `suggestions` came back empty, so a skip button disappeared
  and quick replies vanished for as long as the subflow was open. `stage_index`
  was wrong in its own way: the subflow's stage name is absent from the main
  flow's stage list and was reported as index `0`, a progress bar that jumps
  back to the start whenever a subflow opens -- while the same object's
  `stages` roadmap correctly marked the *parent* stage as current, so the
  snapshot contradicted itself. It now resolves through `_fsm_for_state()`,
  the state-derived accessor the rest of the class already uses, and reports
  main-flow progress against the parent stage that pushed the subflow.
  `suggestions` now also goes through the reader the canonical metadata
  writer uses, so a quick reply is Jinja-rendered (a UI was being handed the
  raw `{{ ... }}` as a button label) and type-checked (a `suggestions:`
  written as a bare string became one button per character).
  `total_stages`, `data`, `history` and the task fields are unchanged.

  **The mixed frame is documented on the type.** Inside a subflow the snapshot
  answers `current_stage`, `can_skip`, `can_go_back` and `suggestions` for the
  subflow stage, while `stage_index`, `total_stages` and `stages` stay on the
  main flow and report the parent that pushed it. That table now lives on
  `WizardStateSnapshot` itself as well as in the observability guide, so a
  reader arriving through the API docs sees it.

- **`snapshot_from_metadata()` reports the same state its instance-method
  sibling does.** The static constructor -- the documented path for "you have
  the conversation metadata but not the `WizardReasoning` instance" -- rebuilt
  the stage-derived fields from `fsm_state` plus the caller's
  `stage_definitions`, ignoring the values the wizard had already derived into
  the same metadata dict one level up. Two consequences. `can_skip`,
  `can_go_back` and `suggestions` were never passed to the constructor at all,
  so they took the dataclass defaults (`False`, `True`, `[]`) in **every** flow
  -- a UI on this path never showed a skip button and never showed a quick
  reply, subflow or not. And inside a subflow the recomputation looked for the
  subflow's stage name among the main flow's definitions, found nothing, and
  reported `stage_index: 0` with no stage marked `"current"` in the roadmap.
  All six fields now come from the metadata the wizard wrote.
  **`stage_definitions` is unchanged for callers who need it:** it is the
  fallback for metadata predating those fields or built by hand from
  `fsm_state`, and is simply not consulted when the metadata is current.
  `data` and `history` deliberately still come from `fsm_state`, so the two
  constructors agree on those as well -- and are now **copied** out of it, as
  the instance method already copied them. Returned by reference, a consumer
  appending to `snapshot.history` on a type documented read-only silently
  rewrote persisted wizard state.

- **A `navigation:` block is type-checked before it is used, wherever it was
  written.** The block is authored config and reached its readers uncoerced,
  and every level of it was consumed without a check. `navigation: "yes"`
  raised `AttributeError` on an ordinary turn (and, at wizard level, a
  `ValueError` about *dictionary update sequences* out of `dataknobs-common`,
  naming neither the wizard nor the field); a command declared as a scalar
  raised one level down; `keywords: [1, 2]` raised out of `.lower()`; and
  `enabled: "false"` is a **truthy string**, so a command the author turned off
  stayed on while the field held a `str` on a dataclass declaring `bool`.
  Quietest and worst: `keywords: "done"` was *iterated*, arming `d`, `o`, `n`
  and `e` as four one-letter keywords, so a user answering `d` triggered a
  command meant for `done` -- nothing raised, nothing was logged, and the
  config read correctly.

  A field that cannot be read now falls back to its documented default **alone**
  -- a bad `skip` does not discard a good `back` -- and is reported at WARNING.
  This is the contract `WizardFSM` gained for wrong-typed stage fields, applied
  to the one config block that has two readers.

  **Every reader now shares one implementation.** Wizard-level
  `settings.navigation` and a stage's own `navigation:` block had a copy of the
  merge logic each, and both copies had all four defects; they now call
  `NavigationCommandConfig.normalize_raw()`, so what a field means cannot
  depend on which of the two places it was written in. The stage-level report
  is de-duplicated per stage and field, because that reader runs on every
  navigation check of every turn.

  The same is true of `keywords` wherever it appears. Four things in this
  package are authored as a keyword list and then *iterated*, and they now
  share one predicate (`is_keyword_list`) while responding at the layer each
  belongs to:

  - `NavigationCommandConfig.__post_init__` is the narrowest path every writer
    goes through, including `from_dict` -- the base coercion passes a `str`
    through untouched, so a string reached the constructor and became a tuple
    of characters there. Constructing one with a non-list now raises
    `TypeError`; authored config never reaches it, because `normalize_raw`
    substitutes the documented default first.
  - `intent_confirm:` **rejects** a wrong-typed `intents.<name>.keywords` at
    load, beside the shape checks it already ran there -- nothing has started
    yet and an author can still fix the file.
  - A hand-rolled `intent_detection:` block, which no synthesizer validates,
    **drops** an unusable override at classification time (falling back to the
    classifier's own vocabulary, which is what declaring no keywords already
    means) and reports it once per intent. With `per_intent_booleans`, a
    one-character message had been matching an intent meant for a word,
    writing its flag and firing its transition.

- **A subflow guard now reads what its own stage prepared.** The condition on
  a `_subflow` transition was evaluated *before* the stage's pre-transition
  preparation ran, while every other transition condition is evaluated after
  it. A guard reading a key written by its stage's `routing_transforms:` or by
  a transition's `derive:` block therefore could not fire on the turn the key
  was written; it fired on the next one, against a message the user had
  written in answer to a prompt they never saw -- which was extracted against
  the wrong stage's schema and discarded when the push finally replaced the
  data. The guard is now a step of a shared pre-transition sequence
  (`_prepare_and_route`), after the preparation and still before the FSM step,
  so a push continues to pre-empt the self-loop arc a subflow transition
  compiles into. `wizard-subflows.md` states the resulting visibility boundary
  per writer.

- **`advance()` can push a subflow.** The non-conversational API ran the
  pre-transition preparation but never asked whether a subflow should be
  pushed, while reaching `should_pop` through the shared post-transition
  sequence -- so it could be carried *out* of a subflow it had no way to enter.
  Given one config and one data value it stayed where `chat()` pushed. Both
  paths now run the same sequence.

- **A wizard tool's writes to collected data are no longer discarded.** A
  `ContextAwareTool` running in a wizard turn reached wizard state through
  `ToolExecutionContext`, which rebuilt it from the *persisted* conversation
  metadata. The wizard rewrites that metadata from its own state when the turn
  is saved, so anything the tool wrote there was overwritten -- while the tool
  reported success. `WizardReasoning` now publishes its live state on
  `ConversationState.live_wizard_state` for the duration of the turn, so a
  tool's writes land in wizard state and survive the save. The channel is
  cleared beside `turn_data` when the turn tears down.

- **A flow change mid-turn no longer strands a tool on abandoned data.**
  `WizardState.data` was *rebound* on a subflow push, a subflow pop and a
  restart, so a flow change between the start of a turn and a tool call left
  the tool reading the collected data of the run the user had just finished
  and writing where nothing would read it again -- both indistinguishable
  from success. `begin_turn` auto-restarts a completed wizard when
  `allow_post_completion_edits` is off and then continues the turn, which is
  the shortest path to it. The three sites now call the new
  `WizardState.replace_data()`, which empties and refills the dict in place.

- **A wizard tool's reads are no longer a turn behind.** The same rebuild made
  every read as old as the last save, which was hardest to diagnose where a
  stage declared `tool_result_mapping`: `params:` read live state, so the tool
  was *called* with this turn's values and *read* the previous turn's. One
  call, two channels, one turn apart. Both now read the same state.

- **A tool on the first turn of a wizard that does not greet gets real state.**
  There was no `"wizard"` key in metadata until the first save, so
  `context.wizard_state` was `None` and the shared accessor handed back a fresh
  empty dict on every call -- a tool wrote into a throwaway and reported
  success. Two shipped behaviours were wrong because of it: KB resources added
  on the first turn vanished, and `AddBankRecordTool` /
  `UpdateBankRecordTool` stamped `source_stage` / `modified_in_stage` as `""`
  on the first record of every such conversation, against a comment stating
  the opposite intent. The bank tools needed no change; they start working
  because the strategy publishes.

- **The KB tools report missing wizard state instead of writing into nothing.**
  The five tools in `kb_tools` now reach wizard data through the public
  `ToolExecutionContext.wizard_data()` and return an error result when it is
  `None`. The private reference-returning accessor they shared with
  `config_tools`, and the cross-module private import that reached it, are
  gone; the read-only accessor `config_tools` uses for its own three tools
  remains and still hands out a copy.

- **`AddBankRecordTool`'s documented constructor arguments are the real ones.**
  `tools.md` showed `banks_override=` / `catalog_override=` /
  `artifact_override=`, which are the internal helper's parameter names. The
  constructor takes `banks=` / `catalog=` / `artifact=`; the example as
  published raised `TypeError`.

- **An abandoned stream no longer leaks a turn's `turn_data` into the next
  one.** `ConversationState`'s per-turn channels are documented as cleared
  when the turn completes, but the cleanup ran in turn *finalization*, which
  `stream_chat()` skips when the stream was not fully consumed -- deliberately,
  so partial output is never written to history. A caller that broke out of
  the stream, or a client that disconnected, therefore left `turn_data`
  populated on the cached manager, visible to anything reading that manager
  before the next turn overwrote it. The cleanup now runs in the `finally`
  every turn driver executes, so it covers the success, error and
  stream-abandon paths alike.

### Changed

- **`WizardConfigBuilder.stage()` no longer declares a `skip_extraction`
  keyword.** `skip_extraction` is a per-turn state flag set by auto-advance,
  never a stage config field, so the loader discarded what the keyword wrote
  and warned about an unrecognized field. Callers passing it are unaffected --
  it lands in `**extra_fields` and behaves exactly as before. The typed
  `skip_default` parameter changed from `bool | None` to
  `dict[str, Any] | None`, which is the only shape the runtime has ever
  honoured. A registry-sync test now asserts that every explicit `stage()`
  keyword names a field the loader reads, so the class of drift this closes
  cannot reopen silently.

- **`WizardFSM` no longer reports a matched subflow transition as "none
  matched".** A subflow transition compiles to a self-loop arc, so a matched
  one leaves the FSM where it started -- indistinguishable, in the DEBUG log,
  from a condition that failed. The step-outcome log now decides from the
  step's own reported transition rather than from what the stage declares: a
  step that matched nothing still says so -- the ordinary case, since a guard
  that carries pushes the subflow and a push skips the step entirely -- and a
  step that stood still on an arc names that arc and the subflow transitions
  the stage holds, whose push is decided before the step. The message was
  written twice, once in `step` and once in `step_async`; both now call one
  `_log_step_outcome`.

- **A declined subflow push says so.** `SubflowManager` logs the guard's
  decision at DEBUG on both branches, naming the conditions that were asked.
  A decline previously left no trace at all, so it looked the same as a stage
  with no subflow transition, as a misspelled condition, and as one that
  raised.

- **A `WizardFSM` stage accessor returns the type it declares.** Stage
  metadata is authored config carried through uncoerced, so a stage written
  `can_skip: "no"` gave a *truthy string* from a method declared `-> bool` and
  the stage the author marked unskippable was skippable; `tools:` written as a
  bare string iterated character by character. The seven typed accessors now
  share one `_stage_field` read that returns the field's documented default
  when the authored value is of the wrong type, and warns once per stage and
  field. `get_transition_condition` reports a non-string condition as `None`
  rather than writing it into a transition record as the expression that
  fired, and `resolve_function` treats a non-callable registry entry as
  absent instead of handing it to a caller that will call it.

- **Loading a config with an ill-typed text field no longer raises.** Two of
  the loader's *warning* heuristics — the Python-format check over the
  template fields and the natural-language check over conditions — searched an
  authored value with a regex directly, so a non-string prompt or condition
  took the whole load down with a `TypeError` out of `re`. A check that exists
  to advise about a config now skips what it cannot read.

### Added

- **The loader now says which of a subflow's config is inert.** Two config
  surfaces parsed, validated and read as correct while doing nothing, with no
  report at load, at runtime or in a log. Both are now warnings from
  `WizardConfigLoader`, which loads every subflow through the same entry point
  the top-level wizard uses.

  **A pushed subflow's `settings:` block is never read.** A wizard's settings
  are hoisted once, off the top-level flow, into the collaborators built from
  them -- the extractor holds `extraction_scope`, the navigator the merged
  navigation config -- and those outlive any push. So a subflow declaring
  `extraction_scope: current_message` runs under whatever the parent declared,
  including while its own stage is current. Honouring the block would mean
  rebuilding that collaborator graph on every push and pop, so the config is
  answered where it is authored instead: the warning names the keys it found
  and points at `extraction_scope` and `auto_advance`, which are stage fields
  and *are* read from the flow a push made active. The same file loaded on its
  own is a wizard and honours every key, so the warning is about how the config
  is being used, not about the config -- and a top-level wizard is not warned.

  **`auto_advance: true` on an end stage is never acted on**, in a subflow or
  out of one. `can_auto_advance` returns `False` for any stage carrying
  `is_end` before it reaches the schema or the transition conditions; the
  exclusion is deliberate, since advancing out of a flow that has ended has
  nowhere to go. `auto_advance: false` on an end stage is not reported: it asks
  for what already happens.

  Both are warnings and neither refuses the config, which is this validator's
  contract for all eight of its checks. A subflow's `settings:` is not *wrong*,
  it is *unread*.

- **`BotTestHarness.create(custom_functions=...)`** threads transition
  functions -- routing transforms, transforms, validators -- into the wizard
  reasoning config, so a test exercising them stays on the harness rather than
  hand-building a bot config. **`WizardConfigBuilder.transition(derive=...)`**
  declares a transition's derivation rules.

- **`WizardState.replace_data()`** -- replaces a wizard's collected data in
  place, keeping the dict's identity. `WizardState.data` is handed out by
  reference, so code holding it across a subflow push, a subflow pop or a
  restart keeps the dict the wizard is actually using.

- **`greeting_template`: a wizard stage field for an opening line the stage
  says once.** It renders on the turn the stage first speaks — the start stage
  on `greet()`, any other stage on the turn it is entered — and is then stepped
  over, whatever the stage's mode.

  A structured stage had no way to open with fixed text. Its
  `response_template` is the stage's *response*, deliberately re-rendered every
  turn so a review summary tracks the data behind it, and pressed into service
  as a greeting it repeats the same sentence for as long as the wizard stays
  there. The only escape was `mode: conversation`, which turns extraction off.
  A `greeting_template` opens the stage and steps aside: extraction,
  confirmation and transitions behave exactly as they would without it.

  Two things follow from "steps aside" and are worth stating, because a shared
  counter would break both. Greeting a stage does not consume the first render
  its `confirm_first_render` is waiting for, so a stage that greets still
  confirms on the user's first answer. And a greeting delivered on the subflow
  push path — which deliberately leaves the render count at 0, so the pushed
  stage's template reads as an unanswered question — is still recorded as
  delivered, so it is not said twice.

  Available as `greeting_template` on the stage config, on
  `WizardConfigBuilder.add_structured_stage()` and `.add_conversation_stage()`,
  and on the test builder's `stage()` (which also gains the
  `clarification_template` parameter it was missing). The strategy-level
  `greeting_template` under `reasoning:` composes with it — see *Changed*.

- **The loader reports a `response_template` that a `greeting_template` has
  made unreachable.** On a `mode: conversation` stage both fields mean "first
  render" and the greeting wins, so the `response_template` beside it never
  appears. That is the same silent inertness the new field exists to remove,
  so it is named at load time — a `WARNING` on the stage, pointing at
  `clarification_template` for the later turns. `WizardConfigBuilder.validate()`
  reports the same warning, so a config built in Python hears it before it is
  ever loaded.

  The loader's existing `str.format()`-vs-Jinja2 check now covers
  `greeting_template`, `clarification_template` and `confirmation_template` as
  well as `response_template` and `prompt`; it had enumerated template fields
  by name and never caught up. Same for the builder-side copy.

- **Wizard transition conditions are checked when the wizard is loaded.** A
  condition the expression engine will refuse — multiline, a syntax error,
  dunder access — is reported once, at load, as a `WARNING` naming the stage,
  the target and the reason. The wizard still loads and the transition still
  registers: this is a report, not a rejection, so one unusable condition does
  not take the rest of the config down.

  Nothing said so before the condition was evaluated, and by then a refusal is
  indistinguishable from a condition that is merely unsatisfied — every wizard
  condition site passes `default=False`, so both are `False`. A stage guarded
  by a refused condition therefore never advances, and the person who can fix
  it is the config's author, who is no longer in the loop by the time a turn
  runs. Load time is the last moment the report reaches them.

### Changed

- **`greeting_template` is declared once for the reasoning-strategy family, and
  read from one place.** A new `ReasoningConfig` base carries the field and
  every strategy config inherits it; `ReasoningStrategy.greeting_template` is
  a read-only property resolving the two routes that supply it — the typed
  config, and the constructor keyword a directly-subclassed strategy uses.

  `ReasoningStrategy` has always documented the field as universal, but each
  of the five configs declared it and each of the five strategies copied it
  onto itself. A strategy that skipped either half was not reported: a config
  class that omits a key does not reject it, it drops it. "Universal" was
  therefore a property every strategy had to re-establish by hand, and the
  wizard's config had already lost it once.

  No behaviour changes for any built-in strategy. Consumers writing their own
  strategy get a shorter obligation: inherit `ReasoningConfig` and the field
  arrives, instead of declaring it and binding it correctly. The
  constructor-keyword pattern in `custom-strategies.md` is unchanged and
  remains supported — and is now checked, along with the config route, over
  every registered strategy.

  Read the value through `ReasoningStrategy.greeting_template`.
  `_greeting_template` is private, is half of one route, and no longer holds
  the config value.

- **BREAKING: the five reasoning-strategy configs are keyword-only.**
  `SimpleReasoningConfig`, `ReActReasoningConfig`, `GroundedReasoningConfig`,
  `HybridReasoningConfig` and `WizardReasoningConfig` no longer accept
  positional arguments. Construct them by keyword, or by `from_dict` — which
  is how they are built from YAML, and which is unaffected.

  Inheriting `greeting_template` from a base is what forces it. A base field
  is declared ahead of every subclass field, so a defaulted one would sit in
  front of the wizard's required `wizard_config` and the class would be
  rejected at import. `kw_only` moves it behind the `*` instead — and that
  shifts each config's own fields one position left.

  Four of the five would have raised on a call written against the old order.
  The wizard would not: its second positional was `greeting_template` and
  became `config_base_path`, both `str | None`, so
  `WizardReasoningConfig(cfg, "Hello!")` would still have constructed, with no
  greeting and a nonsense base path. Nothing about the value distinguishes a
  greeting from a path, so only the signature can reject it. Making the whole
  family keyword-only turns every stale positional call into a `TypeError` at
  the call site rather than a wrong field somewhere downstream.

  Migration is mechanical: name every argument, including the first.
  `WizardReasoningConfig(wizard_config=cfg, greeting_template="Hello!")`.

- **`ReasoningConfig` is exported from `dataknobs_bots.reasoning`.** It is the
  base a consumer's own strategy config inherits.

- **Wizard bots honour the strategy-level `greeting_template`.** A wizard
  configured with `reasoning: {strategy: wizard, greeting_template: ...}` now
  opens with that line instead of ignoring it. It stands in as the start
  stage's `greeting_template` when that stage sets none, so it renders once and
  is stepped over — and a start stage carrying its own greeting still wins.

  `ReasoningStrategy` documents the field as universal and every other strategy
  reads it, but `WizardReasoning` overrides `greet()` and had nowhere to put
  it: until a stage could carry a greeting, "render this once at the opening"
  had no wizard-shaped meaning. So the value was discarded — silently on the
  config-driven path, where `WizardReasoningConfig` projected the unknown key
  away, and as a `TypeError` on the direct constructor.

  **This is a documented limitation lifted, not a bug fixed.** The old
  behaviour was stated in the configuration guide, which now documents the
  precedence chain instead. Nothing can have depended on the old behaviour —
  the value never reached anything — but a bot that sets the field today will
  start greeting with it.

- **A condition that fails on a turn's data is logged at `DEBUG` rather than
  `WARNING`.** `data['name']` before `name` has been captured is the ordinary
  state of a guard whose input has not arrived; warning on it every turn, for
  a config that is entirely correct, is how a log teaches its reader to skip
  it. `WARNING` is now reserved for a condition the engine refuses — which no
  data can satisfy and which an author has to change.

  `WizardResponder.evaluate_condition` makes the same split. Its conditions
  reach it from the auto-advance gate and from subflow guards, neither of
  which passes through the loader, so it has no load-time moment and runs the
  static check itself, on the failure path only.

  One consequence is worth stating: a condition naming something that never
  resolves — `artifact`, which the FSM arc does not bind — is a *runtime*
  failure by this split and so lands at `DEBUG`, where it used to warn. The
  load-time check reports what the engine will refuse, not whether every name
  in the expression exists. `configuration.md` now documents both the split
  and the `artifact` case.

- **The three places a bot builds an LLM provider from config call
  `create_llm_provider()`** — bot construction, summary memory, and the
  grounded query transformer — rather than
  `LLMProviderFactory(is_async=True).create(...)`. The two build the same
  object, but `is_async` is an argument on the function and a constructor flag
  on the factory, so only the function can say which provider comes back.
  Each site had been absorbing the difference in its own way: bot construction
  assigned the union to an attribute typed for one half of it and awaited
  lifecycle methods that exist on only that half, while the other two erased
  the result to `Any`, which reports nothing because it checks nothing. See
  the `dataknobs-llm` entries for the typing that made this possible.

### Fixed

- **`ErrorRaisingStrategy` accepts `greeting_template`, so building it from
  config no longer raises `TypeError`.** It is a direct `ReasoningStrategy`
  subclass, so it inherits the base `from_config`, which calls
  `cls(greeting_template=...)` — the base class's own universal field, which
  its constructor did not accept. Any config-driven construction of the
  shipped strategy failed on the way in. Its `greet()` still raises, which is
  what the construct is for; the template it now accepts is never rendered.

- **Every registered reasoning strategy is held to the universal
  `greeting_template` contract.** `ReasoningStrategy` documents the field as
  universal, but the family has no shared config base: each strategy config
  re-declares the field and each strategy re-binds it, and a strategy that
  skips either half fails silently, because an undeclared key is dropped
  rather than reported. The parity guard already in place cannot see it — it
  compares a config class against a constructor signature, and for a
  consumer-mixin adopter that signature is the mixin's variadic one, so the
  comparison is empty. A registry-driven test now asserts that every
  registered strategy accepts the field at construction and renders it from
  `greet()`, which also covers the three built-ins whose config-factory
  greeting round-trip was untested. Because it iterates the registry, a
  strategy a consumer registers is held to the same contract.

- **A conversation-mode stage renders its `clarification_template` when the
  turn is streamed, not only when it is buffered.** The two response paths
  each carried their own copy of the template-selection rule, and only the
  buffered copy was updated when `clarification_template` was introduced, so
  the same stage config nudged on `chat()` and called the LLM on
  `stream_chat()`. Both paths now go through one selection helper, so a
  template kind is added in one place rather than two.

- **A `clarification_template` set without a `response_template` renders.**
  The render count only advanced for stages that configured a
  `response_template`, so a stage with only a clarification template was
  permanently on its "first" render and the clarification branch was
  unreachable on every turn: the field loaded, validated, and did nothing.
  The count now advances for any stage whose template choice depends on it.
  Such a stage renders the LLM's answer on its opening turn, as before, and
  the clarification from the second turn on.

- **A stage the wizard auto-advances past no longer repeats an opening line it
  has already delivered.** The collector that captures a stage's output on the
  way past read `response_template` directly, which is the rule the two
  response paths follow only until a stage has spoken once. So a
  conversation-mode stage being advanced past re-contributed its opening line
  however many turns it had already taken, rather than its
  `clarification_template`. It now selects through the same helper as the
  buffered and streaming paths — the third and last copy of the rule.

  One consequence is worth stating: such a stage with no
  `clarification_template` now contributes *nothing* on a later pass, where it
  used to repeat itself. That is not what the turn would have said — the turn
  would have gone to the LLM — but a collector cannot call one, and the
  interstitial line is dropped rather than duplicated. The turn is never
  silent either way: these lines are only a prefix to the landing stage's own
  response, which is unaffected.

- **The wizard loader no longer prepends `return` to a condition before
  handing it to the expression engine.** The engine has done this since it was
  introduced; the loader's copy predated the engine and was never removed when
  the loader was pointed at it, so the rule was written in two places and both
  copies carried the same defect — a substring test where a token test was
  meant, which left `return_code == 0` unwrapped and silently `False`. The
  engine's half is fixed in `dataknobs-common`; this deletes the copy, so the
  rule is written once.

  A visible side effect: the expression logged alongside a failed condition is
  now the one the author wrote, rather than the loader's rewrite of it.

### Documented

- **The subflow push/pop lifecycle table said nothing about the end stage's
  render.** Its `Pop` row went straight from "subflow reaches an `is_end` stage"
  to `result_mapping`, so a reader following the table still had the belief that
  an end stage is silent -- the one the pre-pop render exists to falsify, and
  which the subflow guide contradicts at length. The row now names the render
  and its order.

- **The subflow guide now says which of a subflow's own config is live inside
  a push.** The rule is by level rather than by field: everything a subflow
  declares on a *stage* means the same thing inside a push as it does when the
  same file is loaded as a wizard of its own, because the stage carries its
  fields into the subflow's own FSM and every read of the stage in play goes
  to the FSM that owns it. A subflow's wizard-level `settings:` block is the
  exception and is never consulted -- settings are read once off the top-level
  flow when the strategy is built, and the collaborators built from them
  outlive every push and pop. The block is parsed and stored on the subflow's
  FSM, which is why nothing about it looks wrong.

  `navigation` is called out on its own, being the one word that appears at
  both levels while the levels disagree: a subflow stage's own `navigation:`
  block is live, and the same block written under that subflow's `settings:`
  is not. The guide points at the stage-level `extraction_scope` and
  `auto_advance` fields as the way to say per-subflow what `settings:` cannot,
  and notes that `auto_advance: true` on an `is_end` stage is inert for a
  reason unrelated to subflows. The configuration guide's push/pop lifecycle
  table now links here from its "normal wizard processing" line, which was the
  natural place to read the opposite.

- **`WizardFSM.stages` documented a stronger guarantee than it delivers.**
  Both the property and the configuration guide said it returns a copy "to
  prevent external modification". The copy is shallow, so it protects the
  table's shape and nothing else: adding or removing a key leaves the wizard's
  stage list alone, but the stage dicts are the live ones, and the natural
  `for name, meta in fsm.stages.items(): meta[...] = ...` edits the running
  wizard's configuration for the life of the process. `current_metadata` and
  `stage_metadata_for()` hand out the same live dicts with no copy at all.

  The documentation now states the boundary and shows the `deepcopy` a caller
  needs before editing a stage it read. The behaviour is unchanged and
  deliberately so -- the callers inside this package only iterate, one of them
  on a per-turn path, and a deep copy measures roughly 2500x the shallow one
  for a guarantee none of them asks for. Two tests pin the boundary, so
  documenting a stronger guarantee again fails in this package rather than in
  a consumer's wizard.

- **The multi-tenancy guides now say that the API they document is
  deprecated.** `BotManager` and the `dataknobs_bots.api` singleton helpers
  around it warn at runtime, so a sample pasted from either guide raised a
  `DeprecationWarning` on its first call — and neither page mentioned it. Both
  now open with the notice and show the registry imports that replace them.
  The guides still document the manager surface, because the registry API is
  not a rename of it: it registers and looks up bots rather than
  getting-or-creating them, and it owns an initialize/close lifecycle the
  manager did not have.

- **Grounded retrieval isolates sources from one another**, and the guide
  now says so: a source that raises is logged with its cause and dropped
  for that turn while every other source still contributes, and a source
  that is reachable but matches nothing contributes an empty list instead.
  The guard was already there and had no test; it was also unreachable
  through a `database` source or a `ClusterTopicIndex`, both of which
  absorbed their own failures — see the `dataknobs-data` entries for
  those. A topic index that absorbed its failure was read as a vocabulary
  gap, so the turn fell back to plain text retrieval instead of reporting
  a broken index.

- **The built-in tool table is the catalog again.** `config-toolkit.md` gave
  twelve rows for the twenty-one tools `default_catalog` registers, and stated
  twelve as the count above them; the site page repeated the count. The nine
  wizard tools — bank CRUD, artifact lifecycle, and the two completion signals
  — were therefore absent from the one page that claims to enumerate the
  catalog, and a subset is the worst shape this can take,
  because twelve rows under the words "all 12 built-in tools" reads as a closed
  set rather than an obviously partial one. The rows are in, both counts are
  right, and a workspace guard now compares the table and every documented
  count against the registry in both directions, so the next tool registered
  cannot land unlisted.

  An empty **Requires** cell also now says what it means: the tool declares
  nothing to be handed at *construction*, not that it needs nothing. Those nine
  read whichever of `banks`, `catalog` and `artifact` they use from
  `context.extra` on the turn that calls them, taking a constructor argument
  only as an override, which is why they declare no `requires` — by the time
  the value exists there is no constructor left to inject it into. Their
  parameters and effects stay documented in `tools.md`, which the table now
  points at.

- **What a custom `storage_class` actually has to provide.** Three pages
  listed implementing `ConversationStorage` and supplying an async
  `create(config)` classmethod as equal requirements, when only the second is
  checked. The loader resolves the path without an
  `issubclass` gate on purpose — duck-typed storage is meant to work — so
  `create` is the only thing it can insist on, and it now says which is which
  on each page. The same distinction reached the two comments in `bot/base.py`
  that still described a tool's `from_config` as receiving "simple
  YAML-compatible parameters": a declared `requires` dependency is injected
  into that dict as a live object before `from_config` sees it, and the phrase
  described the contract that a tool rebuilding the value from its YAML
  spelling was violating. Those were the last two sites describing it that
  way in the source.

### Fixed

- **A `storage_class` with no `create` names itself.** The config path
  resolved the dotted path and called `create` on whatever came back, so a key
  pointed at a resolvable class lacking the method failed with a bare
  `AttributeError` naming neither the config key nor the path that produced
  it. It now raises `ConfigurationError` carrying both, checked through a
  runtime protocol that asks only for the method the path calls — the duck
  typing the missing `issubclass` gate was there to preserve is unchanged.

- **Both tool-loop deliveries answer "what is pending" the same way.** The
  buffered delivery read `tool_calls` defensively for `has_pending()` and
  directly for `pending_calls()`, so a provider response object carrying no
  such attribute made the first return `False` and the second raise
  `AttributeError`; the streaming delivery was `None`-safe in both. Nothing
  reached it, because the loop asks the boolean first every time — but the
  declared `list[Any] | None` return advertised a `None` that one of the two
  would have thrown rather than produced. Both now return an empty list when
  nothing is pending.

- **`validate_config` and `save_config` now reach one verdict.**
  `ValidateConfigTool` built its own schema-less `ConfigValidator` and ran it
  over the builder's internal config, while `SaveConfigTool(portable=True)`
  ran the builder's own schema-aware validator. A config carrying a
  misspelled `$resource` marker was reported valid and then refused at save,
  with nothing in either message to reconcile the two — and it needed no
  consumer schema to reproduce, since `DynaBotConfigBuilder` supplies
  `DynaBotConfigSchema` when none is given. With a `builder_factory` present
  the builder's validator now decides, via the public `validate()` the tool
  had been reimplementing three lines at a time through a private method.

  The disagreement was symmetrical and only half of it was the tool's.
  `SaveConfigTool(portable=False)` — the constructor and `from_config`
  default — built through the unvalidated path, so it wrote to disk exactly
  the config `validate_config` had just refused, and reported success. Both
  settings of the flag now validate: `portable` selects the output shape,
  not whether the config is checked.

- **Every `ContextAwareTool` answers an omitted required argument.**
  `ContextAwareTool.execute` forwarded the model's arguments straight into
  `execute_with_context`, so any tool declaring a parameter `required` in its
  schema without defaulting it in its signature raised `TypeError` from the
  call itself when an LLM omitted it. Nine tools across `artifacts`,
  `kb_tools`, `knowledge_search` and `config_tools` carried that shape. The
  base class now checks the declared-required set before forwarding and
  returns a result naming what is missing; `GetTemplateDetailsTool` overrides
  `missing_arguments_result` to add the template names that would have
  worked. This was also the `[override]` incompatibility the type checker
  reported against the base signature at all nine sites.

- **A repeated wizard-validation message is emitted once.** `Duplicate stage
  name: 'x'` was emitted per offending stage, so three stages sharing a name
  reported it twice; `Stage 'x' has transition to unknown stage 'y'` was
  emitted per transition, so two transitions to one missing stage reported it
  twice. Both messages already name everything they have to say, so the
  repeat was countable but not readable. Fixed where they are emitted —
  once per duplicated name, once per `(stage, target)` pair.

- **The successors named by a deprecation are now importable from the same
  place as what they replace.** `dataknobs_bots.api` exported
  `get_bot_manager`, `init_bot_manager`, `reset_bot_manager` and
  `BotManagerDep` — each of whose docstring says to use the registry spelling
  instead — while exporting none of the four registry names, all of which were
  already defined in the same module. A consumer who read the
  `DeprecationWarning`, changed the name and left the import path alone got an
  `ImportError`, so the only working code was the deprecated code. The package
  root had the same gap: `BotManager` points at `BotRegistry` or
  `InMemoryBotRegistry`, and only the first of those was exported.

- **A memory bank no longer sends a table name to a backend that has no
  table.** `_create_bank_db` set `table` to the bank name unconditionally.
  That was meaningful for the SQL backends and meaningless for the file,
  memory and S3 ones, where the key was silently discarded — harmless until
  the backend configs began rejecting a key they do not accept, at which
  point a file-backed bank failed to build. The bank name is now offered
  only where the backend's own config says it is accepted.

- **A `database` grounded source only worked against the in-process store.**
  The source builds its own backend, and the one configuration needing no
  configuration was the only one it could express — which is also the only
  one under which the three gaps below are invisible.

  It forwarded `backend` and a `connection` string that no backend accepts
  under any spelling, dropping every key that names a store: a source
  declaring `backend: sqlite` with a file path got `:memory:`. It then never
  connected what it built, and a backend that needs connecting raises on
  every query — which `DatabaseSource` reports as an empty result set, so a
  source grounded on a real database returned nothing on every turn and
  nothing said so. And `schema.fields` written as the documented list of
  `{name, type}` mappings raised `AttributeError`, because the builder read
  only the mapping form.

  Options that are the source's own (`content_field`, `text_search_fields`,
  `schema`, `description`) stay with the source; every other option now goes
  to the database factory, which is the only code that knows what each
  backend takes. The backend it builds is connected, and both spellings of
  `schema.fields` are read. **Breaking** for a config carrying an option
  that no backend accepts and that was previously discarded: it is now an
  error naming the source and the key.

- **A tool handed the dependency it declares now uses it.** Five of the six
  built-in tools with a `requires` entry define `from_config`, and all five
  read only the YAML spelling of the key: `ListTemplatesTool`,
  `GetTemplateDetailsTool` and `SaveConfigTool` discarded the live object
  and built their own from a directory path, while `PreviewConfigTool` and
  `ValidateConfigTool` put it through `resolve_callable` and raised
  `DottedPathError` on a callable that was already resolved. Only
  `KnowledgeSearchTool` worked, and only because it defines no `from_config`
  and so reaches the constructor instead.

  The two channels meet in one dict — `DynaBot._resolve_tool` copies the
  objects named by `catalog_metadata()['requires']` into the same `params`
  the YAML block fills, and `ToolCatalog.instantiate_tool` puts its keywords
  there under any name at all — so every `from_config` has to tell them
  apart. Each of the five now does, preferring the live object and keeping
  its YAML path as the fallback. `SaveConfigTool` accepts a live `on_save`
  and `builder_factory` on the same terms, since the catalog can supply
  either. A guard parametrized off the catalog covers every entry that
  declares a dependency, so a tool added later is covered when it is
  registered.

  One end of this is still shut: `DynaBot.from_config()` builds the
  dependency map itself and puts only the configured `knowledge_base` in it,
  with no route for a consumer to add to it. Code that constructs tools —
  through `ToolCatalog.instantiate_tool`, `create_tool_registry(overrides=)`,
  or `from_config` directly — is where the live spelling reaches a tool
  today.

### Changed

- **A `ConfigValidator` passed to `ValidateConfigTool` is additional, not a
  replacement.** Its errors merge with the builder's instead of standing in
  for them, so a consumer supplying both gets strictly more errors, never
  fewer. "Instead" is what let the two tools disagree, and the failure
  directions are not symmetric: an extra error stops an author, a missing one
  misleads them. The parameter is now optional, and `from_config` no longer
  constructs one — a schema-less validator built by default is the construct
  whose verdict contradicted `save_config`. `catalog_metadata()` accordingly
  declares `requires: ("builder_factory",)` rather than `("validator",)`.

- **`ValidationResult.merge_unique` added; `merge` still concatenates.**
  Composing two validators that cover overlapping ground reports each shared
  failure twice — `validate_completeness` runs inside every `ConfigValidator`
  — and `merge_unique` is the operation for that case. `merge` is unchanged:
  whether a repeated message is one finding or two is a property of the
  composition, and the ~20 call sites that accumulate findings from a single
  validator would lose real findings if it decided otherwise.

### Added

- **`DynaBotConfigBuilder.build_unvalidated()`** — the public name for
  building the config without validating it. `build()` and `build_portable()`
  validate and raise; callers that want to *report* a `ValidationResult`
  rather than raise on one were reaching through `_build_internal` from
  outside the class, which the config-toolkit tools all did. The name states
  what separates it from `build()`, because the two are otherwise
  indistinguishable at a call site and this is the one that can hand back a
  config nothing has checked. Pair it with `validate()`. `_build_internal`
  remains the implementation and is unchanged.

- **`ContextAwareTool.missing_arguments()` and `missing_arguments_result()`**
  — the declared-required check and the result it returns, both overridable.
  A tool that can offer the model something to retry from (a list of valid
  names, say) adds it by overriding the latter.

- **`dataknobs_bots.config.injected_dependency()` and `InjectedCallable`** —
  the one line a `from_config` needs to tell a live dependency from config
  data, and the constraint to pass it for a key whose YAML form is a dotted
  path. Public rather than private because a consumer writing a tool with a
  `requires` entry has the same problem and no way to discover the answer
  otherwise. Handing it a constraint that is not usable as one — a bare
  function, an undecorated protocol — raises `TypeError` rather than
  answering "not injected", and does so whether or not the key is present,
  so the mistake cannot lurk until the first call that happens to inject.

## v0.11.0 - 2026-08-19

### Security

- **Deleting one knowledge base destroyed every tenant's ingest state.** A
  tenant context contributes a prefix to every state key, and that prefix's
  first segment landed at the same level as a knowledge base's own root.
  `BoundTenantContext` projects `tenants/{tenant_id}/_state/`, so `tenants`
  was an ordinary legal `domain_id` whose listing prefix contained every
  tenant's state for every domain — and both persistent backends delete by
  prefix, so `delete_kb("tenants")` removed all of it and returned `True`.
  On the file backend the overlap also made `create_kb("tenants")` report a
  knowledge base nobody had created as already existing.

  Neither existing guard sees it. The segment rule passes `tenants` because
  it *is* one segment; the prefix check passes `tenants/acme/_state/` because
  it is contained. Both are satisfied and the slot still collides — the same
  reading error the segment rule was introduced to close, one level up.

  Scoped state is now rooted under a `_scoped/` segment the layout owns, so
  the two namespaces are disjoint by construction, and a `domain_id`
  beginning with `_` is refused. A reserved-word list would not have worked:
  `PrefixedTenantContext` takes its whole prefix from a deployment's own
  configuration, so the colliding name is one this package never sees.
  **Breaking** for an existing multi-tenant deployment — per-tenant state
  keys move from `{prefix}tenants/…` to `{prefix}_scoped/tenants/…` and
  must be relocated, or the affected tenants re-ingest. Single-tenant
  layouts are untouched: a context contributing no prefix (`ctx=None` or
  `SingleTenantContext`) still composes exactly the pre-tenancy key.

  **Breaking, separately, for any deployment holding a knowledge base
  whose `domain_id` already begins with `_`** — an internal `_staging` or
  `_archive`, say. The reservation applies at every entry point, not only
  at `create_kb`, so such a base becomes unaddressable: `get_info`,
  `get_file`, `list_files` and `delete_kb` all raise for it, and
  `list_kbs()` stops returning it. Nothing is deleted — the objects and
  files are exactly where they were — but the API can no longer name
  them, so rename the base before upgrading. If one slips through,
  `list_kbs()` logs a WARNING naming it rather than dropping it silently;
  the layout's own `_scoped` root is skipped without one, since it is
  expected there.

- **A knowledge base could overwrite and delete another one, on every
  persistent backend.** `domain_id` is interleaved with the layout's own
  literal segments — `{domain}/content/{path}`, `{domain}/_metadata.json`,
  `{domain}/_snapshots/{version}` — and nothing constrained it to one of them,
  so a domain containing a separator addressed a *different* knowledge base's
  slots. `acme/content` composes exactly the key an ordinary content file named
  `_metadata.json` under `acme` composes: writing that file replaced the other
  base's metadata document with whatever the writer supplied. `delete_kb("acme")`
  removed the whole of `acme/content` with it, because both persistent backends
  delete by prefix. Executed end to end against real S3 and reproduced on the
  file backend.

  **Containment did not catch this**, and could not: `acme/content` never
  leaves the base, so the guard the file backend adopted last release passed it
  while the slot collided. The invariant is one segment, not one tree, and it
  now lives on the shared backend mixin
  (`dataknobs_common.safe_segment` under the hood) rather than in whichever
  backend noticed. **Breaking** for a deployment using a nested `domain_id` —
  but such a knowledge base was never visible to `list_kbs()` on either
  persistent backend, both of which enumerate one level, so it could be created
  and then not found. The three backends also disagreed about the whole
  sequence (S3 overwrote, file refused at `create_kb` for an unrelated reason,
  memory did nothing), so the rule is applied to all three: a name refused in
  production is refused in the backend you develop against.

  A resource `path` inside `content/` is unaffected. It names a location rather
  than occupying a slot, nesting is its purpose, and it stays bounded.

- **`S3KnowledgeBackend` composed every key unchecked.** Its file-backend twin
  was guarded last release and this one was not, so two backends that had been
  consistently unguarded became inconsistently guarded — the reasoning in the
  file backend's own guard (that the tenant prefix must be bounded because
  `PrefixedTenantContext` takes a consumer-supplied pattern) is entirely
  backend-independent, and this backend consumed the identical value. Now
  bounded at all of them: the resource `path` against the `content/` tree, the
  snapshot `version` against its lineage, and the formatted tenant state prefix
  against the backend's key namespace. The prefix check lives on the shared
  mixin rather than on this backend, because "all of them" was not true of
  `key_pattern` while each helper had to remember to call it — that one
  composed the raw prefix and returned a subscription pattern naming a
  location no write would ever produce, which fails by matching nothing and
  reporting success. A non-canonical prefix is refused rather
  than silently normalised, since rewriting it would move every state document
  for that tenant.

  "S3 resolves nothing" is not a defence: a `..` is stored literally, and the
  bucket is routinely read by something that does resolve — `aws s3 sync`, a
  CloudFront origin, this repository's own file backend over the same layout. A
  key that only misbehaves once it is copied is worse than one that misbehaves
  immediately, because nothing on the write path reports it.

- **One tenant could read another tenant's ingest state.** With a tenant
  context, `FileKnowledgeBackend` composes a state path in two hops —
  `base_path`, then the context's `state_key_prefix()` — and containment was
  judged against the outer one. A `domain_id` that walks *sideways* rather
  than out satisfies that check: under `_scoped/tenants/acme/_state/`, a domain
  of `../../bob/_state/proj` resolves to `_scoped/tenants/bob/_state/proj`,
  which never leaves `base_path` and is squarely inside the wrong tenant. Tenant `acme`
  could read tenant `bob`'s state-version token through the public
  `get_state_version` — with `acme`'s own view of that domain still empty.

  Each hop is now bounded against the hop before it, so a `domain_id` is
  contained to the tenant's own subtree and the tenant prefix is contained
  to `base_path`. The segment rule now refuses every nested spelling ahead of
  that check, so containment survives as the bound on the prefix — free-form
  text from a `prefix_pattern` — rather than as the constraint on the domain.
  Without a context the prefix is empty, the two hops collapse to one, and the layout
  is byte-identical to the single-tenant one.

- **A knowledge-base `domain_id` or resource path could address any file on
  the volume.** `FileKnowledgeBackend` composed both identifiers straight
  onto its base directory with no containment, and every sink was live:
  `create_kb("../elsewhere/x")` created directories outside the base,
  `delete_kb("../elsewhere/x")` `shutil.rmtree`'d a tree outside it and
  returned `True`, and a resource path reached `put_file`'s atomic write,
  `get_file`'s read and `delete_file`'s `unlink` in the same way. An
  absolute identifier discarded the base outright.

  Both identifiers are now bounded at `_kb_path` and `_file_path`, the two
  methods that compose them, and the path the guard returned is what gets
  opened, rather than a recomposition from the raw name. Two further sites
  compose on top of those and are bounded in their own right, because
  routing through a guarded method covers only what that method joined:
  `_snapshot_file` appends a caller-supplied `version` *after* `_kb_path`'s
  check (the value arrives from the public `list_changes_since` as a token
  the caller persisted and handed back), and `key_pattern` builds its glob
  as a string from `str(base_path)` and reaches neither guard — its output
  goes to `Path.glob` or an inotify watch, so an escaping `domain_id` there
  installs a watch over a tree the deployment did not choose. Nesting is unaffected: a `domain_id` of `team/alpha`, a resource
  path of `subdir/file.md`, and an interior `a/../b` that never leaves the
  base are all still legal. An escaping identifier raises `ValueError`
  before any filesystem call — including from the read-shaped `get_file`,
  `file_exists`, `stream_file` and `get_info`, which previously answered
  *successfully* about a file outside the base rather than returning
  `None`/`False`.

- **A wizard's subflow name could load a state machine from outside its
  config directory.** `WizardConfigLoader` probes `{base}/{name}.yaml` and
  `{base}/subflows/{name}.yaml` for a subflow, and the name comes from
  config *content* — a `subflows:` key or a transition's `subflow.network`
  value — so a wizard config naming `../../elsewhere/other-wizard` pulled in
  a foreign FSM together with its transitions, transforms and function
  references. Both probes are now bounded to the wizard's **config tree**,
  fixed where the load started; the `subflows/` layout the second probe
  exists to serve is unaffected, and an escaping name raises
  `dataknobs_common.PathEscapeError`.

- **A `wizard_config` path could load a wizard from outside the base declared
  beside it.** `WizardReasoning.from_config` composed
  `config_base_path / wizard_config` and opened the result unchecked — and
  when `wizard_config` was absolute it skipped the composition entirely, so a
  declared base could be bypassed outright. Both operands come out of the
  bot's typed config, the same provenance as the subflow name one call below,
  which was already bounded.

  `config_base_path` now means the tree: `wizard_config` is bounded to it, and
  the same root threads down through every subflow name at any depth. A wizard
  in a subdirectory of the base may name a subflow above itself, because the
  boundary is the tree rather than the last directory a load landed in.
  Declaring no `config_base_path` declares no tree and bounds nothing, which
  is the migration for a deployment that genuinely wants an absolute
  `wizard_config`.

  **Breaking**: an absolute `wizard_config` outside a declared
  `config_base_path` now raises. A nested subflow reaching a sibling *inside*
  the tree, which the previous release refused, now loads.

  `WizardConfigLoader.load` and `load_from_dict` take a `config_root` argument
  for a wizard whose subflows live in a directory beside its own — widening
  the boundary keeps it a boundary, where switching the check off would not.

- **A config name or draft id could write or delete outside the draft
  manager's output directory.** `ConfigDraftManager` composed three paths
  from identifiers it did not check — the final name in `finalize`, the
  alias name in `update_draft`, and `draft_id` — reaching a YAML write and
  two `unlink`s. The name is not only caller-supplied: `finalize()` without
  an explicit `final_name` reads it back out of the draft file's own
  metadata, and `SaveConfigTool` feeds the manager from LLM tool arguments
  and wizard data.

  Containment now lives at the manager, where the paths are composed, via
  the new public `ConfigDraftManager.config_path(name)`. Two consequences
  beyond the direct fix:

  - `SaveConfigTool._persist_config` used to compose its own path and
    re-check it with `resolve().is_relative_to`. That check guarded its own
    `open()` but ran *after* the `finalize()` above it had already written
    through the unguarded manager — so an escaping name was refused *and*
    the file appeared outside the directory. It now resolves through the
    manager before finalizing, and fails closed with nothing written.
  - the tool's entry-point `config_name` check remains, and still returns a
    structured `{"success": False, "error": ...}` the model can correct
    rather than raising. It is now documented as the naming policy it is —
    deliberately stricter than containment — rather than as the boundary.

### Added

- **Every entry point from a portable config to a bot now takes
  `strict_resources`.** A `$resource` reference naming a resource the
  environment does not define degrades to the reference's inline defaults
  with a warning, which is usually right in development and usually wrong in
  production: an empty config handed to a factory rarely fails, it produces
  the factory's default, and a degraded `conversation_storage` binding is an
  in-memory database that holds state perfectly until the process restarts.

  `dataknobs-config` made that policy declarable at four levels. Two of them
  live in code, and neither was reachable here — `DynaBot.from_environment_aware_config`
  did not accept the argument, and a caller handing in a plain dict cannot
  reach the levels that live on `EnvironmentAwareConfig` either, because that
  object is built inside the entry point. So the only levels available were
  the reference's own `$required` and the environment file's setting, and a
  deployment that could edit neither had no switch at all.

  It is now on `DynaBot.from_environment_aware_config` (per call),
  `BotRegistry`, `InMemoryBotRegistry` and `BotManager` (per instance), and
  on the `create_memory_registry` factory, which forwards it.
  Registry-wide rather than per-call on the registries because both cache: an
  argument passed to one `get_bot` would silently decide what every later
  caller received. `None` is the default everywhere and defers exactly as
  before, so nothing changes for a caller who does not pass it.

- **`ConfigCachingManager` takes it too, defaulting to `True`.** That
  manager resolves strictly, which is the behaviour it has always had and is
  preserved — but the posture was a literal, so the environment's own
  `strict_resources` setting was the one level it never consulted. Passing
  `None` hands that decision back to the operator; passing `False` degrades.
  The default is `True` rather than `None` because defaulting to `None` would
  turn every existing deployment lenient without anyone asking for it.

  `ResourceNotFoundError` subclasses `KeyError`, so code wrapping bot
  creation in `except KeyError` for unrelated reasons will swallow a strict
  failure.

### Changed

- **`InMemoryBotRegistry` and `create_memory_registry` forward keywords to
  `BotRegistry` instead of re-declaring its parameter list.** All three held
  a copy of the same seven parameters, so widening the base reached the other
  two only by hand. **Breaking** for a caller that passes them positionally:
  both now take keyword arguments only. That is deliberate rather than
  incidental — `BotRegistry` takes `backend` first while the subclass did
  not, so inheriting the signature would have made a positional first
  argument land on `backend` and mean something else entirely, silently.
  Every documented example already uses keywords.

  `InMemoryBotRegistry` still refuses a `backend`, now with an error that
  says why rather than an unexpected-keyword message; the base already
  defaults to `InMemoryBackend`, so that refusal was the only thing the
  override contributed.

- **`ConfigCachingManager` resolves `$resource` references through
  `dataknobs-config` instead of walking the config itself.** It carried a
  third reader of the reference format, and a format with three readers has
  three definitions. This one recognised `$resource` and `type` and nothing
  else, so it discarded every inline default a reference declared, ignored
  `$required` and `$requires`, passed a misspelled marker on as ordinary
  config data, and left a reference nested inside a resolved resource as a
  literal `{"$resource": ...}` dict for whatever read the config next.

  It also required both `$resource` *and* `type` before it recognised a block
  at all, so a reference relying on the documented `type: default` fallback
  was passed through as a literal dict. Both are now read as the format
  defines them.

  It also carried a fallback for a resource the environment does not define —
  log a warning, return the reference unchanged — that had never run, because
  the lookup it guarded raises rather than returning `None`. Raising is
  therefore the behaviour this has always had and it is preserved; what
  changes is that a reference can now ask for the other one, with
  `$required: false` and inline defaults to degrade to. Inline defaults are
  merged, nested references are followed, and a malformed marker raises
  `ConfigError` rather than reaching a consumer as data.

  A manager with no `environment` still returns the config untouched.

- **`ConfigValidator` reports a misspelled marker in a `$resource` section.**
  Schema validation skipped every `$`-prefixed key, since a component section
  may be a reference whose keys are markers rather than schema fields. But a
  reference's marker vocabulary is closed, so the blanket skip let `$requred:
  true` — which reads as *not required* — past the one check that runs before
  resolution, leaving it to fail in whichever deployment lacked the resource.
  The check now consults `RESOURCE_MARKER_KEYS` rather than the `$` prefix, so
  it cannot drift from the resolver that enforces the set. A `$`-prefixed key
  on a section that is *not* a reference is still left alone.

- **An S3 content key is now normalised, so a non-canonically-spelled object
  written by an earlier release is unreachable.** `_s3_key` returns the
  contained, normalised path, so `sub/../guide.md` composes
  `{prefix}acme/content/guide.md` where it previously composed the literal
  key `{prefix}acme/content/sub/../guide.md`. Two spellings of one intended
  file were two distinct objects, which is the defect — but an existing
  bucket holding such an object needs it re-keyed, and `list_files` still
  reports the old path out of `_metadata.json` while a read of it now
  raises. The same applies to a `path` refused outright (`../x`, `/abs/x`):
  an ingest over such a knowledge base fails rather than degrading. Affects
  only buckets written with non-canonical or escaping paths.

- **An empty `domain_id` is refused on every method, not only
  `key_pattern`.** `""` reached `get_file`, `create_kb` and the rest and was
  composed as a key; `InMemoryKnowledgeBackend.get_file("", "x")` returned
  `None` where it now raises. `None` remains the all-domains spelling
  wherever one is accepted — an empty string now means a caller passed an
  unset variable, which is what it always was.

- **A non-canonical `PrefixedTenantContext` pattern now fails every S3 state
  call** rather than only the file backend's. The pattern is refused, not
  rewritten, for the same reason it always was on the file backend:
  normalising it would move every state document that tenant has written.

### Removed

- **Three review tools that could not be called successfully.**
  `ReviewArtifactTool`, `RunAllReviewsTool` and `GetReviewResultsTool` were
  exported from `dataknobs_bots.review` and broken against the
  `ArtifactRegistry` they address: `registry.get()` had become `async` and was
  still called without `await`, while `add_review()` and `get_definition()` do
  not exist on it at all. The un-awaited `get` was the quietest of the three —
  a coroutine is truthy, so the `if not artifact` guard passed and a coroutine
  travelled onward in place of an `Artifact`.

  Nothing reported it. The module's entire test suite was skipped under a note
  saying the tools still used the old synchronous registry API — accurate, and
  never acted on. A skipped suite reports green, so three broken public classes
  sat behind a passing build.

  No working caller can exist, because every entry point raises or misbehaves
  on first use. This therefore removes a latent `AttributeError` rather than a
  capability, and converts it into an `ImportError` at the point where it is
  actionable. The capability itself was superseded by
  `ArtifactRegistry.submit_for_review()` and `get_evaluations()` when the
  rubric system replaced persona-based reviews.

  The rest of the package is unaffected: `ReviewExecutor`,
  `ReviewProtocolDefinition` and the personas remain exported and remain
  reachable through a wizard's `review_protocols` configuration.

### Fixed

- **Replacing a knowledge base's vector store no longer orphans the chunks
  it already wrote.** `RAGKnowledgeBase.vector_store` was a plain public
  attribute, and a knowledge base without its own `domain_id` takes the
  binding from the store — a value that folds into every chunk id. So
  assigning a differently scoped store silently repointed the id namespace:
  `count()` stopped seeing the existing corpus, the skip-if-populated gate
  re-ingested over rows it could no longer see, and `clear()` could not
  reach them. Nothing raised. The attribute is now a property whose setter
  refuses a swap that would move the effective binding, because once the
  ids are on disk there is no correct continuation and a warning would
  leave the corpus split in two. Construction is not a swap, and neither is
  a rebind on a knowledge base whose own `domain_id` pins the binding.

- **`set_provider()` no longer inverts the close-ownership gate.**
  `RAGKnowledgeBase`, `VectorMemory` and `SummaryMemory` replaced their
  provider without clearing the `_owns_*` flag that decides what `close()`
  tears down. The result was exactly backwards: the provider the caller
  injected got closed, while the config-built one it replaced was never
  closed at all — a leak and a use-after-close from one call. An injected
  provider is now caller-owned at every one of these sites, matching
  `QueryTransformer.set_provider`, and the contract is stated on
  `BaseMemory.set_provider` so an override cannot miss it. Rebinding
  `vector_store` hands ownership back the same way, and warns that the
  outgoing store — which a synchronous setter cannot await — is the
  caller's to close.

- **Two knowledge bases over one vector store no longer overwrite each
  other's chunks.** `domain_id` folds into the chunk-id prefix, but nothing
  supplied the value unless a `KnowledgeIngestionManager` threaded it per
  call — so a knowledge base built from config derived every id from the
  source filename alone, and two domains that each held an `overview.md`
  both produced `overview_0`. The second ingest took the first's row, or,
  against a vector store that refuses a write capturing another domain's
  row, failed every file while reporting no error and zero files.

  `RAGKnowledgeBase` now carries a `domain_id` binding beside `tenant_id`,
  and resolves it from the **bound vector store** when the config does not
  set one. The store already holds that value to scope its own reads and to
  tag rows on write, so deriving it means a consumer who scoped the store
  gets namespaced chunk ids without configuring the same value twice, and
  the chunk-id namespace cannot disagree with the tag on the row. An
  explicit `domain_id` still wins — that is the shape for a deliberately
  unscoped store — and a binding that contradicts a scoped store is
  reported at WARNING, because such chunks are written and can never be
  read back.

  A store-derived binding shapes chunk ids and the metadata stamp only.
  It is deliberately **not** composed into read or write filters: the
  store already confines every read, count, clear and update to that
  domain, by its own means and identically on every backend. Naming the
  key in the filter as well would move the knowledge base onto the one
  surface the store layer documents as *not* uniform — `pgvector` keeps
  the domain in a column and stores caller metadata verbatim, making an
  explicit `domain_id` a containment probe against a key the column
  consumed — and would hide every chunk written before this release
  began stamping `domain_id` into chunk metadata. On that backend a
  knowledge base over a domain-scoped store would have read zero rows,
  counted zero, re-ingested over a corpus it could no longer see, and
  been unable to `clear()` the result.

- **A bound `tenant_id` now scopes `count()`, `clear()` and
  `update_metadata_where()`.** It scoped `query` and `hybrid_query` and
  nothing else, so on a shared store `clear()` on a knowledge base bound to
  one tenant removed every other tenant's rows — data destruction from a
  call whose documented warning was only about passing no filter. The two
  filter-driven mutations take **bound-wins** precedence, so a filter can
  narrow within the scope but not widen past it — naming a bound key with a
  value outside the binding is refused (the request resolves to the vector
  store's unsatisfiable empty-list value and is logged at WARNING) rather
  than redirected to the binding's own value, which would widen a call that
  should match no rows into one that matches every row the knowledge base
  owns. Reads keep explicit-filter-wins, so admin tooling can still read
  across scopes. An
  **unbound** knowledge base is unchanged and is the supported way to act
  across scopes deliberately — `clear()` on one still means every row.

  `count()` being unscoped had a second consequence: it is the count
  `KnowledgeIngestionService.check_needs_ingestion` reads, so a second
  tenant over a store the first had populated was told it was already
  populated and skipped forever, never receiving any chunks of its own.

- **`AutoIngestionMixin` now forwards the whole knowledge-base config to the
  ingest knowledge base.** It hand-copied six keys, while the bot's own
  knowledge base is built from the entire section — so the two disagreed
  about everything the list did not name. `tenant_id` was one: the ingest
  wrote untagged chunks that the bot's tenant-scoped reads could never
  match, a total retrieval blackout reported as a successful ingest. A
  nested `embedding` section was another, silently replaced by the
  hard-coded Ollama defaults, which lands ingest and query in different
  vector spaces. Those defaults now apply only when the section names none
  of `embedding`, `embedding_provider` or `embedding_model`, and they fill
  as a pair: the bot's own knowledge base applies no defaults of its own, so
  filling one key while leaving the other to resolve from an absent value is
  the same divergence in miniature. Three keys are still excluded and each says why in the
  code; `documents_path` in particular must not reach the knowledge base,
  because construction would ingest it ahead of the skip-if-populated check
  and ignore `force`. The registration's own `domain_id` becomes the
  knowledge base's binding when the config does not set one, so an adopter
  sharing a store across bots is correct with no config change.

- **An empty-string binding is a binding at every surface.** The chunk-id
  fold tested truthiness while identity stamping and filter composition
  tested `is not None`, so `domain_id: ""` got scoped reads and a scoped
  write tag with an *unnamespaced* chunk id — the collision the fold
  exists to prevent, at the one value where the two spellings disagree.
  This is the split `VectorStoreBase._is_scoped` settled for the store
  layer after a truthiness test made an empty-string domain isolate on
  three backends and run unscoped on a fourth. Only absent/`None` is
  unbound now. A knowledge base carrying none of the fold keys still
  produces the historical `stem_index` id, byte for byte.

- **Every distinct identity conflict is reported, not just the first.**
  The once-per-instance warning guard was keyed by the metadata key
  alone, so a knowledge base handed several different contradicting
  values reported one and silently re-tagged the rest. The offending
  value is now part of the key. The per-chunk flood the guard was
  written for repeats one value and still collapses to a single line.

- **A knowledge-base config that overrides the registration's domain says
  so.** `kb_config["domain_id"]` outranking the registration is right for
  a section written for one bot, and that precedence is unchanged — but
  the same section is routinely reused as a template across every
  registration, where it quietly points every domain at one namespace and
  their chunks stop separating. Reported at WARNING when the two differ.

- **A `KnowledgeIngestionManager` refuses a destination bound to another
  domain.** The manager's per-call `domain_id` is authoritative — that is
  what lets one manager hold many domains in one destination — but a
  destination `RAGKnowledgeBase` carrying a binding of its own
  contradicted it on every surface at once: identity is sacred at the
  write boundary, so the destination stamped *its* domain over the call's
  and the chunks landed in a scope nobody asked for, while its
  filter-driven mutations were scoped to that binding, so the swap's
  `clear`, tombstone and rollback stopped naming the rows they exist to
  replace. The ingest reported success having written the wrong tag and
  cleaned up nothing. Neither value is derivable from the other, so the
  pairing is a `ConfigurationError`, raised at the first per-domain call
  that reveals it — `ingest`, `ingest_if_changed`, `ingest_changes` and
  `reconcile`. A destination bound to the one domain it is asked for
  agrees with the call and is unaffected, as is the unbound destination
  the manager exists for.

- **`embedding_base_url` reaches the embedder on the legacy flat config
  shape.** The mixin read the key and forwarded it under a name no config
  field carries, so it was discarded in silence and the endpoint it named
  was never used. It is now a legacy alias for a top-level `api_base`,
  and `api_base` wins when both are present. Scope worth stating: the
  top-level passthroughs are consulted only on the legacy `embedding_`
  prefix path. A configured nested `embedding` section supplies the
  provider's endpoint, key and dimensions from inside itself, and neither
  spelling of the top-level key is read at all — prefer the nested form.

  **Migration — chunk ids change for a knowledge base over a domain-scoped
  store**, from `overview_0` to `bot-a\x1foverview\x1f0`. A knowledge base
  with no binding over an unscoped store is byte-identical to before.

  A consumer at one domain per store was already correct and will write
  new-id rows alongside the old ones on a deliberate `force=True`
  re-ingest; `await kb.clear()` — now correctly scoped to that domain —
  followed by a forced re-ingest re-keys them.

  **A consumer sharing one unscoped store across domains has a one-time
  cleanup**, and it is not automatic. Those chunks were written before
  anything stamped `domain_id` into chunk metadata, so a newly-bound
  knowledge base counts none of them, `check_needs_ingestion` reports it
  as never ingested, and the ingest stores the corpus a second time — the
  ingest path appends, and no implicit clear was added to it. The old copy
  is then invisible to every scoped read, and a *bound* `clear()` will not
  remove it either, because it composes a tag those rows do not carry.
  Clear them with `await kb.clear()` on an **unbound** knowledge base over
  the same store before letting each domain re-ingest. Nothing adopts them
  automatically and nothing should: untagged rows on a shared store belong
  to no one domain — several wrote them and collided on the same ids,
  which is the defect being repaired — so assigning them to whichever
  binding looked first would invent an answer.

- **`VectorMemoryConfig.backend` now defaults to `None`, not `"memory"`.**
  Same laundering as the four `.get(key, default)` sites above, in the
  typed-dataclass spelling: the default was written into the dict handed to
  `VectorStoreFactory`, so a config naming nothing arrived as a choice and
  the factory never reported having guessed. An unpersisted vector store
  loses every embedding on restart, which is exactly the consequence that
  report names. Code reading `config.backend` must handle `None`.

- **A bank config spelling the default backend as an alias took the wrong
  branch.** `mem` resolves to the same class as `memory`, but the four
  sites compared against the literal, so `backend: mem` went through the
  factory and came back in *external* storage mode where `backend: memory`
  came back inline — a different storage mode chosen by spelling. All four
  now resolve aliases through the registry they would have built with.


- **Three call sites turned an unchosen database backend into a chosen one.**
  `ArtifactBankCatalog.from_config`, `DataKnobsRegistryAdapter` and the
  `database` grounded-source factory each read the `backend` key with their
  own `"memory"` default and wrote the result into the config they passed
  down, so the factory saw an explicit choice and logged INFO. The absence
  was consumed one frame above the only code positioned to report it — and a
  config arrives empty most often because a `$resource` reference named a
  resource the environment does not define, which is exactly what
  `dataknobs-data`'s WARNING tells the reader to check. The key is now
  forwarded only when the config names one; the object built is unchanged.

- **`ArtifactBankCatalog.from_config` overwrote a backend named inside
  `backend_config`.** The outer default was written into the inner dict
  unconditionally, so `{"backend_config": {"backend": "postgres"}}` silently
  got `memory`.

- **Four more sites spelled the default backend a second time.** The wizard's
  bank creation and restore, its artifact registry, and
  `ArtifactBank.from_dict` each asked "does this config want something other
  than the in-process store?" as `cfg.get("backend", "memory") != "memory"` —
  the constant twice per site, in three phrasings, none of which normalised
  the value. `backend: MEMORY` therefore read as a non-default choice and was
  sent to a factory, and `backend: null` read as one too, reaching the
  factory as `None`. They now share `is_default_backend()`, which reads the
  key the way the factory does and rejects a present-but-unusable value where
  it is written rather than one layer down.

  These deliberately do *not* report an absent key, unlike the three above. A
  bank config naming no backend is asking for conversation-scoped storage —
  the documented default and the recommended answer — and nothing on that
  branch reaches a factory, so there is no provenance to lose. The
  distinction is what separates a laundered default from an ordinary one, and
  it is now pinned by tests rather than left to a reader's inference.

- **A wizard skipped extraction after an auto-advance under `advance()` and
  never under a conversation.** `skip_extraction` is set while the landing
  stage's response is generated and read at the start of the *next* turn, so
  it crosses the boundary only by being persisted. `advance()` hands the
  `WizardState` back for the caller to serialize with `to_dict()`, which
  carries the flag. The conversational path persists through
  `_save_wizard_state`, which built its dict from a second field list — and
  that list omitted the flag. One field, two serializers, two behaviours.

  Nothing reported it, because the restore supplies a default for every key
  it reads: a dropped field arrives as `False` rather than as an error. So
  the user's first message to an auto-advanced stage was extracted in every
  case the flag existed to suppress, and both clears guarding it — the one in
  `process_input` and the one after re-extraction — were unreachable in that
  path. The round-trip test that existed exercised `to_dict`/`from_dict`,
  which the conversational path does not call.

  The second list is gone: persistence now derives its fields from
  `to_dict()` and re-encodes only the two entries needing JSON sanitization.
  A test compares the persisted key set against `to_dict()`, so a field added
  to `WizardState` and forgotten here fails rather than silently reverting to
  its default.

- **An inadmissible `domain_id` was reported as a missing domain.**
  `KnowledgeIngestionManager.ingest_if_changed` wraps its change-detection
  call in `except ValueError` to turn "this domain does not exist" into a
  benign `None`. The identifier guards raise `PathEscapeError`, which *is* a
  `ValueError` — deliberately, so one `except` reaches every refusal — so
  that clause swallowed them: the caller asked for an ingest, the name was
  refused, and the manager logged `Domain not found` and returned `None`.
  No ingest happened, nothing raised, and the one diagnostic naming the real
  problem was replaced by one naming a different problem. A refusal now
  propagates; a genuinely missing domain still returns `None`.

- **The two persistent backends disagreed about a snapshot `version`.** Both
  looked guarded and asked different questions: S3 applied the segment rule
  and the file backend applied containment. `a/b` composes
  `_snapshots/a/b.json`, which never leaves the snapshot directory — so it
  was accepted on the file backend and refused on S3, through the same
  public `list_changes_since`. Containment is the wrong question for a name
  with one slot; both now use the shared segment rule. The test asserting
  the backends agreed had picked the one input where the two rules happen to
  give the same answer.

- **`InMemoryKnowledgeBackend.key_pattern` accepted names the others
  refused.** It returns `""` — no filter is meaningful for in-process
  storage — and skipped validating its arguments on the way. Producing no
  pattern is a property of the store; accepting a name the other two refuse
  is not, and this is the backend consumers develop against. It now
  validates, then ignores.

- **`key_pattern()` named the wrong document in a tenanted deployment, and
  the watch built from it looked healthy.** The method took no
  `TenantContext` on either production backend, while the key-derivation
  helpers it exists to mirror all do — `_metadata_path(domain, ctx)` and
  `_snapshots_path(domain, ctx)` on the file backend, `_metadata_key` and
  `_snapshot_key` on S3. So a `METADATA` or `SNAPSHOT` pattern was built
  without `ctx.state_key_prefix()`.

  The consequence is worse than an empty result. A tenanted deployment has
  *two* metadata documents per domain: the domain-keyed one `create_kb()`
  writes, and the per-tenant state document every `ctx`-scoped write lands
  in. The pattern matched the first. A consumer's inotify watch or
  EventBridge rule therefore installed cleanly, fired once at KB creation,
  and then never again — never seeing a single ingestion-status transition
  for any tenant, with nothing anywhere raising an error.

  `key_pattern` now takes a keyword-only `ctx`, matching `get_info` and
  `set_ingestion_status`. `ctx` and `domain_id` scope independent axes: a
  context picks the tenant, `domain_id=None` wildcards the domain within
  it. `ctx=None` reproduces the previous pattern byte for byte, so a
  single-tenant deployment sees no change. `CONTENT` is deliberately
  unmoved — content is keyed by `domain_id` alone, because tenants share a
  corpus and are isolated on ingest state — and passing a `ctx` with
  `kind=CONTENT` is accepted and ignored so a caller building watches for
  all three kinds needs no special case.

  There is no all-tenants spelling, and cannot be a derived one: the prefix
  comes from `TenantContext.state_key_prefix()`, which for
  `PrefixedTenantContext` is a consumer-supplied format string, so no
  wildcard form of an arbitrary convention exists to compute. Watch the
  base prefix and sort with `classify_key`, or install one pattern per
  tenant.

  The rule "which kinds are tenant-scoped" now lives in one place,
  `KnowledgeResourceBackendMixin._pattern_state_prefix`, rather than being
  re-derived by each backend — that re-derivation is what let the patterns
  drift from the helpers in the first place. The conformance suite's
  layout-drift pin gained a tenancy axis; it had been asserting
  pattern-versus-helper agreement while calling every helper with `ctx`
  defaulted to `None`, the one case where the two agreed by construction.

- **Every path-containment refusal is now one catchable type.** The guards
  in `FileKnowledgeBackend`, `ConfigDraftManager` and `WizardConfigLoader`
  raised a bare `ValueError`, which a consumer cannot distinguish from any
  other `ValueError` on the same call. They now raise
  `dataknobs_common.PathEscapeError`, a `ValueError` subclass — so existing
  `except ValueError` handling is unaffected and a consumer translating
  "your name addressed outside" into a 400 can finally do so precisely. Two
  places that wrapped the refusal in a broader error, `WizardConfigLoader`'s
  subflow loop and the FSM file resource's `acquire`, now let it through
  rather than restating it as something else.

- **`SaveConfigTool` raised out of a tool call for an escaping `_draft_id`.**
  Its entry-point naming policy covers `config_name`, but `_draft_id` comes
  from wizard data and reaches the draft manager unchecked — so the
  manager's guard was what caught it, and it raised where every other
  refusal in that tool returns `{"success": False, "error": ...}`. It is now
  translated into that same structured error, which is what the model can
  act on. The redundant second `mkdir` on the output directory is gone.

- **A config name addressing a subdirectory needed the directory to exist.**
  `ConfigDraftManager` created the parent in `finalize` and
  `_write_named_file` but not in `_write_draft`, so a nested `draft_id`
  would have raised `FileNotFoundError` where a nested config name of the
  same shape worked. That was latent — `create_draft` generates a flat id
  and `update_draft` requires the file to exist, so no public call reached
  it — but the `mkdir` now lives in `_write_yaml`, the one place all three
  writes funnel through, rather than in two of its three callers.

- **The KB tools rejected every resource path when the knowledge directory
  was `.`.** Their containment guard compared normalized path strings, and
  `os.path.normpath(".")` is `"."` while `normpath("./kb/x.md")` is
  `"kb/x.md"` — so with a relative knowledge directory of `.` nothing was
  ever a prefix of anything, and `add_kb_resource` / `ingest_knowledge_base`
  answered *"resolves outside the knowledge directory"* for paths that were
  plainly inside it. The guard is now `dataknobs_common.paths.safe_join`,
  which compares path components and treats a `.` base as the empty prefix
  it is. Behaviour is otherwise unchanged: a `..` segment or an absolute
  component is still refused, and a resource path may still name a
  subdirectory.

## v0.10.0 - 2026-08-11

### Changed

- **A config key naming something that cannot be imported is now fatal, where
  four of them used to be ignored.** This is the consumer-visible half of
  consolidating nine dotted-path resolvers onto
  `dataknobs_common.imports`. Each of these was a WARNING and a bot that
  started cleanly while quietly doing less than its configuration said:

  - an unusable **derivation rule** (`derivations[].custom_class`, and the
    eleven other authoring faults `parse_derivation_rules` rejected alongside
    it — a missing `source`/`target`, an unknown `transform`, a malformed
    regex, a parameterized transform missing its parameter);
  - a **wizard hook** path (`on_enter`, `on_exit`, `on_error`);
  - a **turn-lifecycle hook** entry (`on_turn_start`, `on_turn_end`) —
    including an entry with the `function` key forgotten entirely, which
    previously registered nothing and did not even log;
  - a **task-injection hook** path, and a hook naming an unknown event;
  - a `context_transform` of the wrong *type* (a typo'd path was already
    fatal, so this function disagreed with itself).

  **All faults in one block are reported together**, so an author with three
  bad rules learns about three rather than fixing one and re-running to
  discover the next. Nothing is registered from a block containing a fault:
  partial loading is the silent degradation with an exception attached.

  A deployment running on a stale path will now fail at bot construction where
  it previously started. That is the intended change, and the reason it is
  called out here rather than buried in the consolidation note below.

  **Before upgrading**, find out whether you are affected without waiting for
  the failure: every one of these logged a WARNING first, so the evidence is
  already in your logs. Grep a representative run for the messages that are
  becoming errors —

  ```
  grep -E "Failed to load hook function\
  |Failed to resolve lifecycle hook callback\
  |Lifecycle hook callback path must be\
  |Ignoring unknown event\
  |context_transform must be a callable\
  |^.*(Unknown derivation|Derivation rule|Derivation transform)" app.log
  ```

  — and fix any path or key they name. A clean grep means no config key you
  exercise is on a stale path. Note the last alternative is deliberately
  broad: it also matches derivation warnings raised at *execution* time
  (a transform that failed on a particular value), which are unchanged. Those
  are worth reading anyway, but they are not what will start failing.

- **Both `module:name` and `module.name` are accepted at every config key that
  takes a dotted path.** Three keys previously accepted only `:`, four only
  `.`, and two either — so the same value was valid or invalid depending on
  which key it was written under. Only paths that were previously *rejected*
  start resolving; no working configuration changes meaning.

- **A wrong-shape `merge_filter` or `custom_class` no longer runs the target's
  `__init__` before rejecting it.** Both used to instantiate and then check the
  protocol, so a mistyped path executed an unrelated class's constructor —
  arbitrary code, with whatever side effects it has. `resolve_class` returns
  the class and leaves construction to the caller, which makes
  validate-before-instantiate the only expressible order. Tool and middleware
  specs already behaved this way, by a policy held in a twelve-line comment;
  that comment is deleted because the resolver now enforces what it described.

- **Error types at the resolution sites are unified.** `ValueError`,
  `ImportError`, `AttributeError` and `KeyError` are all `DottedPathError` — a
  `ConfigurationError` subclass carrying a machine-readable `reason`. A caller
  catching `ConfigurationError` is unaffected; one catching a specific stdlib
  type is not. The rubric `FunctionRegistry.get` case is the sharpest: a bad
  `function_ref` raised `KeyError`, which read to a caller as a missing
  dictionary key rather than a bad config value.

  The `context` dict changes shape with the type. `load_merge_filter` used to
  attach `context={"merge_filter": <path>}`; it now carries the family's
  `ref` and `reason`, so code reading `exc.context["merge_filter"]` gets a
  `KeyError`. Read `exc.ref` for the path — it is on the exception as an
  attribute as well as in `context`.

- **Error text at those sites is bounded** — the message names the reference
  and the failure type, and the caught exception's text moves to `__cause__`
  and the logs. Three sites interpolated it directly. Importing a module
  executes it, so that text is arbitrary, and `ConfigurationError` is rendered
  at the HTTP boundary.

- **`dataknobs_bots.tools.resolve` is now a re-export** of
  `dataknobs_common.imports`. Existing deep imports keep working;
  `resolve_optional_callable` raises `DottedPathError` rather than
  `ValueError`.

- **A bad `chunking.chunker` path now escapes
  `create_knowledge_base_from_config` as a `DottedPathError` rather than an
  `OperationError`.** No bots code changed here — this follows from
  `dataknobs-xization` adopting the shared resolver, and it is recorded in
  this changelog because the type a bots API raises is a bots contract.

  The knowledge-base registry unwraps a backend's exception out of the
  registry's `OperationError` wrapper when it is a `ValueError` or a
  `DataknobsError`, and re-raises the wrapper otherwise. `create_chunker`
  used to fail with `ImportError`, which is neither, so the wrapper escaped;
  it now fails with `DottedPathError`, which *is* a `DataknobsError`, so the
  cause escapes in its place. A caller writing `except OperationError:`
  around knowledge-base construction to catch a mistyped chunker path stops
  catching it; `except ConfigurationError:` catches it, and matches how every
  other dotted-path fault in a bot config already reports.

  Unchanged at the HTTP boundary — both types render as `500`.

- **`DynaBotConfigBuilder.merge_overrides` documents list-replace, and no
  longer carries its own merge.** The behavior is unchanged — the private
  `_deep_merge` it used was byte-equivalent to `dataknobs_config.deep_merge`,
  which it now calls — but the contract was only ever implied. Lists in an
  override **replace** the base's rather than extending it; nested dicts
  merge key by key. A fourth copy of the same function elsewhere in the
  workspace had drifted to extending lists, which is what prompted the
  consolidation; `tests/test_deep_merge_agreement.py` now fails if any entry
  point stops agreeing.

  If you imported the module-level `_deep_merge` from
  `dataknobs_bots.config.builder` — underscore-private, but importable — use
  `from dataknobs_config import deep_merge` instead.

- **FSM functions-layer errors now get a status matching the failure.** The
  exceptions in `dataknobs_fsm.functions.base` were rooted at a plain
  `Exception` rather than at `DataknobsError`, so they never reached
  `dataknobs_error_handler` at all — they fell through to the `Exception`
  catch-all, which is precisely where that handler's own docs say DataKnobs
  errors no longer arrive. Every one of them came back as an indistinguishable
  `500 / InternalServerError`.

  With those types rebased onto their common counterparts (see the
  `dataknobs-fsm` changelog), the table resolves them: a resource-acquisition
  failure from any of the six resource backends is now `503`, masked, and a
  validation failure from the validator library is `422` with the failed
  checks in `detail`. No new rows — they resolve through `ResourceError`,
  `ValidationError` and `OperationError` by MRO, as any subclass does.

### Added

- **`register_exception_handlers` returns the effective policy table**, and
  rejects a row it could never consult. Middleware cannot raise to reach a
  handler — Starlette consults the per-type handlers only below the middleware
  stack — so the documented pattern is to *call* one; calling
  `dataknobs_error_handler` without a `table=` silently applies
  `DEFAULT_ERROR_POLICY` rather than the table registered on the app, and the
  two differ exactly when someone bothered to pass `error_policy=`. Naming
  cannot prevent an omitted argument, so registration now hands back the table
  to pass.

  A key that is not a `DataknobsError`, or that is an `APIError` subclass —
  which takes its disclosure from `client_safe` and never reaches the table —
  now raises at registration instead of being accepted and ignored. Neither is
  detectable from the outside: the response is identical whether the row
  applied or not, so a deployment that writes one believing it has set a
  disclosure policy has no way to discover it has not.

- **A `DEFAULT_ERROR_POLICY` row for `InvalidTransitionError`** —
  `409 Conflict`, message and `detail` both disclosed. It is an
  `OperationError`, so it inherited a masked 500: the server blamed for the
  caller's mistake, and the `allowed` targets its `context` carries — the
  remedy — withheld. "Cannot go from `draft` to `shipped`" is the textbook
  409, the request conflicting with the resource's current state.

  A row rather than a rebase. `OperationError` is the right base for a
  library, because an invalid transition is permanent and retry logic keyed on
  that base correctly declines to re-attempt it; rebasing onto
  `ConcurrencyError` would have bought the same status and broken that. The
  two axes disagreeing is what this table is for.

  This is the first row whose type is not in `dataknobs_common.exceptions`,
  which exposed a hole in the guard around the table: both the recorded
  contract and the published status table were keyed off that module's
  `__all__`, so a row for a type from anywhere else shipped an undocumented
  status with the suite green. A row is now required to appear in both.

- **`WizardFSM` lifecycle** — `close()`, `aclose()`, `__enter__`/`__exit__`,
  and `__aenter__`/`__aexit__`, mirroring the wrapped `AdvancedFSM`
  one for one. A synchronously-stepped wizard can now be released:

  ```python
  with WizardConfigLoader().load_from_dict(config) as fsm:
      fsm.step({"name": "Alice"})
  ```

  Both close forms are idempotent, and both leave the FSM **steppable** —
  a later `step()` lazily rebuilds its bridge — so an unconditional
  teardown is safe without tracking whether the FSM was ever stepped. That
  covers the bridge, not registered resources: closing is terminal for the
  resource manager, so an FSM holding resources should not be stepped
  after close. Loader-built wizard FSMs register none, which is what makes
  the unconditional teardown safe in practice. Prefer `aclose()` from
  async code: it does everything `close()` does, additionally awaits
  providers whose cleanup is a coroutine, and keeps the bridge join off
  the event loop.
- **`WizardFSM.register_subflow(..., owns=True)`** — closing a wizard
  cascades to the subflows it owns, error-isolated per child so one
  failing subflow cannot orphan the siblings registered after it. Every
  loader-built subflow is parent-owned, which covers the
  configuration-driven case entirely. Pass `owns=False` to register a
  subflow whose lifecycle belongs to its caller. Re-registering a name
  replaces both the subflow and its ownership, and closes the *owned*
  subflow it displaces — reusing a name is the parent's last chance to
  release it. Re-registering the same object closes nothing, so handing a
  subflow back to its caller with `owns=False` does not destroy it.
- **`dataknobs_bots.behavior_packs`** — the bot-flavored vocabulary for
  `dataknobs_common.packs`. `BehaviorPackSpec` is a `PackSpec` subclass
  naming five optional fields and the rule each composes under:
  `required_strategy` (`UNANIMOUS` — two packs demanding different
  strategies is unsatisfiable rather than resolvable, so it raises),
  `strategy_overrides` (`MERGE`), `middleware` and
  `conversation_middleware` (`CONCAT` — order is behavior, and a repeated
  spec is a deliberate second installation), and `stage_synthesizers`
  (`CONCAT_UNIQUE` — these are names, and registration is idempotent, so a
  duplicate is noise). The two middleware fields hold the **raw spec
  mappings** the bot config already accepts rather than live instances, so
  a pack stays serializable and cannot drift from `DynaBotConfig`; pair
  them with `build_middleware` / `build_conversation_middleware` and the
  additive `platform_middleware=` channel to install a composed pack.
  `verify_stage_synthesizers(names)` closes the gap between a pack
  *declaring* a synthesizer name and some module *registering* it, raising
  `ConfigurationError` listing every missing name plus what is registered —
  without which a typo would surface only as a wizard stage whose primitive
  silently never expands. `BehaviorPackRegistry` is a type alias for
  `PackRegistry[BehaviorPackSpec]` so consumer signatures can name the
  concrete type. DataKnobs ships **zero** packs and no module-level
  registry: pack content and binding are per-deployment policy, and a
  process-global registry would be a multi-tenant hazard.
  `BehaviorPackSpec`, `BehaviorPackRegistry`, and
  `verify_stage_synthesizers` are exported from the top-level
  `dataknobs_bots` namespace. See `docs/behavior-packs.md`.

- **`build_middleware()`, `build_conversation_middleware()`, and
  `resolve_middleware_from_spec()`** in `dataknobs_bots.middleware`
  (also exported from the top-level `dataknobs_bots` namespace) — the
  spec-to-instance resolution `DynaBot` applies to its own configured
  `middleware:` and `conversation_middleware:` blocks, now free-standing so
  anything assembling middleware declaratively can produce instances to
  hand to `from_config(..., platform_middleware=...)` without reaching into
  bot internals. The two wrappers take an iterable of specs and return a
  list of live instances; both delegate to `resolve_middleware_from_spec`,
  so there is exactly one resolution body and the two flavors cannot drift.
  Middleware specs are **trusted configuration**: a spec's `class` is a
  dotted path that is imported and instantiated, so specs must never be
  built from end-user or per-tenant input.
  `optional: true` continues to cover only transient resolution failures
  (missing module / class / bad constructor params) — a class-shape
  mismatch, such as a turn-lifecycle `Middleware` listed under
  `conversation_middleware:`, always raises `ConfigurationError`, and the
  `issubclass` check runs *before* instantiation so a misplaced spec never
  executes its constructor. A skipped optional spec is absent from the
  result rather than a `None` hole, so the returned list is directly usable
  as `platform_middleware`; resolve specs one at a time when you need to
  know which was skipped.

  Two observable consequences of the move, both intended. Resolution now
  logs under `dataknobs_bots.middleware.factory` rather than
  `dataknobs_bots.bot.base`, so anything filtering middleware-resolution
  logs by logger name needs updating. And a configured bot-turn
  `Middleware` whose class defines a falsy `__bool__` / `__len__` is now
  installed rather than silently dropped: the bot-turn path tested
  truthiness where the conversation path tested `is not None`, and the two
  now agree on `is not None`.

- **`dataknobs-bots[postgres]` and `dataknobs-bots[faiss]` extras.** Both
  were already documented in the README, user guide, and package index
  but never existed, so `pip install dataknobs-bots[postgres]` warned and
  resolved to the base package — leaving the PostgreSQL registry backends
  and FAISS vector memory to fail at runtime on a missing driver.
  `postgres` forwards to `dataknobs-data[postgres]` (registry storage via
  `create_registry_backend` / `DataKnobsRegistryAdapter`) and
  `dataknobs-common[postgres]` (the cross-replica ingest lock reached by
  `IngestOrchestrator` -> `create_lock({"backend": "postgres"})`), so the
  driver floors stay owned by those packages. `faiss` forwards to the
  narrow `dataknobs-data[faiss]` rather than `dataknobs-data[vector]`,
  whose roll-up would also pull chromadb — carrying an unfixed
  pre-authentication code-injection advisory (GHSA-f4j7-r4q5-qw2c /
  PYSEC-2026-311, CVSS 9.3) — and pgvector, needed only by the
  Postgres-backed store.

- **`dataknobs-bots[all]` extra**, resolving to
  `dataknobs-bots[faiss,http,postgres,s3,server]` — the roll-up every
  sibling package already provides (`dataknobs-data[all]`,
  `dataknobs-fsm[all]`, `dataknobs-llm[all]`). Also already documented
  and likewise nonexistent, so it too silently resolved to the base
  package. Note it does **not** pull chromadb or pgvector, per the
  narrow `faiss` scoping above.

- **`dataknobs-bots[http]` extra** declaring `aiohttp>=3.14.3`, the
  transport `HTTPRegistryBackend` (`registry.http_backend`) has always
  used. The module continues to lazy-import aiohttp so the package
  imports cleanly without the extra; `initialize()` raises `ImportError`
  with a `pip install 'dataknobs-bots[http]'` hint (previously an
  unqualified `pip install aiohttp`).

- **`bedrock` rate family in `CostTrackingMiddleware.DEFAULT_RATES`**, priced
  against the Anthropic models Bedrock resells. Listed separately from
  `anthropic` rather than aliased, so the two can diverge when Bedrock's
  pricing does and so a `cost_rates={"bedrock": …}` override cannot rewrite
  Anthropic's. Bedrock's fully-qualified model IDs
  (`anthropic.claude-3-5-sonnet-20241022-v2:0`) resolve through the existing
  partial-match fallback. An `echo` entry is priced at zero — a test double
  performs no inference, so zero is its true price.

- **A WARNING when cost tracking cannot price a request.** A rate-table miss
  recorded `$0.00` silently, which is what let the provider-key defect below
  survive for the life of the feature. Misses now log once per
  `(provider, model)` pair, naming whether the family or the model was
  unknown and pointing at `CostTrackingMiddleware(cost_rates=…)`. Providers
  with a genuine zero price (`ollama`, `echo`) do not warn. `huggingface` is
  deliberately left unpriced rather than defaulted to zero: it covers both
  free local inference and the paid Inference API, so a zero entry would
  assert that paid traffic is free.

- **`TurnState.provider_impl`** and a matching `provider_impl` field on the
  `turn_complete` structured log payload, carrying the concrete provider
  class. See the `### Changed` note below — this is the field that preserves
  what `provider` used to contain.

- **`Retry-After` header on API error responses.** `api_error_handler` now
  emits one whenever the exception carries a `retry_after` — in practice a
  429 from `dataknobs_bots.api.RateLimitError`. `detail.retry_after` in the
  JSON body is this project's own shape and nothing outside it knows to look
  there; `Retry-After` is what HTTP clients, proxies, and SDK retry policies
  already act on, and RFC 6585 says a 429 SHOULD carry one. The value is
  RFC 7231 delay-seconds, so the float the rate limiters report is rounded
  **up** (rounding down returns the client while it is still throttled) and a
  negative value clamps to zero. An exception with no hint gets no header
  rather than a default, since a made-up wait would assert something the
  server never computed.

- **`TurnState.pricing`** — the per-model USD rates the provider resolved for
  the turn's model, or `None` when it sources none. Captured while the
  provider object is still in hand, because `TurnState` discards it after
  reading a name: a rate not taken there has to be guessed from a second
  table later, which is how the cost middleware's hand-written duplicate came
  to exist and to drift.
- **A `DataknobsError` handler in `register_exception_handlers`**, with the
  `ErrorPolicy` / `DEFAULT_ERROR_POLICY` / `resolve_error_policy` surface it
  is driven by. Eleven rows map the `dataknobs_common.exceptions` types to a
  status and a disclosure decision, and resolution walks the exception's MRO
  rather than looking up its exact type — so the forty-plus `DataknobsError`
  subclasses the other packages define inherit their nearest listed ancestor's
  row. `RecordNotFoundError` returns 404 without appearing in the table. An
  exact-type table would have covered the twelve listed classes and silently
  500'd the rest, which is the behaviour being replaced.

  A row decides the message and the `context` separately —
  `ErrorPolicy(status, disclose_message, disclose_context)`. A withheld message
  is replaced by a generic one, a withheld `context` by `{}`, and both halves
  are logged whichever way the row falls, so a diagnostic is relocated to the
  server log rather than discarded — the half of the defect that mattered.

  The two are separate because the types disagree about which half is safe, in
  both directions. `NotFoundError`'s message is the caller's own key echoed
  back while its `context` carries `available_keys` — a registry's entire
  keyspace, a "did you mean" for a library caller and an inventory listing for
  an HTTP one. `TimeoutError`'s own docstring example puts a SQL query in
  `context`. Conversely `ValidationError`'s `context` is the caller's own
  fields while its message can be a database driver's. One bit would force each
  of those rows to give up a useful half to withhold an unsafe one. Both are
  now `(status, True, False)`: they keep their message and drop their context,
  with no change at any raise site. `disclose_context` defaults to `False`, so
  a row written without thinking about `context` fails closed.

  The `dataknobs_bots.api` family keeps a single `client_safe` bit, because a
  subclass writes its message and its `detail` in one constructor for one
  audience, and because `to_dict()` is overridable and returns an arbitrary
  dict — partial disclosure could only mean allow-listing keys, which would
  silently drop one an override added.

  `ConfigurationError` is **masked by default**, and that is the one row where
  the setting is a judgement rather than a reading of the type. Most config
  diagnostics are authored — a key name, a sorted list of the valid ones — and
  are exactly what a failing config route should return. But that type is also
  where the funnels wrapping a third-party constructor or module import land,
  and their text is unbounded: a database or cache client raises with its
  connection URL, credentials included. DataKnobs bounds its own funnels (see
  `### Security`) but cannot audit a consumer's, and bots are built lazily on
  the request path.

  The handler covers errors raised at any depth **under a route**, not
  anywhere in the ASGI stack. Starlette builds `ServerErrorMiddleware` → user
  middleware → `ExceptionMiddleware` → router, and only `ExceptionMiddleware`
  consults the per-type handlers; an error raised in an `app.add_middleware`
  layer is above that and still returns a generic 500. That was already true
  of `APIError`, and is Starlette's layering rather than something this
  handler could reach. Middleware wanting a status should return
  `await api_error_handler(request, exc)` instead of raising, or move the work
  into a route dependency.

  `register_exception_handlers` takes an `error_policy=` mapping, merged over
  the defaults, that turns it on in one line for a route that is not public,
  and gives a deployment's own `DataknobsError` subclasses a policy:

  ```python
  register_exception_handlers(
      app, error_policy={ConfigurationError: ErrorPolicy(500, True)}
  )
  ```

  `DEFAULT_ERROR_POLICY` is a read-only mapping. It is process-global and read
  per request, so assigning a row into it would change the disclosure policy of
  every app in the process — including ones registered earlier, which no care
  at registration time would catch. `error_policy=` is the per-app route.

  Two parameters carry a policy table and they compose differently, so they are
  named differently: `register_exception_handlers(error_policy=...)` takes
  *overrides*, merged over the defaults, while `resolve_error_policy(table=...)`
  and `dataknobs_error_handler(table=...)` take the *whole* table, replacing
  them. A table passed to the latter that omits `DataknobsError` has no
  terminal row, and resolution fails closed to a masked 500 rather than
  falling through.

### Changed

- **Handled DataKnobs errors no longer propagate to the ASGI server.**
  Starlette routes the `Exception` catch-all through `ServerErrorMiddleware`,
  which calls the handler *and then re-raises* so the server sees the failure;
  handlers registered for narrower types do not re-raise. A DataKnobs error
  used to be both returned as a 500 and propagated to uvicorn, producing a
  server-level error log and a tick on whatever the deployment counts as an
  unhandled exception. It is now handled cleanly and does not propagate.

  That is the intended semantics — a config typo is not a server fault — but
  it is a monitoring behaviour change, and a deployment alerting on unhandled
  exceptions will see that signal drop. The handlers log every error they
  handle instead, so the information is relocated rather than lost: a 4xx at
  `info`, a 5xx at `error` with the traceback, and a masked error's message
  and `context` on that line since the response does not carry them.

  The level follows the status class, not the disclosure bit. A 404 is a
  routine outcome of serving traffic — logging one at `warning` makes a
  working service look like a failing one and buries the 5xx that need
  attention — and a 504 is a server-side fault even though it is disclosed.
  Both handlers share one helper, so the API family and the policy table
  cannot drift into disagreeing about severity.
- **`dataknobs_bots.api.RateLimitError` is now also an `OperationError` and
  reaches `DataknobsError` by a second route.** This follows from the
  subclassing fix under `### Fixed` below and is the intended semantics — a
  rate limit *is* an operation failure, which is why the common class is
  shaped that way — but it widens what an existing `except` clause catches:

  ```python
  try:
      raise dataknobs_bots.api.RateLimitError(retry_after=30)
  except dataknobs_common.exceptions.OperationError:
      ...   # NEW: previously fell through
  ```

  A consumer with an `except OperationError` block that previously did not
  see API rate-limit errors will start seeing them. Relying on the old
  non-relationship seems unlikely to be deliberate, but unlikely is not
  impossible, which is why this is disclosed here rather than filed only
  under "Fixed".

- **`dataknobs_bots.api.BotCreationError` is now also an `OperationError`**,
  by the same reasoning and with the same consequence for an existing
  `except OperationError` block. Creating a bot is an operation and failing
  to create one is an operation failure; unlike the twinned classes there is
  no same-named counterpart to reach for, so a consumer catching
  `OperationError` had no way to discover that this particular failure was
  excluded from it. `APIError` is now the only class in the module with no
  counterpart closer than `DataknobsError`, which is correct — it is the base
  the common hierarchy is being extended *into*.

- **The `provider` value in turn logs and in `after_message` middleware
  kwargs is now the canonical provider family key rather than the provider
  class name** — `"openai"` where it previously read `"OpenAIProvider"`. A
  dashboard, alert, or middleware keyed on the old string will stop matching.
  The class name has not been dropped: it moved to `TurnState.provider_impl`
  and to the `provider_impl` log field, so both identities remain available
  and each now has a name that says which one it is.

  The same rename reaches every cost-stats surface: the `by_provider` keys
  returned by `get_client_stats()` / `get_all_stats()`, and the buckets in
  `export_stats_json()` / `export_stats_csv()`. A billing pipeline parsing
  that output sees a bucket rename, which is the most likely place a
  consumer is keyed on the old value.

- **Cost is now priced from the provider's own model profile when it has
  one**, falling back to the middleware's table only for families and models
  the catalogs do not cover. Resolution order is: a rate you supplied via
  `cost_rates=`, then the provider's catalog pricing, then the built-in
  table, then `$0.00` with a warning. Recorded costs will change for
  providers whose catalog pricing differs from the built-in table — which is
  most of them, since the table is a hand-maintained duplicate marked
  "Updated Dec 2024" and the catalogs carry a verification date. `o1-mini`,
  for instance, was billed at roughly 2.7x its real rate on an exact match.

- **Persisted conversation metadata records the canonical family key.** The
  `provider` value in the metadata `DynaBot` seeds a `ConversationManager`
  with previously carried `config.provider` verbatim, so a `provider: OpenAI`
  deployment stored `"OpenAI"` while the same turn's cost bucket said
  `"openai"`. This metadata is durable, so the mismatch outlived the process
  that wrote it.

- **`llm.provider` config validation reads the provider registry** instead of
  a transcribed list of five names. The literal had already drifted — it
  rejected `bedrock`, a family DK ships and prices — and it rejected
  `provider: OpenAI`, which the runtime accepts because the registry
  canonicalizes its lookups. Most importantly it could never accept a
  provider registered through `LLMProviderFactory.register_provider`, so the
  documented extension point was silently disabled at the validation layer.
  Comparison is now case-insensitive and the valid-options list is live, so
  `DynaBotConfigSchema.get_valid_options("llm", "provider")` and the
  generated documentation include consumer-registered providers. Genuinely
  closed sets (`memory.type`, `reasoning.strategy`) keep literal enums —
  though case-insensitive comparison now applies to *every* schema enum, not
  only the registry-backed one. Those sets are also resolved through
  registries built with `canonicalize_keys=True`, so `type: Buffer` already
  worked at runtime and the validator no longer disagrees with it.

  A property may declare `enum_registry: <name>` to be checked against a live
  registry instead of a literal `enum`. A name this build does not have leaves
  the field unconstrained and logs a warning, rather than rejecting every
  value for it — a consumer schema may name a registry a newer DK supplies.

### Fixed

- **A handled error's log line dropped what it was raised `from`.** For a 5xx
  the chain came free with the traceback, but a 4xx logs at `info` with no
  traceback and `%s` renders only the outer exception's own message — so
  nothing recorded the cause. That is exactly the case a library raising
  wrapped errors produces: a provider translating a vendor failure it must not
  disclose raises `ValidationError("openai API error (HTTP 400)")` with the
  vendor's response body on `__cause__`. Logging only the outer message made
  every such 422 read identically. Both handlers now append
  `cause=<type>: <message>` when there is one, and nothing when there is not.

- **A malformed `retry_after` turned a 429 into a generic 500.** `math.ceil`
  raises on the non-finite floats and on anything that is not a real number,
  and it ran inside the handler — where the only catcher left is Starlette's
  error middleware, so the status, the message, and the retry hint were all
  replaced by the response these handlers exist to stop returning. Reachable
  rather than theoretical: a provider parses the value from the upstream
  `Retry-After` header with `float()`, which accepts `"inf"` and `"nan"`, and
  the endpoint is consumer-configured. A hint that cannot be converted now
  costs the header and nothing else. (`dataknobs-llm` additionally stopped
  producing one — see its changelog.)

- **An arbitrary object in `context` was rendered with `str()` into a
  disclosed response.** Five rows disclose `context`, and the rule was argued
  from a `StructuredConfig`, whose repr redacts its own secrets — generalising
  from the one cooperative type to every type. The objects a raise site
  actually holds when it fails do the opposite: a SQLAlchemy `Engine` renders
  as `Engine(postgresql://user:pw@host/db)` and a psycopg2 connection quotes
  its DSN, both deliberately, because a repr is a debugging aid written for a
  log. Values are now rendered as their type name unless they are one of the
  types whose text *is* their value — `Path`, `UUID`, the datetime family,
  `Decimal`, `Enum` — which is what keeps the fix from costing the diagnostic.

  A value whose `__str__` raises, and a `Mapping` whose `items()` raises, no
  longer take the response with them either; and the response builder now
  catches whatever the walk still cannot handle, since everything it touches
  is arbitrary code and "no value can make this raise" is not something it can
  promise on its own.

- **A masked 4xx is logged at `warning` rather than `info`.** The level follows
  the status class everywhere else, deliberately — logging a 404 at `warning`
  makes a working service look like a failing one. But a masked 4xx is the one
  combination where the log is the *only* record of a failure the caller was
  told nothing about, and `info` is a level production deployments routinely
  filter. No default row is affected; this is for a consumer `APIError`
  subclass with `client_safe = False`, or a consumer row that masks a 4xx.

- **A `context` value the JSON encoder could not represent turned any error
  response into a generic 500.** The response body is rendered with
  `json.dumps(..., allow_nan=False)`, so a `Path`, an object, or a
  `float("inf")` raised *inside* the handler; Starlette's error middleware
  then caught that and returned exactly the `500 / "An unexpected error
  occurred"` these handlers exist to replace, losing the real status and
  message. `context` is a free `dict[str, Any]` and raise sites fill it with
  whatever the failure was about, so this was not an exotic input — a 404 from
  a config loader carrying the path it looked in was enough.

  Such values are now rendered with `str()`. Not with
  `fastapi.encoders.jsonable_encoder`, which falls back to `vars(obj)` and
  would put an object's whole attribute dict into a response body the raiser
  only meant to carry the object; `str` discloses what the type chose to say.
  The walk is depth-bounded, which also terminates a self-referential
  `context` — another input `json.dumps` rejects outright. It runs in the
  shared response builder, so a consumer's overridden `APIError.to_dict()` is
  covered by the same guarantee.
- **Every DataKnobs error raised from a route returned
  `500 / "An unexpected error occurred"`.** `register_exception_handlers`
  covered `APIError`, `HTTPException`, and `Exception`, and every
  `DataknobsError` that is not an `APIError` — which is all of them, from well
  over a hundred raise sites across DataKnobs' own source — fell to the
  catch-all with its message and `context` discarded. The reported case was a config-validation
  diagnostic: DataKnobs generated `embedding: no variant registered for
  'ollamaa'`, then threw it away one layer later and returned a 500 that named
  nothing. See the new handler under `### Added`.
- **The common `RateLimitError` returned no `detail.retry_after`, while the
  API twin did.** The two classes store the hint in different places — the API
  variant writes it into `context`, which `to_dict()` serializes, and the
  common one keeps it as an attribute only — so one condition produced two
  response bodies depending on which of two same-named classes the raiser
  reached for. Both now report it in `detail` as well as in the `Retry-After`
  header, and a `retry_after` a raiser put in `context` deliberately is left
  alone.
- **`dataknobs_bots.api.RateLimitError` was not a subclass of
  `dataknobs_common.exceptions.RateLimitError`.** It was a *sibling*, so
  `except RateLimitError` written against the common name — the name
  DataKnobs' own `RateLimitMiddleware` raises — silently never fired for the
  API variant. Six of the seven `bots.api` exception classes already
  subclassed their common counterpart; this was the one that did not, which
  is why the gap read as a working pattern right up until it dropped an
  error. It is now `RateLimitError(APIError, CommonRateLimitError)`, so one
  `except` against the common name covers both.

  `retry_after` is now also exposed as an **attribute** (the structured form
  the common hierarchy defines) alongside the existing `detail["retry_after"]`
  serialized field, and its type widened from `int | None` to `float | None`
  to match both the common class and the `RateLimitStatus.reset_after` value
  the middleware supplies. Existing `int` callers are unaffected.

  Handler dispatch is unchanged: `APIError` still precedes the common base in
  the MRO, so `register_exception_handlers` routes the API variant to
  `api_error_handler` exactly as before — same 429, same body. A parametrized
  test now pins every API exception to the common class it should subclass,
  so a future divergence in the family fails at the source rather than in a
  consumer's `except` clause. An API-layer-only class records *why* it has no
  counterpart rather than a bare `None`, and that entry is asserted too — a
  skipped branch would have made the cheapest table entry, the one a new
  class gets when nobody decides, also the one nothing verifies.

- **A `retry_after` of zero was dropped from the API error response body.**
  `RateLimitError` populated `detail["retry_after"]` under a truthiness test
  while assigning the attribute unconditionally, so at zero the two views of
  one value disagreed: the attribute read `0.0` and the serialized field was
  absent entirely. The client learned nothing rather than "retry now". Zero
  is reachable, not hypothetical — `PyrateRateLimiter.get_status` reports
  `reset_after=0.0` unconditionally and `InMemoryRateLimiter` does so
  whenever the window has just drained, and the re-raise recipe in the API
  docs forwards that value straight into this constructor. The test is now
  `is not None`.

- **`general_exception_handler` logged with an f-string**, which interpolates
  before the logging call and discards the exception object carrying the
  traceback that `logger.exception` exists to record. It now uses lazy `%s`
  formatting, and the logger is resolved once at module scope rather than on
  every unhandled request.

- **A synchronously-stepped wizard leaked a daemon thread with no way to
  release it.** `WizardFSM` wraps an `AdvancedFSM`, which allocates a
  process-lifetime event-loop thread on first synchronous `step()`, but the
  wrapper exposed none of the six lifecycle members its wrapped object
  provides — so nothing, including `DynaBot.close()`, could reach it. The
  leak is silent by construction: the threads are daemons, the FSM behaves
  correctly, and nothing raises. A full test run accumulated 32 of them,
  noticed only because it made an *unrelated* package's teardown
  assertions fail depending on test ordering. `WizardReasoning.close()`
  now releases the FSM, and `DynaBot.close()` reaches it end to end.

  The bridge is allocated by the synchronous `step()`, which the bot
  itself never calls — it drives the wizard through `step_async`. The
  reachable leak is therefore in code that steps a `WizardFSM` directly,
  which `step()` being public API makes a supported thing to do; the
  end-to-end close path is what keeps it from recurring once someone
  does.

  Ownership is explicit and defaults to **not owned**. The FSM is a
  *required* constructor parameter, so the common shape is a caller
  handing over an FSM it built and may still be stepping; closing that
  would be a use-after-close at every direct-construction site. Only
  `from_config`, which builds the FSM itself, takes ownership. The two
  errors are not symmetric — the default fails toward the pre-existing
  leak rather than toward tearing down a live FSM.

- **Re-registering a subflow name orphaned the subflow it replaced.**
  `WizardFSM.register_subflow` overwrote the registry entry and updated
  the ownership set but never closed the object it displaced, so an owned
  subflow that had been stepped lost its only route to `close()` the
  moment its name was reused — the same unreachable-daemon-thread defect
  the wrapper's lifecycle exists to prevent, one level down. The displaced
  subflow is now closed when it was owned and is a different object.

- **The subflow registry passed to `WizardFSM(...)` was aliased, not
  copied.** Ownership is recorded once from the mapping's contents at
  construction, so a caller that kept its reference and added an entry
  afterwards produced a subflow the FSM would step but never close —
  present in the registry, absent from the ownership set. The mapping is
  now copied.

- **Cost tracking recorded `$0.00` for every paid provider.** Usage was keyed
  on the provider *class* name (`"OpenAIProvider"`), which matches no entry in
  a rate table keyed by family (`"openai"`), so every lookup missed and every
  request priced at zero. Turns are now keyed on the family, and spend is no
  longer split across two buckets when a config author capitalizes the
  provider name. The lookup key comes from the provider itself rather than
  being reconstructed, so the four consumers of provider identity agree.

- **Per-instance `cost_rates=` overrides permanently rewrote the class-level
  defaults.** `DEFAULT_RATES` was shallow-copied, so each instance shared the
  same nested per-family dicts as the class attribute, and the merge mutated
  them in place. One middleware instance's custom rates therefore changed
  pricing for every instance constructed afterwards in the same process —
  including instances belonging to other tenants. The defaults are now
  deep-copied per instance — and so is the dict the caller supplies, which
  was the other half of the same problem: the merge inserted the caller's
  nested dicts by reference, so a module-level rate constant shared across
  per-tenant instances put every tenant back on one object and let this class
  mutate a constant it was only given to read.

- **The rate table's model lookup billed dated model IDs at the wrong
  model's rate.** Repairing the family key made a code path live that had
  never run in production: with the lookup key a class name, `_calculate_cost`
  returned `0.0` before reaching the table at all. Its fallback scanned model
  keys in dict-insertion order and took the first whose name was a substring
  of the requested model — and `gpt-4o` is a prefix of
  `gpt-4o-mini-2024-07-18`, which is the id OpenAI actually returns. The mini
  model was billed at the full model's rate, ~16x. The scan now takes the
  **longest** matching key, and no longer matches in the reverse direction
  (`model in model_key`), which had let a request for `gpt-4` be priced as
  `gpt-4o` — a different model. Neither case was caught by the miss warning,
  because a wrong match is still a match.

- **`TurnState` no longer records a class name as the provider family.** An
  object that served a turn without declaring a family key left its class
  name in `provider_name`, which keys rate tables, metrics labels, and log
  fields — re-creating one layer down the defect the family/implementation
  split exists to close. The field is left empty instead, so consumers read
  `"unknown"` and the miss warning fires, rather than a plausible-looking key
  missing silently. `provider_impl` still carries the class, and is now read
  from the provider's `impl_name` rather than re-derived, so a provider that
  declares one is honored.

### Security

- **The tool and middleware resolvers no longer put a failed constructor's
  message into the `ConfigurationError` they raise.** Both wrap
  `except Exception` around code the deployment supplies — `import_module`,
  which executes the target module, and the class constructor itself — so the
  text they were interpolating was unbounded. A tool or middleware whose
  constructor opens a database or a cache raises with its connection URL in
  the message, and bots are built lazily on the request path, so that string
  reached an HTTP response body.

  The message now names the class path (which comes from the config, not from
  the exception) and the exception type (a class name); the original travels
  on `__cause__` and reaches the logs from there. The `optional: true` warning
  log is unchanged and still carries the full text.

  Affects `DynaBot._resolve_tool` and
  `dataknobs_bots.middleware.factory.resolve_middleware_from_spec`, and so
  both `build_middleware` and `build_conversation_middleware`. The
  `ImportError` and `AttributeError` branches are untouched: those are scoped
  to failures whose text is module and attribute names.

- **`BotCreationError` no longer returns its `reason` to the caller.** The
  class carries a new `client_safe` class attribute, declared `True` on
  `APIError` and `False` here; a class with it `False` renders as
  `"An unexpected error occurred"` with an empty `detail`, and its message and
  detail are logged instead.

  `BotCreationError` is the only class in the family whose entire payload is
  one free-text field. The others put the authored part in `detail` or
  `config_key` and keep the caller's own input in the message, so what they
  disclose is bounded by construction; `reason` is not, and the pattern this
  package's own documentation showed for it was
  `raise BotCreationError(bot_id, str(e))`. Bots are built lazily on the
  request path, and the tool and middleware factories wrap `except Exception`
  into a message ending in `{e}` — so a tool whose constructor opened a
  database or a cache put the driver's error text, connection URL and
  credentials included, into an HTTP response body.

  The documented pattern is now an authored reason with `raise ... from e`,
  which keeps the underlying error in the logs where it was always wanted.
  A deployment that authors its own `reason` and wants it returned subclasses
  and sets `client_safe = True`.

  The other six API classes and any consumer subclass of `APIError` are
  unaffected: `client_safe` defaults to `True`, because a class written for
  the HTTP boundary is written to be shown.

- Put `HTTPRegistryBackend`'s aiohttp transport under a declared version
  floor. aiohttp was never declared by this package, so it reached
  consumers only transitively — through the `s3` extra's
  `aioboto3` -> `aiobotocore` chain, or a sibling package's own extra —
  and no dataknobs floor governed the version an `HTTPRegistryBackend`
  user actually installed; the floor-resolve audit could not see the
  dependency at all. The new `http` extra pins `aiohttp>=3.14.3`,
  matching the `dataknobs-llm` / `dataknobs-fsm` floors, which clears
  GHSA-cq5v-8q36-5273 / CVE-2026-69244 (CVSS 7.1, out-of-bounds heap
  read in the C response parser while building an error message for a
  malformed chunked response, causing a client-side DoS). That advisory
  is reachable on this path: `HTTPRegistryBackend` is an outbound
  `ClientSession` parsing responses from a consumer-configured registry
  server, and the advisory's `AIOHTTP_NO_EXTENSIONS=1` workaround is not
  set. The floor also sweeps the 3.14.2 fixes GHSA-mfx4-hv73-q22v (CVSS
  6.3, server-side request smuggling via WebSocket upgrade) and
  GHSA-mq44-7p77-q5h7 (CVSS 6.9, WebSocket client RSV1 decompression
  without a negotiated `permessage-deflate` extension), both unreachable
  here, plus the prior `<=3.14.1` sweep (highest CVSS 9.1:
  GHSA-63hf-3vf5-4wqf).

- Declared `aiohttp` in the dev dependency group as well, so the
  in-process `aiohttp.web` test server backing the `HTTPRegistryBackend`
  tests no longer depends on aiohttp arriving transitively via the
  unrelated `aioboto3` dev dependency.

## v0.9.4 - 2026-07-29

### Added

- **Additive `platform_middleware` / `platform_conversation_middleware`
  kwargs on `DynaBot.from_config`.** A second, additive pre-built middleware
  channel distinct from the existing `middleware=` / `conversation_middleware=`
  replace channel: where the replace kwargs substitute the config-resolved
  list, the new kwargs **append** to whatever the resolve produced (config path
  or replace-override path). For installing always-on, cross-cutting middleware
  carrying a live shared collaborator on every bot a platform builds, without
  dropping each bot's own config-declared middleware. Appended middleware runs
  after config middleware (last on every bot-turn hook; innermost-request /
  outermost-response on the onion `conversation_middleware` list). Omitting the
  new params is byte-identical to prior behavior. `BotTestHarness.create` grows
  matching `platform_middleware=` / `platform_conversation_middleware=`
  pass-through params routed through `from_config`.

## v0.9.3 - 2026-07-29

### Added

- **`ReasoningStrategy.on_conversation_evicted(conversation_id)` hook.** A new
  per-conversation teardown seam (default no-op) alongside the existing
  checkpoint hooks. The bot calls it when a conversation's in-memory state is
  reclaimed (LRU eviction or explicit clear) so a strategy holding
  per-conversation resources can release them. Any reasoning strategy with
  per-conversation resources can adopt it.

### Changed

- **`ReasoningStrategy.undo_to_checkpoint` now takes the `ConversationManager`
  as its first argument** — `undo_to_checkpoint(manager, checkpoint_node_id)`.
  A strategy that scopes state per conversation needs the conversation identity
  to revert the correct state, since the bot reaches this hook on undo paths
  where `restore_from_checkpoint` does not run. Strategies overriding this hook
  must add the `manager` parameter; the base implementation remains a no-op.

### Fixed

- **`WizardReasoning` memory banks are now scoped per conversation.** The
  wizard's live memory banks, artifact, and catalog were held as a single
  strategy-instance slot, while one strategy is shared across every conversation
  a bot serves — so two concurrent conversations contended over the same bank
  references, and each turn's state restore could clobber (or, after the recent
  bank-teardown fix, close the live database connection of) another
  conversation's banks. Banks / artifact / catalog are now keyed per
  conversation (via a task-local active-conversation key, so each turn's task
  resolves its own conversation across `await` boundaries), so concurrent
  conversations no longer share, clobber, or tear down each other's bank
  databases. A conversation's owned bank databases are released when its cached
  state is evicted, via the new `ReasoningStrategy.on_conversation_evicted`
  hook, which the bot fires from its single conversation-reclamation choke
  point (error-isolated so a failing release cannot break cache eviction).
  Strategy `close()` now tears down every resident conversation's banks (and
  cancels every conversation's pending ephemeral tasks), not merely the most
  recently accessed one. Undo and rewind — including undoing back through the
  first turn — revert the banks of the conversation being undone rather than
  whichever conversation was last active, even when the undo request runs in a
  fresh task. Single-conversation and sequential-per-turn behavior is unchanged
  — the first conversation adopts the construction-time banks, building them
  exactly once.
- **`AsyncMemoryBank` database lifecycle parity.** `AsyncMemoryBank` now
  supports owned-vs-injected database teardown — an `owns_db` constructor flag
  plus `close()` / `aclose()` methods routed through `close_if_owned`, matching
  the sync `MemoryBank`. `from_dict` now accepts an optional `db` (with inferred
  ownership) and no longer leaks the `AsyncMemoryDatabase` it builds when none is
  supplied. A caller-supplied db is left open (caller-owned); a self-built db is
  owned and closed. Purely additive — `owns_db` defaults to `False`.
- **`ArtifactBank` / `ArtifactBankCatalog` database teardown.** Both now support
  `close()`, releasing the per-section and catalog databases they own and
  matching `MemoryBank`'s owned-vs-injected convention. `ArtifactBank.close()`
  delegates to each section's `MemoryBank.close()` (a section closes its db only
  when it owns it, isolating one section's failure from the rest);
  `ArtifactBankCatalog` gains an optional keyword-only `owns_db` (default
  `False`) and closes a `from_config`-built db while leaving a caller-injected
  db open. `WizardReasoning.close()` now closes the artifact catalog it creates,
  and its teardown cascade is error-isolated per step (extractor, each section
  bank, catalog) so one subsystem's failing `close()` can no longer orphan the
  owned db connections the later steps release — matching `DynaBot.close()`'s
  per-subsystem isolation. `WizardReasoning` also releases the prior turn's
  memory banks before a restore rebuilds them: the strategy outlives a single
  turn, so a persistent (non-memory) bank/section backend previously opened a
  fresh connection on every restore and orphaned the last turn's — now closed
  first via the shared bank-teardown path. The restore also no longer double-
  builds an artifact wizard's section banks (it rebuilt them once for the
  standalone-banks path and again for the artifact, discarding the first set),
  and a bank added to the wizard config after a conversation was last saved is
  now rebuilt fresh and open on restore (previously it could be left
  referencing a just-closed connection, or its restored siblings' data wiped by
  a wholesale re-init). Purely additive.

## v0.9.2 - 2026-07-27

### Added

- **Bounded conversation-manager cache (`max_cached_conversations`).** DynaBot's
  in-memory `ConversationManager` cache is now an access-ordered LRU that can be
  bounded via the new `DynaBotConfig.max_cached_conversations` field. The
  default is `None` (unbounded — byte-for-byte the prior single-user / embedded
  behavior); a positive value caps the cache, evicting the least-recently-used
  conversation once the bound is exceeded so a long-lived multi-conversation
  server cannot accumulate per-conversation state without limit. Eviction
  co-drops the evicted conversation's undo checkpoints through the same single
  teardown choke point as `clear_conversation`, and the in-flight conversation
  of an active turn is pinned so it is never evicted out from under its own turn
  (if every cached conversation is in-flight, the bound is exceeded transiently
  rather than evicting a live turn). Any read of a cached conversation marks it
  most-recently-used. Built on the new `dataknobs_common.BoundedLRUCache`
  primitive. A `0` or negative bound is rejected at config validation.
- **Bounded per-conversation undo history (`max_undo_checkpoints`).** A
  companion `DynaBotConfig.max_undo_checkpoints` field independently bounds the
  undo checkpoints retained per conversation. The default is `None` (unbounded
  — every turn's checkpoint is kept for the life of the conversation, exactly
  as before); a positive value tail-retains only the most recent N checkpoints,
  trimming the oldest from the front so a single very long conversation cannot
  grow its undo history without limit. `undo_last_turn` (relative) is
  unaffected; `rewind_to_turn` to a turn whose checkpoint has been trimmed
  raises a clear "beyond the retained undo window" error rather than silently
  landing on the wrong node. A `0` or negative bound is rejected at config
  validation.
- **Structured ReAct termination reason.** Every ReAct turn now surfaces *why*
  it ended as always-on `reasoning_termination` conversation metadata
  (`{"strategy": "react", "reason": <value>, "iterations_used": <int>}`),
  written independent of `store_trace` (which stays `false` by default). The
  reason is one of six `ReActTerminationReason` enum values (`completed`,
  `max_iterations_reached`, `truncated_tool_call`,
  `duplicate_tool_calls_detected`, `tools_not_supported`,
  `truncation_retry_exhausted`), whose `.value` is byte-identical to the
  reasoning-trace `status` string so the two never drift. An opt-in
  `react:turn:end` callback topic (register on
  `ReActReasoning.termination_callbacks`, EventBus-composable via
  `also_publish_to`) fires the same payload once per terminated turn for
  dashboards / alerting / adaptive policy — zero overhead until a callback or
  fan-out target is registered. `ReActReasoning` advertises
  `Capability.CALLBACK_REGISTRY`. All terminal branches across both the phased
  and monolithic loop paths route through one shared recorder, and the two
  previously log-only reasons (`tools_not_supported`,
  `truncation_retry_exhausted`) are now surfaced uniformly. Additive — existing
  `reasoning_trace` consumers are unaffected.
- **Opt-in in-loop history compaction for the ReAct strategy.** A long
  tool-using turn can grow the conversation history until it trips a model's
  input-context window; `ReActReasoningConfig.history_compaction` (a nested
  `HistoryCompactionConfig`, disabled by default) opts a bot into bounding it.
  When enabled, both ReAct loop sites — the phased `process_input` path DynaBot
  drives and the monolithic `generate` — proactively estimate the path's tokens
  and compact the oldest complete tool iterations when over budget (via
  `ConversationManager.compact_history`), and reactively compact-and-retry once
  on a caught `ContextLengthExceededError`. The budget is a fraction
  (`budget_fraction`, default `0.75`) of the provider's resolved input ceiling,
  with an absolute `history_token_budget` fallback for providers that publish no
  ceiling. Strategy is pluggable via the `CompactionStrategy` extension point:
  `"window"` (default — drop the oldest iterations, LLM-free) or `"summarize"`
  (fold them into one summary node, reusing the bot's provider or a dedicated
  `summary_llm`). A `tool_use` is never separated from its `tool_result`.
  Disabled config is byte-identical to prior behaviour (no estimation, no
  compaction). `SummaryMemory` now composes the shared `dataknobs_llm`
  summarization seam (behaviour unchanged).

### Changed

- **Unified DynaBot's buffered and streaming tool-execution loops onto one
  shared core** (`_run_monolithic_tool_loop` + a per-mode delivery seam in
  `bot/tool_loop.py`). The `chat()` and `stream_chat()` non-phased paths
  previously carried two hand-written copies of the same
  cap / wall-clock-timeout / execute / budget / re-call / cap-warning
  lifecycle, so a loop-control change had to be made in both and could drift.
  The lifecycle now lives in one place; each mode supplies only the axes on
  which it genuinely differs (pending source, usage accounting, the
  clear-before-budget-gate step, and buffered `complete` vs streaming
  `stream_complete` re-invocation). Behavior is unchanged — including the two
  deliberate per-mode asymmetries (streaming clears pending before the budget
  gate so a budget-break flags no orphan; the buffered re-call is deadlined
  while the streaming re-stream is bounded only by the pre-stream budget gate).
  Internal refactor: no public API, config, or behavioral change.

### Fixed

- **Wizard collection-mode records are now revertable by undo.** Records added
  in a collection stage were stored without conversation-tree provenance, so
  they defaulted to the root anchor — an ancestor of every checkpoint — and a
  later-turn undo silently left them in place. Collection records now stamp the
  current node, so undoing a collection turn reverts the records added in it
  (and undoing back to the conversation start clears them entirely).
- **Undoing or rewinding back through the first turn no longer leaves a phantom
  leading user message.** With no system prompt, the first user message
  *becomes* the conversation tree's root node, so the turn-0 undo checkpoint —
  recorded on the then-empty tree — was reoccupied by that message. Undoing to
  the start (`rewind_to_turn(context, -1)`, or `undo_last_turn` on a single-turn
  conversation) switched back onto the reoccupied root and left a stale leading
  user message in the tree path (`manager.messages`, i.e. what the LLM sees)
  while memory rolled back correctly. The next turn then sent two consecutive
  user messages — rejected as a 400 by strict providers (Anthropic) and silent
  context corruption elsewhere. A start-boundary undo now anchors on an empty
  sentinel and resets the conversation to genuinely empty (tree, memory, memory
  banks, and per-turn reasoning-strategy state cleared in lock-step, via the new
  `ConversationManager.reset()` in `dataknobs-llm`), reusing the same
  `conversation_id` on the next `chat()`. Strategy state persisted through the
  conversation-metadata channel (e.g. a wizard's FSM stage/data under
  `manager.metadata["wizard"]`) no longer resurrects on the next turn — `reset()`
  restores the pristine pre-turn-0 seed — and the bank clear is total, removing
  even records stamped at the root node. Two follow-on symptoms are fixed by the
  same change: `UndoResult.remaining_turns`
  no longer reports `1` for an emptied conversation (it counted the phantom), and
  the memory-vs-tree message counts stay consistent through the start boundary.
  The undone first turn's branch is **discarded** (nothing precedes it to branch
  from), so a start-boundary undo reports `branching=False`; later-turn undo is
  unchanged (real-node switch, sibling branch preserved, `branching=True`).
- **`undo_last_turn` / `rewind_to_turn` now distinguish an *emptied* conversation
  from an *absent* one.** After a start-boundary undo the conversation is empty
  but its manager is still cached (active). The guards previously keyed
  "No active conversation" on the manager's state being absent, which — once a
  reset can empty an active conversation — would have reported "No active
  conversation" for a further undo and collapsed the no-op distinction for a
  rewind. They now key that error on the *manager* being absent (never-started /
  evicted). A further `undo_last_turn` on an emptied conversation reports the
  accurate "Nothing to undo", and `rewind_to_turn` to the start remains a clean
  no-op.
- **`DynaBot.rewind_to_turn` to the turn a conversation already sits at is now a
  clean no-op** instead of raising a misleading "Nothing to undo". Rewinding to
  the current turn computes zero undo work, and the trailing empty-result guard
  had turned that legal no-op — and a never-started conversation — into an
  error. It now returns a well-formed `UndoResult` (empty `undone_*` fields,
  `branching=False`, correct `remaining_turns`) for an active conversation, and
  raises the accurate "No active conversation" when there is no manager
  (mirroring `undo_last_turn`). `undo_last_turn`'s own "Nothing to undo" (an
  empty relative undo) is unchanged.

- **`DynaBot.clear_conversation` now reclaims a conversation's undo
  checkpoints, not just its cached manager.** The per-conversation
  `_turn_checkpoints` entry was never pruned — `clear_conversation` dropped the
  cached `ConversationManager` but left the checkpoint list behind, so a
  long-running process that cleared conversations still accumulated checkpoint
  state without bound. Both cached structures are now reclaimed together through
  a single teardown helper (`_drop_conversation_cache`), making it structurally
  impossible for the two to drift apart at teardown again.

- **`DynaBot` now bounds the terminal synthesis of a phased reasoning turn**
  (ReAct) by the wall-clock budget left unspent by `tool_loop_timeout`.
  Previously `tool_loop_timeout` bounded only the tool *loop*; the synthesis
  that runs after an abnormal loop termination (max iterations, duplicate-tool
  break, loop-timeout break) was unbounded, so a slow or hung provider pushed a
  turn's wall-clock to `tool_loop_timeout` + an unbounded finalize. The finalize
  is now bounded at the DynaBot dispatch layer — buffered via `asyncio.wait_for`,
  streaming via a per-chunk deadline whose source generator (and the source's
  own teardown) are themselves bounded so nothing leaks or hangs. On timeout the
  turn degrades gracefully to a config-overridable
  `DynaBotConfig.tool_loop_timeout_message` (default provider-neutral) carrying
  `finish_reason="length"` + `metadata['termination_reason']="finalize_timeout"`
  (`truncated` stays `False` — a wall-clock timeout is not a token-budget
  cutoff). No `ReasoningStrategy` / `PhasedReasoningProtocol` /
  `StreamingPhasedProtocol` signatures change; `Simple` / `Wizard` / `ReAct`
  strategy code is untouched.

### Added

- **`DynaBotConfig.tool_loop_timeout_message`** — the user-facing text surfaced
  when the phased terminal synthesis exceeds the remaining `tool_loop_timeout`.
  Defaults to a provider-neutral string; override to localize / brand / soften
  the degraded response without subclassing.
- **`ReActReasoningConfig.truncation_retry_max_tokens`** — opt-in, bounded,
  single retry of a tool-call turn the provider truncated at the token budget
  (`LLMResponse.truncated`). When set (positive int), the truncated turn is
  retried **once per truncated tool-call iteration** at the larger `max_tokens`
  before being abandoned, branching off the incomplete `tool_use` so no dangling
  block re-enters history. Shared across the phased and monolithic ReAct paths.
  A non-positive value is rejected at config construction; a still-truncated or
  erroring retry degrades to the existing abandon-and-synthesize path (never a
  hard turn failure). Default (`None`/unset) is byte-identical to the prior
  abandon-only behavior.

## v0.9.1 - 2026-07-18

## v0.9.0 - 2026-07-15

### Changed

- The knowledge-storage backends (`FileKnowledgeBackend`,
  `S3KnowledgeBackend`, `InMemoryKnowledgeBackend`) advertise their
  conditional-metadata-write / optimistic-concurrency contract under
  `Capability.CONDITIONAL_WRITE` — the layer-neutral identifier now shared
  with the `dataknobs-data` record backends. A consumer querying
  `backend.supports(Capability.CONDITIONAL_WRITE)` discovers the CAS
  contract that `get_state_version` + `expected_version` on
  `set_ingestion_status` enforce. (The former, metadata-flavored member was
  removed from `dataknobs-common`; querying the old identifier no longer
  resolves.)

## v0.8.3 - 2026-07-07

### Changed

- `S3KnowledgeBackend`'s `session_config` parameter is now typed as
  `AwsSessionConfig` (renamed from `S3SessionConfig` and relocated to
  `dataknobs-common`). The backend now imports it from
  `dataknobs_common.aws`. The deprecated `S3SessionConfig` alias still
  resolves, so external callers passing one keep working; prefer
  `AwsSessionConfig` from `dataknobs_common.aws`.

## v0.8.2 - 2026-06-29

## v0.8.1 - 2026-06-23

## v0.8.0 - 2026-06-22

### Added

- Knowledge-storage backends support optimistic-concurrency state writes:
  read the current state-version token via `get_state_version`, pass it as
  `expected_version` to `set_ingestion_status`, and a stale token raises
  `ConcurrencyError` instead of clobbering a concurrent writer's status
  transition. S3 uses a native `If-Match` conditional PUT on the metadata
  object's ETag; in-memory uses a monotonic version counter; the file
  backend guards the read-check-write critical section with an ephemeral
  advisory `flock` on POSIX hosts. The token is opaque (round-trip it
  verbatim). Omitting `expected_version` preserves the unconditional
  write. The three in-tree backends advertise
  `Capability.TRANSACTIONAL_METADATA`. Snapshot writes are unaffected —
  they are content-addressed and write-once by identity.
- `KnowledgeIngestionManager` accepts a `tenant_context_config` mapping
  selecting the per-tenant state-context shape (`bound` / `prefixed` /
  `shared_corpus` / `single`) via the shared tenant-context factory; the
  manager's bound tenant and per-call domain remain authoritative, so the
  config never re-targets identity. A tenant-requiring shape on a manager
  with no bound `tenant_id` raises at construction. Default behavior
  (no config) is unchanged.
- A tenant-bound `KnowledgeIngestionManager` (constructed with
  `tenant_id`) isolates its per-tenant ingestion **status** on a shared
  knowledge backend by routing every backend state operation through a
  per-tenant context. Two managers bound to different tenants but sharing
  one backend track independent ingestion status; an unbound manager's
  storage paths are identical to single-tenant. Change detection stays
  minimal for every tenant: a tenant's snapshot diff resolves against the
  shared domain content lineage (content — and the snapshot lineage
  derived from it — is shared by `domain_id`; only ingest status is
  per-tenant). The manager advertises `Capability.TENANT_SCOPED_STATE`
  and `Capability.SNAPSHOT_ISOLATION` (not `Capability.TENANT_SCOPED_LOCKS`
  — cross-replica serialization of concurrent ingests for the same tenant
  and domain remains the ingest orchestrator's distributed lock).
- `KnowledgeResourceBackend` state operations (`set_ingestion_status`,
  `get_info`, `get_checksum`, `has_changes_since`, `list_changes_since`)
  accept an optional keyword-only `ctx: TenantContext` argument. When
  supplied, the backend isolates per-tenant ingest **state** (the
  metadata document and the per-version snapshot lineage) under the
  context's state-key prefix; **content** (the files under
  `{domain_id}/content/`) stays keyed by `domain_id`, so tenants of the
  same knowledge base share content but keep independent ingest state.
  Omitting `ctx` — or passing a context whose `state_key_prefix()` is
  empty, e.g. a `SingleTenantContext` — preserves single-tenant behavior
  and storage paths exactly. The in-tree file, S3, and in-memory
  backends advertise `Capability.TENANT_SCOPED_STATE` and
  `Capability.SNAPSHOT_ISOLATION`. Snapshot *versions* are content
  identities and the snapshot *map* is shared domain content state, so
  per-tenant change detection (`has_changes_since` / `list_changes_since`
  with a `ctx`) diffs against the shared domain-keyed content-snapshot
  lineage — a tenant with no tenant-scoped snapshot of its own still gets
  a minimal diff rather than a forced full re-ingest. `get_info` for a
  tenant that has not written ingest state yet returns a fresh default
  view (`PENDING`, no `generation` token) uniformly across all three
  backends — never the shared domain view.
- `StateBridge[InboxT, OutboxT]` Protocol with `InboxOnlyBridge`,
  `PeekBridge`, `BiDirectionalBridge`, `SubsetBridge`, and
  `SubscribingBridge` reference implementations in
  `dataknobs_bots.reasoning.state_bridge` (re-exported from the package
  root). Codifies the named-key state-bridging contract used by the
  wizard inbox hook — a bridge reads (`read_inbox`) and writes
  (`write_outbox`) named keys on a host's `metadata` mapping. Consumers
  compose bridges with the lifecycle-hook surface for consume-on-read
  (`InboxOnlyBridge`), peek-without-consume (`PeekBridge`), symmetric
  assign-or-merge (`BiDirectionalBridge`), projected-subset
  (`SubsetBridge`, which accepts a bare callable or a scope projector —
  a source-honoring projector projects the write value, while a
  source-capturing projector such as `WhitelistProjector` projects its
  captured source), or observability-aware (`SubscribingBridge`, firing
  `CallbackRegistry` callbacks on every read and write; dispatch is
  synchronous, so a coroutine-function callback raises `TypeError`)
  state bridging. The Protocol's `InboxT`/`OutboxT` type parameters are
  variance-annotated (covariant return / contravariant parameter),
  matching the sibling `ScopeProjector` / `ResourceResolver` families.
- A propose-then-consent example wizard
  (`examples/configs/wizards/propose-consent-wizard.yaml`) demonstrating
  the `intent_confirm:` stage primitive end-to-end: a proposal stage that
  routes to accept / decline / alternative targets, including an
  alternative intent that extracts the user-named value via the LLM tier.
- USER_GUIDE guidance distinguishing the two confirmation surfaces: the
  forward-looking `intent_confirm:` stage primitive (propose, then route
  on the reply) versus the backward-looking `ConfirmationEvaluator`
  (double-check values already gathered, driven by the
  `confirm_first_render` / `confirm_on_new_data` stage knobs).

### Changed

- `make_metadata_inbox_hook` now routes its read step through
  `InboxOnlyBridge`. The public surface
  (`make_metadata_inbox_hook(*, inbox_keys, merge_fn=None)`,
  `write_to_inbox(manager, key, payload)`) and behavior (consume-on-read,
  plain `dict.update` default merge, empty-dict no-op, multi-key support,
  non-mapping payload WARNING) are unchanged.
- The `KnowledgeResourceBackend` protocol (and its shared mixin) now
  documents an async-transport contract — async file methods use an async
  transport or offload blocking disk I/O off the event loop. ruff's
  `ASYNC` lint family is now enforced for the package, catching blocking
  I/O inside `async def` code at lint time. See the `async-transport`
  authoring rule.
- **Knowledge backends no longer block the event loop on storage I/O.**
  The `S3KnowledgeBackend` performs all S3 operations through an async
  (`aioboto3`) client instead of a synchronous `boto3` client, so reads,
  writes, listings, streaming, and change detection run without stalling
  the running event loop. The `dataknobs-bots[s3]` extra now installs
  `aioboto3` alongside `boto3`. The `FileKnowledgeBackend` offloads its
  filesystem reads/writes to a worker thread. Passing a file-like
  `content` argument to any backend's `put_file` no longer blocks the
  event loop on the read. `RAGKnowledgeBase` document ingest offloads its
  file reads, including the `knowledge_base.{yaml,yml,json}` config probe
  and parse that `load_from_directory` performs when no explicit config is
  supplied. The public method signatures and behavior are unchanged — only
  the threading/transport underneath. (One-time botocore data loading on
  the first S3 client creation per session is an `aioboto3`/`botocore`
  characteristic in the shared session factory and is unchanged.)

- The knowledge ingestion manager now publishes lifecycle events on the
  `ingest:domain:start` / `ingest:domain:end` topics (was a single
  `knowledge:ingestion` completion event). The end event fires on both
  success (`status="completed"`) and failure (`status="failed"`), and a
  start event fires at the head of every run. The former
  `knowledge:ingestion` topic is gone; consumers re-subscribe to one or
  both new topics (a single `EventBus` `pattern="ingest:domain:*"`
  subscription catches both with one handler). The manager's `__init__`
  signature and the `event_bus` cross-replica semantic are unchanged.

- **`RAGKnowledgeBase.close()` only closes collaborators it owns.** A
  vector store and embedding provider built from config are owned and
  closed on `close()` as before. A store or provider injected via
  `RAGKnowledgeBase.from_components(vector_store=…, embedding_provider=…)`
  is caller-owned and left open, so a consumer sharing one store/provider
  across several knowledge bases can close each base independently without
  tearing down a resource the others still depend on. Consumers that build
  the knowledge base from config see no change.

- **`close()` across the bot stack now tears down only collaborators the
  holder owns.** A holder that builds a collaborator from config owns its
  lifecycle and closes it; a holder handed a pre-built collaborator leaves
  it open for its real owner. This makes sharing one backing resource
  across several holders safe — closing one holder no longer tears down a
  collaborator the others still depend on. Applies to:

  - **`DynaBot`** — the knowledge base, memory, reasoning strategy, and
    conversation storage are closed only when built from the bot's config.
    Collaborators injected via `DynaBot.from_components(...)` (or the
    pre-built `DynaBot(llm=...)` constructor) are caller-owned and left
    open, so several bots can share one knowledge base or storage backend
    and each can be closed independently. (The main LLM keeps its existing
    behavior: a provider passed to `from_config(config, llm=...)` is
    caller-owned; one the bot builds itself is closed.)
  - **`MemoryBank`** — a caller-supplied `db` is left open by default;
    `MemoryBank.from_dict(db=None)` builds a db the bank owns and closes.
    A new `owns_db` parameter overrides the inference when a db is built
    elsewhere for a bank's exclusive use.
  - **`VectorKnowledgeSource`** — never closes the knowledge base it
    wraps (the KB is supplied by the caller and typically shared with the
    owning bot and other sources). A new `owns_kb` parameter opts into
    closing a dedicated KB.
  - **`GroundedReasoning`** — closes only the extractor and sources it
    built from config. An extractor or source injected by the caller, and
    the query provider (always the bot's LLM or an injected override), are
    left open. `add_source` gained an `owns` keyword to add a shared
    source without transferring ownership.

  `CompositeMemory` and `HybridReasoning` own the children they compose on
  every construction path (the children are dedicated to the parent, not
  shared), and each child independently protects any backing resource it
  was handed — so they continue to close their children unconditionally.
  Consumers that build everything from config see no behavior change.

  **Migration — default `close()` semantics flipped for direct
  construction.** The new default for a directly-constructed holder is
  *leave the injected collaborator open*, the inverse of the prior
  *close-it* default. If you relied on the old behavior — building a
  resource specifically for one holder and using the holder's `close()`
  to release it — you must now opt back into ownership explicitly, or the
  resource stays open (a leak):
  - `MemoryBank(db=db).close()` no longer closes `db`. Pass
    `MemoryBank(db=db, owns_db=True)` (or `from_dict(..., owns_db=True)`)
    for a db built for that bank's exclusive use. `from_dict(db=None)`
    still builds and owns its db. (Passing `owns_db=False` alongside
    `db=None` is contradictory — the bank holds the only reference — so it
    is ignored with a `UserWarning` and the internally-built db is owned.)
  - `VectorKnowledgeSource(kb).close()` no longer closes `kb`. Pass
    `VectorKnowledgeSource(kb, owns_kb=True)` for a KB dedicated to that
    source.
  - `DynaBot(knowledge_base=kb, memory=…, …).close()` no longer closes the
    pre-built collaborators it was handed. Build the bot from config (which
    owns what it builds), or close the shared collaborators yourself.

  Consumers that build everything from config are unaffected — ownership is
  inferred correctly on the config path.

- Re-platformed `LifecycleHooks` (and via composition,
  `WizardHooks`) onto the new `dataknobs_common.callbacks`
  `CallbackRegistry` substrate. The consumer-facing surface
  (`on_turn_start` / `on_turn_end` registration with chaining and
  per-stage scoping; `trigger_turn_start` / `trigger_turn_end`
  triggers; `turn_start_count` / `turn_end_count` properties;
  `clear()` with drain-in-place identity; `from_config` with
  dotted-path callback resolution) is unchanged. New: a `registry`
  property exposes the underlying `CallbackRegistry` as a
  documented escape hatch — consumers can swap orderings
  (`hooks.registry.set_ordering(PriorityOrdering())`), register
  priority-tagged callbacks
  (`hooks.registry.register("turn_start", cb, priority=-100)`), or
  fan turn-lifecycle events out to an `EventBus`
  (`hooks.registry.also_publish_to(bus, topic_prefix="wizard:")`)
  without monkey-patching. A new `LifecycleHooks.load_config(config)`
  instance method registers callbacks against the existing registry
  in place — used internally by `WizardHooks.from_config` so the
  embedded `LifecycleHooks` constructed in `WizardHooks.__init__`
  survives the config load; consumers caching a
  `hooks.lifecycle.registry` reference (e.g. to install a custom
  ordering or fan-out target) keep that reference valid across
  `WizardHooks.from_config`. `LifecycleHooks` declares
  `Capability.CALLBACK_REGISTRY` via `CapabilityMixin` for
  feature-probe-before-use composition. The `WizardHooks.clear()`
  invariant ("lifecycle instance identity is preserved") now also
  extends to the underlying registry — `hooks.registry is
  hooks.registry` survives `clear()` so consumer customizations
  (custom ordering, fan-out targets) persist across resets.
- Replaced the `aioresponses`-driven HTTP fixture in
  `tests/test_registry_http_backend.py` with an in-process
  `aiohttp.web` test server (`_MockHttpServer`). Every request now
  flows through the real aiohttp client/server stacks — wire-format
  bugs (encoding, multi-valued query params, headers) surface in the
  test instead of in production. Eliminates the temporary
  `aiohttp<3.14` cap on dev deps and unblocks the
  `aiohttp>=3.14.1` floor bumps in `dataknobs-llm` and
  `dataknobs-fsm`. Test-side changes: the auth-header test now
  asserts the actual `Authorization: Bearer ...` header reaches the
  server (previously the assertion was implicit), and the
  no-wire-protocol peek test now asserts an empty server-side query
  mapping rather than scanning the URL string.
- `LifecycleHooks` / `WizardHooks` turn-lifecycle triggers
  (`trigger_turn_start` / `trigger_turn_end`) take a single opaque
  `event: dict[str, Any]` argument; `TurnHookCallback` is
  `Callable[[dict[str, Any]], None | Awaitable[None]]`. The
  wizard publishes canonical event keys `stage`, `phase`,
  `reason`, `manager`, and `state`; adopters attach
  subsystem-specific keys (e.g. the stream-abandonment path
  attaches `state_saved=False`) without extending the protocol
  signature. Hook callbacks read named keys off the event
  payload. Documented in user-guide.md "Turn-Lifecycle Hooks".
- `WizardReasoning` fires `on_turn_end` on every turn exit, with a
  per-site `reason` discriminator on the event payload:
  `"normal"` from the canonical `finalize_turn` /
  `stream_finalize_turn` save→fire path (including subflow-push
  variants and the streaming counterpart); `"amendment"` /
  `"navigation"` from the `begin_turn` early-return paths;
  `"clarification"` / `"collection_help"` / `"collection_loop"` /
  `"confirmation"` / `"validation_error"` from the
  `process_input` early-return paths; `"abandoned"` (with
  `state_saved=False`) from the stream-abandonment path when the
  consumer calls `aclose()` on the async iterator; and
  `"advance"` (with `manager=None`) from the non-conversational
  `advance()` API. A consumer observing `chat()` and
  `stream_chat()` sees the same fire-points for the same
  conversation outcomes. State-mirroring consumers ignore
  non-advancing turns by filtering on
  `event.get("reason") == "normal"`; observability / audit /
  metric consumers typically observe every exit and tag records
  with the reason. The shared `_fire_turn_end_hook` helper
  resolves the active subflow FSM at every site, so
  `event["stage"]` reflects the deepest pushed subflow at the
  fire-point.
- `make_metadata_inbox_hook` reads `manager` / `state` / `stage`
  from the event payload — matches the
  `Callable[[dict[str, Any]], ...]` trigger contract.

### Added

- `JinjaInputsProjector` in `dataknobs_bots.prompts.scope` (re-exported
  from `dataknobs_bots.prompts`). A `ScopeProjector` implementation that
  evaluates declarative Jinja-expression inputs against a base context;
  lazily imports `jinja2`. Sandboxed by default — when no `env=` is
  supplied it builds a `SandboxedEnvironment` (via `create_template_env`),
  so attribute-traversal escapes raise rather than leak interpreter
  internals; an unsandboxed environment is opt-in via `env=`. The
  `strict=` flag (default `True`) selects whether a failing expression
  propagates or is logged-and-skipped (`strict=False`).
- Wizard stages accept a declarative `inputs:` mapping
  (`name -> Jinja expression`). The renderer evaluates each expression
  against the assembled template context — through the wizard's sandboxed
  environment — and merges the derived variables into the template scope
  (later-wins), so response templates can reference computed values
  without subclassing the renderer. Declared inputs are evaluated against
  author params, collected user data, and any extra context
  (bank/artifact). A malformed expression degrades gracefully: it is
  logged and skipped, never aborting the stage render.
- `dataknobs_bots.knowledge.events` module — the canonical
  knowledge-layer event topic constants (`INGEST_DOMAIN_START`,
  `INGEST_DOMAIN_END`, `INGEST_METADATA_WRITE`,
  `INGEST_SNAPSHOT_WRITE`, all `Final[str]`), the
  `KnowledgeTriggerPayload` `TypedDict` documenting the ingest-trigger
  payload shape, and the `TenantFilteredCallback` adapter that
  short-circuits an event callback on tenant mismatch. All re-exported
  from `dataknobs_bots.knowledge`.
- `KnowledgeIngestionManager.lifecycle_callbacks` — an in-process
  `CallbackRegistry` that fires `INGEST_DOMAIN_START` at the head of
  every `ingest()` / `ingest_changes()` and `INGEST_DOMAIN_END` at the
  tail (on success **and** failure). When the manager is constructed
  with an `event_bus`, the registry auto-composes
  `also_publish_to(event_bus)` so the lifecycle events fan out to the
  bus for cross-replica observability. Payloads carry `tenant_id` only
  when the manager is tenant-bound. Advertises
  `Capability.INGEST_EVENT_PUBLICATION` / `CALLBACK_REGISTRY` always,
  and `EVENT_BUS_EMISSION` only when constructed with an `event_bus`
  (config-dependent, via `DynamicCapabilityMixin`) — a busless manager
  never fans out, so `require_capability(mgr, EVENT_BUS_EMISSION)`
  reports the truth rather than a false positive.
- `KnowledgeResourceBackend.subscribe_to_changes(bus, *, kinds=None,
  domain_id=None, handler)` and the `changes_subscription(...)` async
  context manager — compose a backend's `key_pattern()` with
  `EventBus.subscribe()` in one call; `kinds` defaults to
  `{KnowledgeKeyKind.CONTENT}` (observe consumer writes, skip
  DK-managed state writes). Default implementations ship on
  `KnowledgeResourceBackendMixin`.
- Backend state-write observability — every backend fires
  `INGEST_METADATA_WRITE` / `INGEST_SNAPSHOT_WRITE` on its
  `state_write_callbacks` `CallbackRegistry` after each metadata /
  snapshot write, via the shared `_fire_state_write` helper on
  `KnowledgeResourceBackendMixin`. Zero-overhead when no callbacks are
  registered. Backends advertise
  `Capability.BACKEND_STATE_OBSERVABILITY` /
  `KEY_PATTERN_FILTERING` / `CHANGE_SUBSCRIPTION` / `CALLBACK_REGISTRY`.
- `IngestOrchestrator` honors an optional `payload["key"]` on trigger
  events: when present it classifies the key via the resolved
  backend's `classify_key(...)` and skips non-`CONTENT` keys (so the
  DK-managed state writes the ingest performs do not re-trigger
  ingestion). Absent `key` proceeds unchanged.
- `BackendKeyDiscriminator(backend)` adapter (frozen dataclass) —
  wraps any `KnowledgeResourceBackend`'s `classify_key` method
  through the generic `Discriminator[str, KnowledgeKeyKind]`
  Protocol from `dataknobs_common`. Use when composing
  backend-key classification with other discriminators
  (payload-field routing, multi-field event-handler dispatch)
  through the generic protocol shape without coupling consumer
  code to the backend interface directly. `frozen=True` gives
  `__eq__` / `__hash__` keyed on the wrapped backend so two
  adapters around the same backend instance compare equal
  (useful for adapter-cache lookups). Exported from
  `dataknobs_bots.knowledge` and
  `dataknobs_bots.knowledge.storage`.
- `KnowledgeKeyKind` enum (`CONTENT` / `METADATA` / `SNAPSHOT` /
  `UNKNOWN`) names the three classes of keys every in-tree
  `KnowledgeResourceBackend` writes. External event sources
  (S3 → EventBridge / SQS / SNS / Lambda; filesystem inotify;
  GCS Pub/Sub) use the enum to filter to the consumer-controlled
  `CONTENT` subtree and skip the DK-managed `METADATA` /
  `SNAPSHOT` writes the ingestion manager performs during
  ingest, which would otherwise create a positive feedback loop.
  Exported from `dataknobs_bots.knowledge` and
  `dataknobs_bots.knowledge.storage`.
- `KnowledgeResourceBackend.classify_key(key) -> KnowledgeKeyKind`
  and `KnowledgeResourceBackend.key_pattern(kind=CONTENT,
  domain_id=None) -> str` — the helper API for source-level
  filtering (`key_pattern`) and per-event filtering
  (`classify_key`). Each first-party backend ships its own
  `key_pattern` returning the backend-native pattern dialect:
  S3 wildcard syntax for `S3KnowledgeBackend` (suitable for
  EventBridge `wildcard` rules or composed into bucket-notification
  `prefix` + `suffix` pairs), a `pathlib.Path.glob`-shaped pattern
  for `FileKnowledgeBackend`, and `""` for `InMemoryKnowledgeBackend`
  (no event-source filter is meaningful in-process; the empty
  sentinel preserves protocol symmetry). `key_pattern(UNKNOWN)`
  raises `ValueError` on the S3 and file backends (fail closed —
  there is no shape for "unrecognized keys"). `classify_key`
  inherits a canonical implementation from
  `KnowledgeResourceBackendMixin` that any out-of-tree backend
  honoring the documented layout gets for free.
- `KnowledgeResourceBackendMixin.METADATA_FILE` /
  `CONTENT_DIR` / `SNAPSHOTS_DIR` `ClassVar[str]` declarations.
  The canonical layout constants live once at the contract layer
  so every in-tree backend resolves them via MRO; an out-of-tree
  backend mixing in `KnowledgeResourceBackendMixin` inherits
  identical values without redeclaring them.
- New docs page **"Event triggers for knowledge backends"**
  (`packages/bots/docs/knowledge/event-triggers.md`) describes
  the layout diagram, the positive-feedback-loop failure mode
  external triggers must avoid, and the wiring recipes per
  source (S3 → EventBridge, S3 bucket notification, filesystem
  inotify, GCS Pub/Sub, and a generic
  `classify_key`-driven fallback). Cross-linked from the
  knowledge-base ingestion guide and `IngestOrchestrator` page.
- `intent_confirm:` wizard stage primitive — declarative block that
  expands at load time into `mode: conversation` +
  `response_template` + (optional) `clarification_template` +
  `intent_detection` + `schema` + `transitions`. Built atop
  `dataknobs_llm.intent` for classifier resolution. Wizard authors
  declare `proposal_template`, an `intents:` map (each intent
  carries a `target`, optional `keywords:` override, optional
  `extract:` field name, optional per-intent `llm_fallback: true`),
  optional `on_no_match: {target?, clarification_template?}` for
  fallback routing and reprompt copy, block-level `llm_fallback:
  true` (shorthand for promoting the classifier to a keyword→LLM
  composite chain), and block-level `negation_filter: true` (wraps
  the resolved classifier in `NegationFilter`); the
  `IntentConfirmSynthesizer` expands the rest. The synthesized
  `intent_detection` block sets `per_intent_booleans: true`, so the
  matched intent writes `state.data[intent_name] = True` (the
  back-compat `state.data["_intent"]` key is still written too).
  Zero new runtime branches — every step runs through existing
  wizard machinery. `IntentConfirmSynthesizer.validate` rejects
  empty / non-mapping `intents`, intent specs that are not mappings,
  intents missing a `target`, and intent names that collide with
  the reserved `_intent` runtime key — all surfaced as
  `ConfigurationError` with the offending stage and intent named.
  After expansion the synthesizer removes the original
  `intent_confirm:` block from the stage dict so the FSM-metadata
  layer carries only the synthesized primitives — no parallel
  source of truth. Documented in user-guide.md "Wizard-as-advisor:
  intent confirmation".
- `clarification_template:` stage field — optional template rendered
  on re-render of a conversation-mode stage when no extraction or
  intent matched. First render still uses `response_template`;
  subsequent renders consult `clarification_template` when set.
  Populated automatically by `intent_confirm:`'s `on_no_match`;
  hand-rolled conversation-mode stages can use it too.
- Stage-synthesizer registry — `StageSynthesizer` Protocol plus
  `register_stage_synthesizer` / `unregister_stage_synthesizer` /
  `iter_stage_synthesizers` / `validate_no_conflicting_fields`
  exports from `dataknobs_bots.reasoning` (also re-exported from
  `dataknobs_bots.reasoning.wizard_loader` for single-module
  imports), and a `stage_synthesizer_backends:
  Registry[StageSynthesizer]`. `register_stage_synthesizer` uses
  `allow_overwrite=True`; re-registering the same `field`
  overwrites by design (consumers commonly replace the in-tree
  synthesizer with a customized one). The loader runs registered
  synthesizers in a dedicated phase BEFORE `_validate_config` and
  FSM translation, so downstream validator and FSM build code see
  only the normalized shape. `IntentConfirmSynthesizer` is the
  in-tree reference adopter; it auto-registers at module import.
  Documented in user-guide.md "Shipping your own wizard stage
  primitive".
- `WizardConfigBuilder` recognizes `intent_confirm:` and
  `clarification_template:` on stage configs. `intent_confirm:` is
  carried as a raw dict because the synthesizer expands it before
  the builder's typed `StageConfig` runs.
- `WizardExtractor.detect_intent` dispatches through
  `dataknobs_llm.intent.intent_classifier_backends`. The
  `intent_detection:` block selects a backend via `classifier:`
  (preferred) and optional `classifier_config:` forwarded to the
  backend's factory. The legacy `method: keyword | llm` /
  `llm_fallback: true` shape is promoted automatically at runtime
  (`method: keyword` → `classifier: keyword`; `method: llm` →
  `classifier: llm`; per-intent or block-level `llm_fallback:
  true` → `classifier: composite` with a keyword→LLM chain), so
  existing wizard YAML continues to work unchanged. New YAML should
  use the `classifier:` shape. `negation_filter: true` wraps the
  resolved classifier in `NegationFilter`. When `per_intent_booleans:
  true` is set on the block, the matched intent also writes
  `state.data[intent_name] = True` alongside the existing
  `_intent` key.
- `dataknobs_bots.reasoning.lifecycle.LifecycleHooks` —
  strategy-agnostic turn-lifecycle hook surface
  (`on_turn_start` / `on_turn_end`, with optional per-stage scope).
  Loadable from config via `LifecycleHooks.from_config({...})` with
  dotted-path callback resolution (sync or async callables). Importable
  standalone — adoptable by any `ReasoningStrategy` implementation, not
  wizard-specific. Documented adoption recipe in user-guide.md
  "Adopting LifecycleHooks in Your Own Reasoning Strategy".
  Public introspection surface: `turn_start_count` /
  `turn_end_count` properties and a `clear()` method that drains
  every registered callback in place (preserving instance identity).
- Wizard turn-lifecycle hook surface: `on_turn_start` and
  `on_turn_end` on `WizardHooks`, following the existing
  `on_enter` / `on_exit` / `on_complete` registration shape.
  `WizardHooks` composes `LifecycleHooks` internally and forwards
  registration / triggering through the embedded instance while
  preserving its own error-handler fan-out (failing hooks still
  reach `on_error` callbacks before re-raising). `on_turn_start`
  fires from `begin_turn` (AFTER per-turn ephemeral-key clear,
  BEFORE early-return dispatch) and symmetrically from `greet` so
  bot-initiated greetings inherit the surface. `on_turn_end` fires
  on every turn exit with a per-site `reason` discriminator (see
  the Changed entry for the full reason table). The embedded
  instance is exposed read-only via the `WizardHooks.lifecycle`
  property for detachment scenarios. `WizardHooks.clear()` drains
  the embedded lifecycle in place alongside the legacy hook lists;
  `WizardHooks.hook_count` reports `"turn_start"` / `"turn_end"`
  counts alongside the legacy keys.
- `WizardReasoning.add_turn_start_hook(callback, *, stage=None)` /
  `WizardReasoning.add_turn_end_hook(callback, *, stage=None)` —
  public runtime-attach surface for turn-lifecycle callbacks.
  Lazy-creates the embedded `WizardHooks` if none was supplied at
  construction, then delegates to `WizardHooks.on_turn_start` /
  `on_turn_end`. Pairs with the canonical fire-points so
  observability / audit consumers can attach hooks without
  re-wiring construction.
- `WizardReasoningConfig.manager_metadata_inbox_key: str | list[str] | None`
  — typed knob that auto-registers an `on_turn_start` hook bridging
  one or more `manager.metadata` keys into `wizard_state.data` at the
  start of every turn. Consume-on-read (popped, not get'd);
  None-as-eviction with the default merge; empty-dict and
  non-mapping payloads tolerated; `greet` inherits the bridge.
  Cross-turn bridge from per-stage sub-strategy output into the
  wizard's transition-eval scope without widening the
  `{data, has, bank, artifact}` safe-eval scope. `None` (default)
  disables the bridge — zero behaviour change.
- `WizardReasoningConfig.inbox_merge_fn` — optional merge function
  for inbox payloads. Defaults to `dict.update` (shallow merge).
  Consumers supply deep-merge or conflict-resolving mergers as
  needed.
- `dataknobs_bots.reasoning.wizard_inbox` module:
  `make_metadata_inbox_hook` factory (for consumers building
  custom variants of the bridge) and `write_to_inbox` helper
  (for writer-side code publishing payloads for the next turn's
  consumption).
- `WizardReasoning` now forwards every construction collaborator
  threaded through `WizardReasoning.from_config(config, **kwargs)`
  (`knowledge_base`, `prompt_resolver`, `prompt_envelope`, every key in
  `reasoning_components`, and any other consumer-supplied kwarg) to the
  per-stage sub-strategy that `WizardResponder._resolve_stage_strategy`
  builds. The forwarding leverages the
  `StructuredConfigConsumer.components` pass-through surface plus the
  new `forwardable_components()` mixin helper (see `dataknobs-common`
  CHANGELOG). `WizardReasoning` declares
  `INTERNAL_COMPONENTS = frozenset({"wizard_fsm"})` so the outer
  wizard's FSM handle is never forwarded to a sub-strategy; every other
  collaborator flows through opaquely. Closes the structural blocker
  for per-stage `reasoning: pipeline` (or other composing) sub-
  strategies that need construction-time collaborators (e.g. a
  knowledge-base-aware `GroundedRetrieval` step). Strictly additive — a
  wizard constructed without extras forwards an empty dict (no-op
  spread on the registry call); the 174 existing direct-ctor call
  sites opt in via the new `_forwarded_components` keyword-only
  parameter on `WizardReasoning.__init__`. Consumer composing
  strategies adopting the same mixin pattern get the same forwarding
  discipline (see user-guide.md "Building your own composing
  strategy").
- `RAGKnowledgeBase(tenant_id=...)` (also accepted on
  `RAGKnowledgeBaseConfig`) binds a tenant identity to the KB. When set,
  every write auto-stamps `tenant_id` into chunk metadata AND folds it
  into the chunk-id prefix; every read AND-composes
  `{"tenant_id": tenant_id}` into the vector-store search filter.
  Defaults to `None` (single-tenant byte-identical posture — no chunk-id
  derivation change, no read-filter change). Write-side precedence:
  auto-derived bound tenant wins on collision with caller-supplied
  `extra_metadata={"tenant_id": …}` so identity cannot be silently
  re-tagged; read-side precedence inverts (explicit-filter-wins) so
  admin tooling can legitimately read across tenants by passing an
  explicit `filter_metadata={"tenant_id": …}`. The asymmetry matches
  the differing write/read threat models and is documented on
  `_resolve_read_filter`.
- `KnowledgeIngestionManager(tenant_id=…, keyword-only)` binds a
  tenant identity to the manager. Threaded uniformly into every chunk
  the manager writes (via the new `_compose_extra_metadata` helper —
  auto-derived `domain_id` / bound `tenant_id` / TOMBSTONE-swap
  `_generation` token all win over caller-supplied `extra_metadata`)
  AND into every destination-side write/delete filter (via the new
  `_scope_for_tenant` helper applied to the CLEAR_FIRST clear, per-file
  purges, tombstone scope, rollback scope, and reconcile scope). Two
  managers bound to distinct tenants but pointing at the same shared
  `RAGKnowledgeBase` now produce disjoint chunk-id namespaces, do not
  delete each other's rows on CLEAR_FIRST / per-file purges, and do not
  tombstone/un-tombstone each other's rows during TOMBSTONE swaps.
  `tenant_id=None` (default) is the single-tenant byte-identical
  posture.
- `extra_metadata=` keyword-only parameter on
  `KnowledgeIngestionManager.ingest()` and
  `KnowledgeIngestionManager.ingest_changes()`. Mapping merged into
  every chunk's metadata before identity auto-derivation; auto-derived
  identity tags (`domain_id`, the bound `tenant_id`, the `_generation`
  token) win on collision; non-identity keys (`region`, `cohort`, any
  custom tag) are preserved as-is. Symmetric with the
  `extra_metadata` parameter on `RAGKnowledgeBase.ingest_from_backend`.
- `extra_metadata=` and `tenant_id=` keyword-only parameters on every
  direct `RAGKnowledgeBase` entry point — `load_markdown_text`,
  `load_markdown_document`, `load_json_document`, `load_yaml_document`,
  `load_csv_document`, `load_documents_from_directory`,
  `load_from_directory`, and `ingest_from_backend` — so consumers
  reaching for any of the seven entry points get the same shape rather
  than being limited to a subset. The `tenant_id=` convenience kwarg
  folds into `extra_metadata` as `{"tenant_id": tenant_id}`; the same
  bound-tenant precedence applies (auto-derived wins on write boundary).
- `RAGKnowledgeBase._CHUNK_ID_PREFIX_KEYS` — ordered tuple of metadata
  keys (`tenant_id`, `domain_id`, `_generation` by default) folded into
  the chunk-id prefix in declared order, with `source_stem` always last.
  Subclasses rebind to add fold positions (e.g. `(... , "region", ...)`)
  without forking `_derive_chunk_prefix`. Single-tenant single-domain
  consumers (none of the declared keys present in metadata) see the
  historical `(source_stem, "_")` prefix unchanged; multi-segment
  prefixes use the `\x1f` record separator so snake_case-tag collisions
  are impossible.
- `RAGKnowledgeBase._RESERVED_METADATA_KEYS` — frozen set of identity
  tag names (`tenant_id` / `domain_id` / `_generation`) the KB owns at
  the write boundary. Caller-supplied `extra_metadata` entries with
  these keys are shadowed by the auto-derived value; non-reserved keys
  flow through unchanged. Documented in the multi-tenant USER_GUIDE
  section "Reserved vs. Consumer-Extensible Metadata Keys".
- `RAGKnowledgeBase` and `KnowledgeIngestionManager` advertise
  `Capability.TENANT_SCOPED_CHUNKS` via the `CapabilityContract`
  protocol (a chunk-layer Tenancy-family identifier added in the
  `dataknobs-common` `Capability` enum — see common's CHANGELOG).
  Consumers fail-fast at config-load time via
  `kb.supports(Capability.TENANT_SCOPED_CHUNKS)` or
  `mgr.supports("tenant_scoped_chunks")` rather than discovering
  chunk-id UPSERT collisions at first write. Advertisement is
  structural ("the class HAS the chunk-layer code path"), not
  activation-state — whether a specific instance is currently
  tenant-scoping is the natural binding check
  (`kb._tenant_id is not None`). The backend-state-layer
  (`TENANT_SCOPED_STATE`) and concurrency-layer
  (`TENANT_SCOPED_LOCKS`) are deliberately NOT advertised at the chunk
  layer; activation lives at the `KnowledgeResourceBackend` /
  `DistributedLock` layers respectively.
- `FileKnowledgeBackend`, `InMemoryKnowledgeBackend`, and
  `S3KnowledgeBackend` inherit `CapabilityMixin` (via the shared
  `KnowledgeResourceBackendMixin`) so the capability-contract surface
  is uniformly present. Each declares an empty `SUPPORTED_CAPABILITIES`
  set today — per-backend widening (e.g. `STREAMING_READS` on backends
  that implement `stream_file`, `CHANGE_SUBSCRIPTION` /
  `EVENT_BUS_EMISSION` / `KEY_PATTERN_FILTERING` once subscribe/emit
  surfaces ship, `TENANT_SCOPED_STATE` once `set_ingestion_status` /
  `get_checksum` / `has_changes_since` are tenant-scoped at the
  contract layer) is captured in the roadmap rather than declared
  speculatively. Adopters checking `backend.supports(...)` for any
  specific capability get the honest "not advertised" answer.

### Fixed

- **`SaveConfigTool` no longer blocks the event loop when persisting a
  configuration.** Finalizing the draft, creating the output directory,
  and writing the YAML config are blocking disk I/O; they are now offloaded
  to a worker thread via `asyncio.to_thread`, so saving a config from a
  wizard tool no longer stalls other concurrent conversations on a shared
  event loop. Behavior and return value are unchanged.
- **The knowledge-base wizard tools no longer block the event loop on disk
  I/O.** `CheckKnowledgeSourceTool` ran a directory `glob`/`stat` walk, and
  `AddKBResourceTool` / `IngestKnowledgeBaseTool` did `mkdir` + file writes,
  directly on the running loop; all three now offload that work to a worker
  thread via `asyncio.to_thread`. Tool results and behavior are unchanged.
- Cross-tenant `chunk_id` UPSERT collision in shared
  `RAGKnowledgeBase` instances under the same `domain_id`: two tenants
  ingesting the same `domain_id` through a shared KB previously
  produced N rows (the second ingest UPSERTed the first's chunks in
  place) instead of 2*N rows with disjoint chunk-id namespaces. The
  chunk-id derivation in `_embed_and_store_chunks` now folds every
  present, truthy key from `_CHUNK_ID_PREFIX_KEYS` into the chunk-id
  prefix — `tenant_id` is the first fold position by default, so a
  bound-tenant `KnowledgeIngestionManager` (or a caller threading
  `extra_metadata={"tenant_id": ...}` through `ingest_from_backend`)
  produces distinct chunk ids per tenant. Single-domain single-tenant
  consumers see no change: with `tenant_id` absent the loop yields the
  historical `[domain_id?, generation?, source_stem]` shape
  byte-for-byte.
- Cross-tenant filter-based deletion in `KnowledgeIngestionManager`
  under a shared destination: tenant B's CLEAR_FIRST re-ingest
  previously wiped tenant A's chunks under the same `domain_id`
  (filter was `{"domain_id": domain_id}` only). All filter-based
  mutations on the destination (CLEAR_FIRST, per-file purges,
  TOMBSTONE swap scope, rollback scope, and reconcile scope) now
  AND-compose the bound `tenant_id` via `_scope_for_tenant`, so a
  tenant-bound manager cannot accidentally delete or un-tombstone
  another tenant's rows. Unbound managers (single-tenant) are
  unaffected.

### Security

- **`SaveConfigTool` now rejects config names that could escape the output
  directory.** The config name flows from an LLM tool argument and
  user-driven wizard data, then becomes a `<name>.yaml` filename under the
  draft manager's output directory. A name containing a path separator or a
  bare parent-directory reference (e.g. `../escape`) is now rejected before
  the path is composed, and the composed path is re-checked to stay within
  the output directory (path-traversal prevention).
- **The knowledge-base wizard tools now reject paths that escape the
  configured knowledge directory.** `AddKBResourceTool` composes a write
  destination from the resource `path` (an LLM tool argument) and
  `domain_id`, and `IngestKnowledgeBaseTool` composes its manifest path from
  `domain_id` (user-driven wizard data); a `..` segment could previously
  write outside the knowledge directory. Both now resolve the composed path
  and reject it if it falls outside that directory, while still allowing a
  resource `path` to contain legitimate subdirectories (path-traversal
  prevention).
- Bumped minimum `starlette` requirement (extra: `server`, and the
  matching dev-dependency floor) from `>=1.0.1` to `>=1.3.1` to
  exclude GHSA-82w8-qh3p-5jfq (CVSS 7.5, `request.form()` silently
  ignores `max_fields` / `max_part_size` on
  `application/x-www-form-urlencoded` payloads, enabling a sub-10MB
  DoS), flagged at the floor resolve by the `dependency-update`
  workflow. The bump also sweeps GHSA-jp82-jpqv-5vv3 (CVSS 3.7,
  `request.url` userinfo poisoning on malformed paths missing a
  leading slash, fixed in 1.3.0), GHSA-wqp7-x3pw-xc5r (CVSS 7.5,
  Windows-only `StaticFiles` NTLM credential leak via UNC paths,
  fixed in 1.1.0), and GHSA-x746-7m8f-x49c (CVSS 5.3, `HTTPEndpoint`
  method dispatch using unvalidated client method names when a route
  is registered without an explicit `methods=` argument, fixed in
  1.1.0). The `fastapi>=0.133.0` floor (its `starlette>=0.40.0` no
  upper-cap constraint) permits the bumped floor unchanged.

## v0.7.3 - 2026-06-08

### Added

- `DynaBot.get_steps_of_type(step_cls)` — typed helper that returns
  every reasoning-strategy pipeline step that is an instance of
  `step_cls` as a `list[step_cls]`. Iterates
  `bot.reasoning_strategy.steps` when the strategy is pipeline-shaped;
  returns `[]` when the bot has no reasoning strategy or when the
  strategy has no `steps` attribute. Intended for post-construction
  injection of runtime collaborators that configuration cannot carry.
- `ReasoningStrategy.restore_from_checkpoint(manager, node_metadata)`
  — public hook called by `DynaBot.undo_last_turn` /
  `rewind_to_turn` so a strategy can reinstate per-state buckets it
  persists into a checkpoint node's metadata. Default no-op.
  `WizardReasoning` overrides to restore wizard FSM state from
  `node_metadata["wizard_fsm_state"]`.
- `ReasoningStrategy.undo_to_checkpoint(node_id)` — public hook
  called by `DynaBot.undo_last_turn` / `rewind_to_turn` so a strategy
  can revert node-keyed state. Default no-op. `WizardReasoning`
  overrides to undo each `MemoryBank` it owns.
- `DynaBot.from_config(config, *, reasoning_components=...)` —
  forwards a consumer-supplied mapping into the reasoning strategy's
  `StructuredConfigConsumer.components` channel at construction time.
  Strategies pick up the keys they read (e.g. `ReActReasoning` reads
  `extra_context`, `artifact_registry`, `review_executor`,
  `context_builder`, `prompt_refresher`); unknown keys are silently
  absorbed. Bot-managed components (`knowledge_base`,
  `prompt_resolver`, `prompt_envelope`) raise `ConfigurationError`
  on collision — use their respective config fields instead.

## v0.7.2 - 2026-06-06

### Added

- **`DynaBotConfig.prompt_envelope` selects the user-prompt and
  synthesis-system-prompt envelope style.** `"markdown"` (default)
  renders the auto-context user prompt as `## Knowledge base` /
  `## Conversation history` / `## Question` sections separated by
  `\n\n---\n\n`, and renders the grounded-reasoning synthesis system
  prompt's knowledge-base block as `## Knowledge base\n\n...`. `"xml"`
  reproduces the previous shape byte-for-byte (`<knowledge_base>` /
  `<conversation_history>` / `<question>` blocks separated by `\n\n`,
  and the legacy `<knowledge_base>...</knowledge_base>` synthesis-prompt
  block). `"prose"` renders bare `Label:\n\nbody` sections.
- New `dataknobs_bots.prompts.PromptEnvelope` and
  `PromptEnvelopeStyle` re-exports — a small typed helper used at
  every site that wraps a labeled context block, so the wrap style is
  chosen in one place and matches across the user prompt and the
  synthesis system prompt.
- `KnowledgeBase.format_context`, `RAGKnowledgeBase.format_context`,
  and `ContextFormatter.wrap_for_prompt` accept a keyword-only
  `envelope=` argument. When supplied, the wrapper renders in the
  envelope's style; when omitted, `wrap_in_tags=True` still produces
  the legacy `<knowledge_base>...</knowledge_base>` shape byte-for-byte
  so direct callers are unchanged.

### Changed

- The bot-assembled user prompt and the grounded-reasoning synthesis
  system prompt now default to markdown envelopes. Small
  instruction-tuned models can complete an XML-wrapped input shape by
  emitting a matching wrapper element around their reply (for example
  `<response>...</response>`); switching the default away from XML
  removes that mirroring cue. Model output bytes will shift on the
  next turn for consumers on the default. Pin
  `prompt_envelope: "xml"` to defer the change.
- The `grounded.synthesis.kb_wrapper` library prompt key is no longer
  registered. `GroundedReasoning.build_synthesis_system_prompt` now
  wraps the knowledge-base block through the bot-wide
  `PromptEnvelope`, so the wrap shape is selected by
  `DynaBotConfig.prompt_envelope` instead of by a separate library key.
  Consumers that overrode `grounded.synthesis.kb_wrapper` in a custom
  prompt library should switch to selecting the envelope style.
- **`context_transform` now receives the unwrapped KB body.** A
  consequence of moving the wrap decision into `PromptEnvelope`: the
  bot now asks the knowledge-base layer for an unwrapped body and
  hands that to `context_transform` *before* the envelope wraps it
  (in any style). Pre-fix, the transform saw
  `"<knowledge_base>\n...\n</knowledge_base>"` because the bot wrapped
  before transforming. Consumers whose `context_transform` callable
  pattern-matched on the XML wrappers (e.g. fenced or stripped them)
  must update their transform to operate on the bare body. Memory
  context (`conversation_history`) is unaffected — pre-fix already
  applied the transform to the unwrapped body before wrapping.
- **`prompt_envelope` validation is case-insensitive.** YAML configs
  written by humans now accept `"XML"`, `"Markdown"`, `"PROSE"`, etc.
  Values are normalized to lowercase on the frozen snapshot, so
  downstream lookups continue to match the lowercase enum values.
- **`DynaBot.HybridReasoning` now forwards `prompt_envelope` to its
  grounded child.** A hybrid-strategy bot configured with
  `prompt_envelope: "xml"` (or `"prose"`) had been silently rendering
  the synthesis-prompt KB block with the grounded child's default
  markdown envelope because hybrid did not propagate the collaborator.
  The envelope now reaches the grounded child unchanged.

### Fixed

- The pre-built `DynaBot(llm=provider, prompt_builder=..., ...)`
  constructor now accepts a `prompt_envelope` keyword. Programmatic
  construction (tests, `BotTestHarness`, advanced callers) can pin a
  non-default envelope without going through a config mapping; absent
  the keyword, the `DynaBotConfig` default `"markdown"` applies, so
  every existing call site is unchanged.
- **`ContextPersister.persist()` now correctly persists conversation
  context across the pre-/post-state boundary.** The previous
  implementation read `manager.metadata` (a read-only `@property`)
  and then assigned the mutated dict back to `manager.metadata` —
  which raised `AttributeError: property 'metadata' has no setter`
  on every call against a real `ConversationManager`. The call is
  now routed through `ConversationManager.update_seed_metadata`, so
  the context section is written to the live `state.metadata` (when
  state has been materialized) and to the initial-metadata seed
  bucket (always), with the same replace-not-merge semantic the
  original implementation intended. Behavioural tests against a real
  `ConversationManager` pin both the pre-state and post-state paths.
- **`DynaBot._execute_tools` now routes through
  `ToolRegistry.execute_tool` so the registry's execution tracker is
  populated on real bot turns.** Pre-fix, DynaBot called
  `tool.execute()` directly, bypassing the registry's recording code
  path. Consumers reading `tool_registry.get_execution_history()`
  always saw an empty list on a real turn — most notably
  `ContextBuilder._extract_tool_history`, which surfaces tool history
  into the prompt-rendered context section. The end-to-end chain
  (DynaBot turn → tool execution → tracker → context history) was
  broken at the first step. Dispatch now goes through
  `registry.execute_tool`, whose forwarding semantic was fixed in
  `dataknobs-llm` (`ContextAwareTool` receives `_context` per its
  docstring; plain tools are unaffected). DynaBot's per-tool timing,
  error handling, and `ToolExecution` records on `TurnState` are
  unchanged — the only behavioural shift is that a registry
  constructed with `track_executions=True` now sees a record per
  tool call during a real bot turn.

## v0.7.1 - 2026-06-02

### Added

- **`history_redactions` on every memory backend config — read-time
  citation redaction.** `BufferMemoryConfig`, `SummaryMemoryConfig`, and
  `VectorMemoryConfig` each carry a new
  `history_redactions: list[HistoryRedaction]` field (default empty —
  passthrough); each backend's `get_context()` rewrites assistant-role
  messages on the way out to the prompt-feed. `HistoryRedaction`
  (re-exported from `dataknobs_bots.memory`; canonical home in
  `dataknobs-llm`) is a `(pattern, replacement)` regex spec, applied in
  declared order — list the more specific pattern (a bracketed citation
  header) before the more general bare token. Stored state is never
  mutated: `BufferMemory.messages`, the `SummaryMemory` recent deque,
  and the vector-store rows keep the original text, so direct reads of
  the buffer, exports, and any UI that bypasses `get_context()` see
  un-redacted content. Backend-specific behavior:
  - `SummaryMemory` also applies the same redactions to overflow
    messages before they are summarized, so a citation token in an
    aged-out turn cannot survive in the system-role summary header.
  - `VectorMemory` applies redactions to search-result rows after the
    similarity search, so stored vectors and scoring are unaffected.
    `item["content"]` is the redacted view; `item["metadata"]` aliases
    the live stored row, so `item["metadata"]["content"]` still reads
    the un-redacted text — treat `metadata` as a read-only reference to
    the stored row.
  - `CompositeMemory` inherits the guarantee via delegation: each child
    configured with `history_redactions` redacts on its own path.
    `CompositeMemoryConfig` deliberately does not carry the field.
    Children configured with mismatched policies can land the same
    source message in two different `(role, content)` dedup buckets, so
    configure consistently across children that may surface the same
    content.
- **`DynaBotConfig.conversation_middleware`.** New optional list of
  `ConversationMiddleware` specs (same `{class, params, optional}` shape
  as `middleware`) forwarded to every `ConversationManager` the bot
  constructs. Distinct from `middleware` (bot-turn lifecycle hooks):
  `conversation_middleware` wraps the LLM-call boundary
  (`process_request` / `process_response`), so it can transform the
  request and response that hit the provider. `DynaBot.from_config(...)`
  accepts a symmetric `conversation_middleware=` kwarg that replaces the
  config-driven list with pre-built instances (matching the existing
  `middleware=` kwarg). Pairs with `HistoryRedactionMiddleware` from
  `dataknobs-llm` for deployments where the bot's memory is not the
  redaction surface.

### Changed

- **Middleware and tool specs are validated against their target
  interface at config-load.** A `middleware` spec whose resolved class
  does not subclass `Middleware`, a `conversation_middleware` spec whose
  resolved class does not subclass `ConversationMiddleware`, or a
  `tools` spec whose resolved class does not subclass `Tool` raises
  `ConfigurationError` before the spec's constructor is invoked, so a
  misplaced spec cannot trigger constructor side effects. `optional:
  true` continues to silence transient resolution failures (missing
  module / class, malformed params) but no longer silences a
  class-shape mismatch — a wrong-shape spec is a config-layout error,
  not a transient environment failure, and always raises. The tool
  resolver's wrong-shape error message changes accordingly (from
  `"Resolved class … is not a Tool instance"` to `"Resolved class …
  must subclass …Tool"`).

## v0.7.0 - 2026-05-26

### Added

- **Pluggable backend registries for memory, knowledge bases, and grounded
  sources.** Each subsystem's `create_*_from_config` factory now dispatches
  through a `PluginRegistry`, so 3rd parties can register custom backends
  without modifying core code:
  `register_memory_backend(name, factory)` (discriminator `type`, built-ins
  `buffer`/`vector`/`summary`/`composite`),
  `register_knowledge_base_backend(name, factory)` (discriminator `type`,
  built-in `rag`), and
  `register_source_backend(name, factory)`
  (matched against `GroundedSourceConfig.source_type`, built-ins
  `vector_kb`/`database`). Companion `list_*_backends()`,
  `is_*_backend_registered()`, and `get_*_backend_factory()` helpers are
  exported alongside each. All three `register_*_backend` functions are
  re-exported from the top-level `dataknobs_bots` namespace, mirroring
  `register_strategy`.
- **Typed subsystem sub-configs** — `BufferMemoryConfig`,
  `SummaryMemoryConfig`, `CompositeMemoryConfig`, and `VectorMemoryConfig`
  (exported from `dataknobs_bots.memory`) plus `RAGKnowledgeBaseConfig`
  (exported from `dataknobs_bots.knowledge`), all `StructuredConfig`
  subclasses. Each concrete subsystem class consumes the matching typed
  config (see the corresponding *Changed* entry).
- **`from_components(...)` on the subsystem classes** — `VectorMemory`,
  `SummaryMemory`, `CompositeMemory`, and `RAGKnowledgeBase` expose
  `from_components(config=None, **collaborators)` for assembling an
  instance from already-built collaborators (a pre-built vector store and
  embedder, an LLM provider, child memory strategies) instead of from
  config. The collaborator-adopting path does not own the resources it is
  handed (`close()` leaves caller-owned resources open).
- **Typed `DynaBotConfig`** (`StructuredConfig` subclass,
  `dataknobs_bots.bot.config`). A `DynaBot` now carries a typed
  `bot.config: DynaBotConfig` snapshot — a thin top-level envelope of typed
  scalars plus the documented config sections. The polymorphic subsystem
  sections (`memory`, `knowledge_base`, `reasoning`) and the provider
  section (`llm`) stay raw mappings, dispatched by their discriminator in
  the subsystem registries. `DynaBot.from_config()` accepts either a config
  mapping or a `DynaBotConfig`.
- **`DynaBot.from_components(...)`**, the named alias of the pre-built
  collaborator constructor (`DynaBot(llm=provider, prompt_builder=...,
  conversation_storage=...)`), for assembling a bot from already-built
  collaborators.
- **Embedder credentials are redacted from config `repr`.** Via the
  `StructuredConfig._SENSITIVE_FIELDS` mechanism in `dataknobs-common`,
  `VectorMemoryConfig.api_key` and `RAGKnowledgeBaseConfig.api_key` are
  masked as `'***'` in `repr(config)` (and therefore in logs, tracebacks,
  and pytest failure output). `to_dict()` is never redacted, so
  round-trip construction is unaffected. A credential nested inside a raw
  mapping section — the `embedding` dict's `api_key`, the `vector_store`
  dict's `connection_string`, an `llm` dict's `api_key` (including in
  `SummaryMemoryConfig`, which declares no `_SENSITIVE_FIELDS`) — is also
  masked: common's repr now descends into raw `Mapping`/`list` fields and
  masks interior keys in its default sensitive-key set ∪ the class's
  `_SENSITIVE_FIELDS`.

### Changed

- **`GroundedSourceConfig` is now a frozen `StructuredConfig`.** Its
  `from_dict` is derived from the dataclass fields with a `_normalize_dict`
  hook that preserves the legacy flat declaration shape (the `type` key
  aliases `source_type`; keys outside the reserved set collect into
  `options`). Public behaviour is unchanged; it gains `to_dict` and
  symmetric round-tripping and is immutable (construct a modified copy with
  `dataclasses.replace(...)`).
- **The memory, knowledge-base, and grounded-source factories dispatch via
  their backend registries** instead of inline type branching. Public
  signatures (`create_memory_from_config`,
  `create_knowledge_base_from_config`, `create_source_from_config`) and the
  `ValueError` raised on an unknown type are unchanged.
- **The concrete memory and knowledge subsystem classes are now
  `StructuredConfigConsumer`s.** `BufferMemory`, `VectorMemory`,
  `SummaryMemory`, `CompositeMemory`, and `RAGKnowledgeBase` carry a typed
  `self.config` and build through the shared construction lifecycle, and
  the memory registry registers the classes directly (the transitional
  per-backend builder functions are gone). Config-driven construction is
  unchanged: `await create_memory_from_config({...})`,
  `await create_knowledge_base_from_config({...})`,
  `await VectorMemory.from_config({...})`, and
  `await RAGKnowledgeBase.from_config({...})` keep their exact signatures
  and behavior (the async warmup classes expose `from_config` as a
  lifecycle-faithful async delegator that runs `_ainit`), and
  `BufferMemory(max_messages=...)` still works. Direct construction from
  pre-built collaborators moves from positional/keyword constructors to
  `from_components(...)` (see the *Added* entry); e.g.
  `SummaryMemory(llm_provider=p, recent_window=2)` becomes
  `SummaryMemory.from_components({"recent_window": 2}, llm_provider=p)`,
  `CompositeMemory([m1, m2])` becomes
  `CompositeMemory.from_components(strategies=[m1, m2])`, and
  `RAGKnowledgeBase(vector_store=vs, embedding_provider=ep,
  chunking_config={...})` becomes
  `RAGKnowledgeBase.from_components({"chunking": {...}}, vector_store=vs,
  embedding_provider=ep)` (the `chunking_config` / `merger_config` /
  `formatter_config` constructor arguments are now the `chunking` /
  `merger` / `formatter` config keys, with pre-built `chunker` /
  `merger_config` / `formatter_config` accepted as `from_components`
  collaborators). The
  `owns_llm_provider` constructor flag is gone: ownership now follows the
  construction path (a dedicated `llm` config section is owned; an injected
  provider is not). Typing/contract refactor only — no runtime, retrieval,
  or ownership-semantics behavior changes.
- **`DynaBot` is now a `StructuredConfigConsumer` and builds through the
  shared async construction lifecycle** (`from_config` → `from_config_async`
  → `__init__` → `_setup` → `_ainit`). `DynaBot.from_config(config, *,
  llm=None, middleware=None)` keeps its exact signature and behavior (now a
  parity-guarded async delegator); the direct constructor
  `DynaBot(llm=provider, prompt_builder=..., conversation_storage=..., ...)`
  is preserved verbatim as the pre-built collaborator shape. Construction is
  a typing/contract refactor only — no runtime, reasoning, or dispatch
  behavior changes.
- **`config/` subcomponent dataclasses now subclass `StructuredConfig`.**
  `ToolEntry`, `TemplateVariable`, `ConfigTemplate`, `ConfigVersion`,
  `DraftMetadata`, `ComponentSchema`, and the wizard-builder configs
  (`TransitionConfig`, `IntentDetectionConfig`, `ContextGenerationConfig`,
  `StageConfig`, `WizardConfig`) inherit
  `dataknobs_common.structured_config.StructuredConfig`, so `from_dict` is
  derived from the dataclass fields (recursing into nested sub-configs —
  `StageConfig`'s transitions/intent/context and `WizardConfig`'s stages
  are rebuilt automatically, replacing the hand-walked deserialization).
  `ComponentSchema` and `WizardConfig` gain a `from_dict` they previously
  lacked. Serialized output is unchanged: classes whose `to_dict` omits
  defaults or renames keys (e.g. `DraftMetadata`'s `id`) keep that
  bespoke `to_dict`, and frozenset/tuple fields are restored on load via
  `__post_init__`.
- **`TemplateVariable`, `ConfigTemplate`, `ConfigVersion`, `DraftMetadata`,
  and `ComponentSchema` are now frozen** (the wizard-builder configs
  already were). They are immutable-by-design value objects; construct a
  modified copy with `dataclasses.replace(...)` instead of assigning to
  attributes.

### Security

- Bumped minimum `starlette` requirement (extra: `server`, and the
  matching dev dependency) from `>=0.49.1` to `>=1.0.1` to exclude
  PYSEC-2026-161 / GHSA-86qp-5c8j-p5mr — missing Host-header
  validation that poisons `request.url.path` and can bypass
  path-based authentication. Flagged at the floor resolve by the
  `dependency-update` workflow. Because `1.0.1` is a major release and
  `fastapi <0.133.0` capped `starlette<1.0.0`, the coupled `fastapi`
  floor was bumped from `>=0.120.1` to `>=0.133.0` (the lowest fastapi
  whose starlette constraint permits 1.x) in both the `[server]` extra
  and the dev group. `registry.server` uses only FastAPI's own API
  surface and never imports `starlette` directly, so the major bump is
  insulated. The new floor preserves the prior sweep of
  GHSA-7f5h-v6xp-fcq8 (CVSS 7.5) and GHSA-2c2j-9gv5-cj73 (CVSS 5.3,
  0.49.1).

## v0.6.22 - 2026-05-19

### Added

- **`IngestOrchestrator(manager_resolver=...)` + `IngestionManagerResolver`.**
  An injectable async resolver seam for multi-tenant deployments:
  the orchestrator calls `manager_resolver(tenant_id=..., domain_id=...)`
  once per trigger event (tenant/domain parsed from the payload) and
  dispatches to the returned per-tenant `KnowledgeIngestionManager`
  (its own KB backend prefix / `vector_partition` / embedder).
  `ingestion_manager=` and `manager_resolver=` are mutually exclusive
  and exactly one is required (`ValueError` otherwise); the static
  single-`ingestion_manager` path is unchanged. `IngestionManagerResolver`
  is a `@runtime_checkable` `Protocol` exported from
  `dataknobs_bots.knowledge`. The trigger payload gains an optional
  `tenant_id` (absent ⇒ `None` passed to the resolver); a present
  non-string `tenant_id` fails closed (logged + trigger skipped)
  rather than being routed or coerced, since a misidentified tenant
  is a cross-tenant data leak.

### Changed

- **`IngestOrchestrator` per-domain lock key is now tenant-scoped:**
  `f"ingest:{tenant_id or '-'}:{domain_id}"` (was `f"ingest:{domain_id}"`)
  so two tenants sharing a `domain_id` do not false-share one lock
  under a cross-replica backend. Single-tenant triggers (no `tenant_id`)
  degrade to the stable key `ingest:-:<domain_id>`. With the default
  process-local `InProcessLock` this is invisible. Deployments using a
  cross-replica lock (`{"backend": "postgres", ...}`) should note that
  during a rolling upgrade old and new replicas briefly compute
  different keys for the same single-tenant domain, momentarily
  relaxing cross-replica serialization for that domain until all
  replicas are upgraded.

## v0.6.21 - 2026-05-18

### Added

- **`KnowledgeResourceBackend.list_changes_since(domain_id, version)
  -> ChangeSet`** — file-level diff (added / modified / deleted +
  the current canonical version) between the current knowledge base
  and the snapshot identified by `version` (a `get_checksum()`
  value). `has_changes_since` is now its degenerate case
  (`not (await list_changes_since(...)).is_empty`) rather than a
  separately-implemented sibling.
- **`ChangeSet`** (frozen dataclass: `added` / `modified` /
  `deleted` / `version`, with `is_empty`) and
  **`InvalidVersionError`** (raised when a version predates a
  backend's snapshot retention; consumers fall back to a full
  re-ingest) — exported from `dataknobs_bots.knowledge` and
  `dataknobs_bots.knowledge.storage`.
- **`KnowledgeResourceBackendMixin`** — the shared canonical
  change-detection algorithm (`get_checksum` / `has_changes_since`
  / `list_changes_since` over `list_files()` plus a `_load_snapshot`
  seam). All in-tree backends inherit it; out-of-tree backends mix
  it in for correct behaviour for free. All three in-tree backends
  retain per-version snapshots so `list_changes_since` is a minimal
  file-level diff: `InMemoryKnowledgeBackend` (in-process map),
  `FileKnowledgeBackend` (`_snapshots/<version>.json` written after
  every mutation), and `S3KnowledgeBackend` (snapshot objects, or the
  metadata object's own S3 version history — see
  `change_detection_mode` below). An out-of-tree backend that does
  not override `_load_snapshot` still gets correct (full, non-minimal)
  change *detection* via the version-equality short-circuit.
- **`S3KnowledgeBackend(change_detection_mode=...)`** (also via
  `from_config`, default `"snapshot"`) selects how per-version
  snapshots are resolved: `"snapshot"` writes a small
  `{path: checksum}` object under `{domain}/_snapshots/<version>.json`
  after every mutation (self-contained, any bucket); `"s3_versioning"`
  writes no extra objects and instead walks the metadata object's own
  S3 version history (`ListObjectVersions`) — requires bucket
  versioning enabled, and with it disabled a stale version safely
  falls back to a full re-ingest. An unrecognized mode raises
  `ValueError` (fail closed).
- **`IngestOrchestrator` trigger-payload dispatch.** The trigger
  event payload now selects the ingest entry point: `since_version`
  → `ingest_changes` (per-file delta), `force_full` →
  `ingest(swap_mode=CLEAR_FIRST)` (full re-ingest), otherwise the
  unchanged `ingest_if_changed(last_version)` default. `since_version`
  takes precedence over `force_full`. Payloads using only
  `domain_id` / `last_version` are byte-for-byte unchanged.
- **`IngestionStatus.SWAPPING`** — set by the `TOMBSTONE` swap path
  while the new generation is written; a crash here leaves the
  domain in this state with the in-flight token recoverable.
- **Interrupted-swap auto-reconciliation + `KnowledgeIngestionManager.
  reconcile(domain_id) -> bool`.** A process crash between the upsert
  and the commit of a `TOMBSTONE` swap leaves the domain in
  `SWAPPING` with the old generation tombstoned-but-intact and orphan
  new-generation chunks possibly present. The next `ingest()` /
  `ingest_changes()` for that domain now reconciles *before* applying
  anything — restoring the previous generation to visibility and
  dropping exactly the crashed swap's orphans by its persisted
  token — so residue never accumulates and unrelated files are never
  left hidden. `reconcile()` exposes the same recovery as an
  idempotent one-shot for domains that will not be re-ingested soon
  (returns `True` if it reconciled, `False` if there was nothing to
  do). Backed by a new `KnowledgeBaseInfo.generation: str | None`
  field (round-trips through `to_dict`/`from_dict`) and a kw-only
  `generation=` parameter on `KnowledgeResourceBackend.
  set_ingestion_status` (always written through, so any non-SWAPPING
  transition clears a stale token); implemented by the in-memory,
  file, and S3 backends.
- **`KnowledgeIngestionManager.ingest_changes(domain_id,
  since_version, *, progress_callback=None, config=None)`** —
  per-file delta re-ingest. Diffs the source against
  `since_version` (a `get_checksum`/`get_current_version` value),
  purges chunks for deleted *and* modified files, then re-embeds
  only the added/modified files through the same internal apply
  path as a full `ingest()` — so swap semantics cannot diverge
  between the full-domain and per-file routes. An S3 `PutObject`
  on one file in a 100-file corpus now re-embeds one file, not
  the whole corpus. If `since_version` predates the backend's
  snapshot retention (`InvalidVersionError`) it falls back to a
  full re-ingest after a warning — never a silent skip.
- **`IngestionResult.files_deleted`** — count of source files
  whose chunks were removed because the file no longer exists at
  the source (populated by `ingest_changes`; `0` for a full
  `ingest`). Included in `to_dict()` and the `knowledge:ingestion`
  event payload.
- **`RAGKnowledgeBase.ingest_from_backend(file_filter=)`** —
  optional keyword-only `Callable[[KnowledgeFile], bool]`
  predicate, evaluated after the pattern match, restricting
  enumeration to a subset of the backend's files. `None`
  (default) is unchanged behavior. This is the seam
  `ingest_changes` uses to re-embed only the changed files
  through the full pattern/chunking pipeline.
- **`IngestSwapMode`** (`CLEAR_FIRST` / `APPEND` / `TOMBSTONE`)
  plus a keyword-only `swap_mode=` on
  `KnowledgeIngestionManager.ingest()` and `ingest_changes()`
  (exported as `dataknobs_bots.knowledge.IngestSwapMode`).
  `TOMBSTONE` is a crash-safe re-ingest: the existing (scoped)
  chunks are marked `_stale` (hidden from reads), the new
  generation is ingested under distinct generation-keyed chunk
  ids so it never overwrites the old rows, and the old
  generation is physically retired **only on a clean commit** —
  on a raised error or partial-error ingest the rollback drops
  the new generation by its token and restores the old one. The
  old generation is never overwritten or deleted before the new
  one commits, so a crash, a raised error, or a racing
  same-domain re-ingest always leaves a fully restorable
  previous generation (unlike the `CLEAR_FIRST`
  delete-then-insert). A crash mid-swap leaves the domain in
  `IngestionStatus.SWAPPING`, auto-reconciled by the next ingest
  (or `KnowledgeIngestionManager.reconcile`). Honored identically
  by all in-tree vector stores (Memory, FAISS, PgVector, Chroma);
  `ingest_changes(swap_mode=TOMBSTONE)` scopes the swap to
  exactly the changed/deleted files. A transient in-swap read
  window remains (closing it needs a generation pointer-flip,
  a future mode).
- **`RAGKnowledgeBase.query(..., include_stale=False)`** and
  **`hybrid_query(..., include_stale=False)`** — a single shared
  read chokepoint hides chunks tombstoned by an in-progress
  `TOMBSTONE` swap on **both** read paths (vector search and
  hybrid, native and client-side fusion); `include_stale=True`
  returns them. `service.py` / retrieval inherit this through
  `query` / `hybrid_query`.
- **`RAGKnowledgeBase.update_metadata_where(filter, set_)`** —
  delegates to the vector store's filter-keyed bulk metadata
  merge; the destination-side primitive the `TOMBSTONE` swap
  uses to mark (and, on rollback, un-mark) a generation without
  enumerating ids.
- **Optional embedder rate-limit seam.**
  `KnowledgeIngestionManager(__init__, rate_limiter=)` and the
  keyword-only `RAGKnowledgeBase.ingest_from_backend(...,
  rate_limiter=)` accept a
  `dataknobs_common.ratelimit.RateLimiter`. When set, every
  per-chunk embed on the ingest path is preceded by
  `await rate_limiter.acquire("embed")`, so a rate-limited
  embedding provider (e.g. a hosted API) cannot fail a whole
  ingest under burst. The manager threads its `rate_limiter`
  through to the embed core for every swap mode. `None` (the
  default) is byte-for-byte the prior behaviour — no pacing,
  correct for a local Ollama embedder.

### Changed

- **`KnowledgeBaseInfo.version`** is now documented as a
  cache-invalidation / display counter only and is **no longer the
  change-detection key** (it is still incremented on every change).
  Change detection uses the canonical content snapshot
  (`get_checksum`). **`KnowledgeIngestionManager.get_current_version()`**
  consequently returns the canonical snapshot identity (a
  `get_checksum` value), not the monotonic counter — so capturing
  it and passing it back to `ingest_if_changed(last_version=...)`
  is now a correct round-trip.
- **`IngestOrchestrator(__init__)`** accepts a new optional
  `lock: DistributedLock | None = None` parameter **and** a
  configuration-driven `lock_config: dict | None = None`
  alternative. Per-domain serialization of ingest triggers is
  backed by a `dataknobs_common.locks.DistributedLock` (keyed
  `ingest:<domain_id>`) instead of an internal `asyncio.Lock`.
  Supply a pre-built lock via `lock=`, or let the orchestrator
  resolve one through the shared `create_lock` factory by passing
  `lock_config={"backend": "postgres", ...}` — so a multi-replica
  deployment selects a cross-replica backend by configuration
  without writing code (no lock logic lives in `dataknobs-bots`).
  The two are mutually exclusive (passing both raises
  `ValueError`); an unknown `lock_config` backend raises
  `ValueError` (fail closed). The default — neither supplied — is
  `InProcessLock()`, process-local and behaviour-identical to
  prior releases for single-replica deployments. Multi-replica
  deployments must configure a cross-replica lock; a process-local
  lock cannot serialize across replicas. The built-in Postgres
  advisory-lock backend (`lock_config={"backend": "postgres", ...}`)
  provides cross-replica serialization out of the box; other backends
  remain registry-pluggable via
  `dataknobs_common.locks.lock_backends`.
- **`KnowledgeResourceBackend.set_ingestion_status`** accepts
  `IngestionStatus | str` (Protocol + memory / file / S3
  backends). The typed enum is the preferred form; legacy
  string values still work and are normalized internally. An
  unrecognized status string now raises
  `dataknobs_common.exceptions.ValidationError` (was a bare
  `ValueError`) — the message enumerates the accepted values, and
  the type is a `DataknobsError`, **not** a `ValueError` subclass,
  so a bare `except ValueError` no longer silently swallows an
  invalid-status bug. Domain-not-found still raises `ValueError`.
  No in-tree caller catches `ValueError` around status
  normalization, so this is contract-tightening only.
- **`RAGKnowledgeBase.count()` excludes tombstoned chunks by
  default.** A mid-`TOMBSTONE`-swap `count(filter)` previously
  delegated straight to the store and reported old+new (≈double)
  while `query()`/`hybrid_query()` only returned the new
  generation. `count()` now returns the read-visible count
  (`count(filter) − count(filter ∧ _stale=True)`, two store-agnostic
  counts); the new kw-only `include_stale=True` restores the prior
  single delegated count (every stored chunk). The numbers differ
  **only** while a swap is in flight; outside a swap there are no
  `_stale` chunks and the result is unchanged.

### Deprecated

- **`KnowledgeIngestionManager.ingest(clear_existing=)`** — pass
  `swap_mode=` (`IngestSwapMode`) instead. `clear_existing=True`
  maps to `CLEAR_FIRST`, `False` to `APPEND`; passing the
  argument emits a `DeprecationWarning`. With neither argument
  set the default is unchanged (`CLEAR_FIRST`), so existing
  callers that omit it are unaffected.

### Fixed

- **`get_checksum()` → `has_changes_since()` round-trip no longer
  spuriously re-ingests.** `has_changes_since` (and so
  `KnowledgeIngestionManager.ingest_if_changed`) previously compared
  the monotonic `KnowledgeBaseInfo.version` counter while
  `get_checksum()` returned a content-snapshot hash — different
  value spaces, so a consumer pairing the two (the intuitive,
  documented usage) always saw "changed" and re-ingested the entire
  domain on every check. Both now derive from the canonical content
  snapshot, so an unchanged knowledge base correctly reports no
  changes across all in-tree backends (memory / file / S3).
- **`IngestOrchestrator` multi-replica race made honest.** The
  previous `asyncio.Lock`-per-domain provided no protection across
  processes, yet the class docstring implied per-domain
  serialization unconditionally. The docstring now states the
  serialization scope is exactly the scope of the injected lock and
  that multi-replica deployments must inject a cross-replica lock.
- **`IngestOrchestrator` per-domain lock-map leak.** The internal
  `dict[str, asyncio.Lock]` was never evicted, so every distinct
  `domain_id` grew it unbounded for the lifetime of the
  orchestrator. The injected `InProcessLock` reference-count evicts
  its key map, closing the leak.
- **`IngestSwapMode.TOMBSTONE` re-ingest is now genuinely
  crash-safe.** Chunk ids were deterministic, so a re-embedded
  file's new chunks upserted *over* the tombstoned old rows in
  place — clearing their `_stale` mark and destroying the old
  generation the instant the new one was written. TOMBSTONE was a
  no-op for the dominant re-ingest case (any file whose content
  changed), a mid-swap crash or partial-error left freshly written
  chunks live with no `_stale` key (leaked partial generation), and
  an `ingest_changes` rollback un-tombstoned the *whole* swap scope
  — resurrecting files that had been deleted at the source. Each
  swap now mints a `uuid4` generation token folded into the new
  chunks' ids and stamped on their metadata (`_generation`), so the
  two generations coexist physically until a clean commit. Rollback
  (raised failure *or* partial error) drops exactly the new
  generation by its token, restores the modified files' old
  generation to visibility, and unconditionally purges files
  deleted at the source (never resurrected). On a clean commit the
  old generation is physically retired. APPEND / CLEAR_FIRST id
  derivation is byte-for-byte unchanged (the token is opt-in by
  presence), so single-domain consumers and existing populated
  stores are unaffected.
- **Native hybrid fusion no longer under-returns mid-swap.**
  `hybrid_query(fusion_strategy="native")` requested exactly `k`
  from the store's `hybrid_search` and *then* dropped tombstoned
  rows, so when `_stale` chunks ranked in the top `k` it returned
  fewer than `k` visible results during a `TOMBSTONE` swap. Both the
  vector and native-hybrid read paths now share a single
  `_fetch_drop_stale_truncate` helper that over-fetches
  `k * _STALE_OVERFETCH` before the stale gate and truncates to `k`,
  so a swap in progress no longer shrinks native-fusion result
  sets. (`_is_stale`'s `None`-guard was also tightened from a
  truthiness check to an explicit `is not None` — same result for
  every real input, but it correctly documents that the guard
  protects against a metadata-less row, not an empty dict.)

### Security

- Bumped the `[server]` extra's and dev group's minimum `fastapi`
  requirement from `>=0.110.0` to `>=0.120.1` and added an explicit
  `starlette>=0.49.1` floor to exclude starlette versions affected
  by GHSA-7f5h-v6xp-fcq8 (CVSS 7.5) and GHSA-2c2j-9gv5-cj73
  (CVSS 5.3), both fixed in starlette 0.49.1. `starlette` reaches
  the package only transitively via `fastapi`, but `fastapi`'s own
  lower bound on `starlette` never rises to the patched version, so
  an explicit floor is required to guarantee a safe `starlette` for
  all resolvers — not only the floor-resolve audit. The `fastapi`
  bump is required for graph satisfiability: `fastapi 0.110.0`
  capped `starlette<0.37.0`, and `0.120.1` is the lowest `fastapi`
  whose constraint (`starlette<0.50.0,>=0.40.0`) permits 0.49.1.
  Surfaced by the floor-resolve OSV audit in the
  `dependency-update` workflow.

## v0.6.20 - 2026-05-13

### Added

- **`Registration.metadata`** — `dict[str, Any]` field on
  `dataknobs_bots.registry.Registration` for cross-cutting context
  (`tenant_id`, audit info, feature flags) that lands in the storage
  backend's ``metadata`` column rather than mixed into the config
  payload.  Round-trips through `to_dict` / `from_dict` and the HTTP
  wire protocol.

- **`RegistryBackend.register(..., metadata=...)`** — kw-only
  parameter routes caller-supplied metadata to the backend's
  metadata channel.  Implemented by `InMemoryBackend`,
  `DataKnobsRegistryAdapter`, and `HTTPRegistryBackend`.

- **Registry filter / pagination surface** on `RegistryBackend`:
  - `list_all(*, status=None, filter_metadata=None, sort=None,
    limit=None, offset=None)` — list with optional status equality,
    equality filter over the metadata column, sort spec, and
    limit/offset pagination.
  - `list_active(...)` / `list_inactive(...)` — symmetric
    convenience wrappers over `list_all` with the status pinned.
  - `count_all(*, status=None, filter_metadata=None)` — routed
    through `AsyncDatabase.count(query)` so backends with pushdown
    counts (`SELECT COUNT(*) WHERE ...`) benefit transparently.
  - `count(*, filter_metadata=None)` / `count_inactive(...)` —
    pinned-status counterparts.
  - `stream(*, status=None, filter_metadata=None, config=None)` —
    async-iterator surface for large tenant populations, yields
    `Registration` instances one at a time.

- **`BotRegistry` surfaces the new metadata / filter / pagination
  surface** so consumers don't drop to ``registry._backend``:
  - ``register(..., metadata=...)`` threads ``metadata`` to the
    backend's metadata channel.
  - ``list_bots(*, filter_metadata=None, sort=None, limit=None,
    offset=None)`` — no-kwarg form returns active bot IDs as
    before; any kwarg routes through ``list_active`` for pushdown
    filtering.
  - ``list_registrations(*, status=None, filter_metadata=None,
    sort=None, limit=None, offset=None)`` — new method surfacing
    full `Registration` objects (timestamps / status / metadata).
  - ``count(*, filter_metadata=None)`` — tenant-scoped counts.

- **`HTTPRegistryBackend` wire-protocol extensions** — optional
  query parameters on `GET /configs`:
  `?filter_metadata=<URL-encoded JSON object>` (sorted keys for
  deterministic cache lines), `?status=<value>`,
  `?sort=<field>[:asc|desc]` (repeatable; wire order is tie-break
  order), `?limit=<int>`, `?offset=<int>`.  Schema is **additive
  optional**: servers that recognize a parameter honor it; servers
  that don't ignore it and return the broader list.  The client
  defensively re-applies idempotent filters (`filter_metadata`,
  `status`, `sort`) after parsing the response; `limit`/`offset`
  are intentionally NOT re-applied client-side (re-offsetting a
  server-paginated window would drop live rows).

- **`POST /configs/{bot_id}/deactivate`** — new server-side
  endpoint that routes directly to ``RegistryBackend.deactivate``.
  Lets HTTP clients soft-delete without first issuing
  ``GET /configs/{bot_id}`` (which bumps ``last_accessed_at``).
  Returns ``204 No Content`` on success or ``404 Not Found``.

- **`create_registry_router(backend)`** — reference FastAPI router
  in `dataknobs_bots.registry.server` exposing `RegistryBackend` as
  the wire protocol that `HTTPRegistryBackend` speaks.  Consumers
  can stand up a config service backed by any `RegistryBackend`
  (`InMemoryBackend`, `DataKnobsRegistryAdapter` over
  Postgres/SQLite/S3, …) with one line of glue.  FastAPI is an
  optional dependency: importing the module without it installed
  succeeds; calling `create_registry_router` raises `ImportError`
  with an install hint (`pip install 'dataknobs-bots[server]'`).
  Protocol is pinned on both sides by client and server test
  suites — drift breaks both.

- **`ArtifactRegistry.query`** — kw-only `filter_metadata=`,
  `sort=`, `limit=`, `offset=` parameters.  Filter / sort push down
  to the database query so SQL backends can use indexes.  Pagination
  is applied **after** the latest-pointer dedup pass (dual-write
  storage shape — pre-dedup row count diverges from post-dedup
  artifact count, so a pushdown ``LIMIT`` is unsafe).  Existing
  positional parameters (`artifact_type`, `status`, `tags`,
  `filters`) unchanged.

- **`ArtifactRegistry.count`** — new method mirroring `query`
  parameter-for-parameter (minus sort/limit/offset).  Equivalent
  to ``len(await registry.query(...))`` after dedup.

- **`RubricRegistry.list_all` / `RubricRegistry.get_for_target`** —
  kw-only `filter_metadata=`, `sort=`, `limit=`, `offset=`.  Same
  post-dedup pagination policy as `ArtifactRegistry.query` (same
  dual-write storage shape).

- **`RubricRegistry.count_for_target` / `RubricRegistry.count_all`**
  — new count methods mirroring the corresponding list/get methods.

- **`GeneratorRegistry.list_definitions`** — kw-only
  `filter_metadata=`, `sort=`, `limit=`, `offset=`.  Unlike the
  dual-write registries, `GeneratorRegistry` writes a single row
  per generator id — no pointer/snapshot divergence — so
  limit/offset push down to the database directly.

- **`GeneratorRegistry.count_definitions`** — new method that routes
  through `AsyncKeyedRecordStore.count`, letting backends with
  pushdown counts skip row materialization.

### Changed

- **`DataKnobsRegistryAdapter`, `ArtifactRegistry`, `RubricRegistry`,
  and `GeneratorRegistry` now compose `AsyncKeyedRecordStore`** (from
  `dataknobs-data`) instead of building `Record(...)` instances
  inline.  The store's
  ``(T) -> (data, metadata)`` serializer signature makes the
  metadata channel part of the function's type, so a future change
  to a model can't accidentally drop the metadata channel without a
  type-visible diff at the serializer site.  Public surface
  preserved; the `DataKnobsRegistryAdapter` stored shape differs —
  see Migration below.

### Fixed

- **`DataKnobsRegistryAdapter` now persists caller-provided
  metadata to the `Record.metadata` column.**  Previously the
  metadata column was always empty (there was no
  `Registration.metadata` field), rendering `metadata.X` filters
  and the Postgres metadata GIN index unreachable.  Multi-tenant
  consumers can now use `filter_metadata={"tenant_id": ...}` to
  scope `list_active` / `list_all` queries.

- **`ArtifactRegistry` and `RubricRegistry` now persist artifact /
  rubric `metadata` to the `Record.metadata` column** (latent
  defect — no consumer had hit it yet).

- **`GeneratorRegistry` no longer silently routes definition
  fields into the `data` column under a `metadata` variable
  name.**  The pre-fix code passed a local variable named
  ``metadata`` positionally to ``Record(...)``, but ``Record(...)``'s
  first positional is ``data`` — so the schema/version/id fields
  landed in the data column and the record's metadata column was
  never populated.  Migrating to `AsyncKeyedRecordStore` removes
  the inline `Record(...)` call, so the variable-name shadow
  cannot recur and `GeneratorDefinition.metadata` lands in the
  correct column.

- **`DataKnobsRegistryAdapter.count()` no longer materializes
  every active row** to compute its result.  It now routes through
  `_db.count(query)`, so backends with `SELECT COUNT(*)` pushdown
  return without row materialization.

- **`HTTPRegistryBackend.register` and `.deactivate` no longer
  issue touching reads.**  Previously both methods called
  ``await self.get(bot_id)`` first — the corresponding
  ``GET /configs/{bot_id}`` route bumps ``last_accessed_at`` per
  the `get` protocol contract, so every re-register and every
  soft-delete contaminated the user-activity signal that timestamp
  is supposed to carry.  `register` now issues a single
  ``PUT /configs/{bot_id}`` (upsert); `deactivate` calls the new
  ``POST /configs/{bot_id}/deactivate`` endpoint.

- **`ArtifactRegistry.revise` / `set_status` / `submit_for_review`
  are now serialized per artifact id**, closing an in-process
  read-modify-write race.  Two concurrent ``revise(id, …)`` callers
  could both read ``v1.0.0``, both compute ``v1.0.1``, and both
  write the same snapshot key — last-write wins and the losing
  revision silently disappeared.  A per-id ``asyncio.Lock`` now
  wraps each read-modify-write flow.  **Scope:** in-process only.
  Two processes writing to the same backing database still race;
  the multi-process fix (optimistic-version / row-lock check at
  the database layer) is tracked as a separate work item.

- Bumped minimum `pyyaml` requirement from `>=6.0` to `>=6.0.2` to
  exclude versions that lack cp312/cp313 wheels and fail to build
  from source against modern Cython.  Surfaced by the floor
  resolve step in the `dependency-update` workflow.

### Migration

- **Stored record shape for `DataKnobsRegistryAdapter` changed.**
  Pre-migration, every field of the `Registration` was written into
  the ``data`` column and the record's metadata column was always
  empty (there was no ``Registration.metadata`` field).
  Post-migration, `Registration.metadata` is written to the
  record's ``metadata`` column.  Existing deployments must rewrite
  stored rows once before the new `filter_metadata=` / metadata
  pushdown will see anything (the column is empty on pre-migration
  rows).

- **Wire-protocol change is additive.** `Registration.to_dict()`
  / `from_dict()` gained a ``metadata`` key.  Old clients that
  ignore unknown keys keep working against new servers; old
  servers that omit the key produce ``metadata={}`` on the new
  client via ``data.get("metadata") or {}``.  No coordinated
  upgrade is required, but until both sides understand the key,
  the metadata channel is effectively absent on that consumer.

- **New `ArtifactRegistry.query` parameters (`filter_metadata=`,
  `sort=`, `limit=`, `offset=`) are kw-only.**  This is the
  contract for the new surface; positional usage of the
  established parameters (`artifact_type`, `status`, `tags`,
  `filters`) is unchanged.

## v0.6.19 - 2026-05-09

### Added

- **`VectorMemory(immutable_metadata_keys=...)`** — declares which
  `default_metadata` keys cannot be overridden by caller-supplied
  `metadata` on `add_message()`. Use for tenant-scoping identifiers
  (e.g. `immutable_metadata_keys=["user_id"]` paired with
  `default_metadata={"user_id": "..."}`). Caller-attempted overrides
  are logged as warnings and the configured value is preserved.
  Plumbed through `VectorMemory.from_config()`.

- **`VectorMemory.clear(filter_metadata=...)`** — filter-aware
  clear. When called with no args on a `VectorMemory` constructed
  with `default_filter=...`, the default filter is auto-applied,
  making `mem.clear()` symmetric with `mem.get_context()` for
  tenant-scoped instances. Pass `filter_metadata=...` explicitly to
  scope a clear to a different subset (e.g. one
  category/conversation within a tenant).

- **`RAGKnowledgeBase.clear(filter=...)`** — filter-aware clear,
  passing through to the underlying `VectorStore.clear(filter=)`.

### Fixed

- **`RAGKnowledgeBase._embed_and_store_chunks` no longer lets
  caller `metadata` overwrite system-controlled chunk fields**
  (`text`, `source`, `chunk_index`, `document_type`,
  `source_path`). Pre-fix, an ingest call passing
  `metadata={"text": "tampered"}` could silently corrupt stored
  chunks; the bug was reachable through every public ingest entry
  point. Caller-supplied values for system fields are now logged
  as warnings via `dataknobs_common.metadata.enforce_immutable_keys`
  and the system value is preserved.

- **`KnowledgeIngestionManager.ingest(domain_id, clear_existing=True)`
  no longer wipes other domains' chunks in a shared vector store.**
  Pre-fix, the manager called the underlying `VectorStore.clear()`
  with no filter, so refreshing one domain in a multi-tenant store
  removed every other domain's chunks silently. Post-fix, the clear
  is scoped by `domain_id` via
  `RAGKnowledgeBase.clear(filter={"domain_id": domain_id})`.
  Consumer-side workarounds (e.g. defaulting `clear_first=False`
  to dodge the issue) can be reverted on upgrade.

- **`RAGKnowledgeBase._embed_and_store_chunks` chunk IDs are now
  scoped by `domain_id` when present in the threaded metadata.**
  Pre-fix, the chunk-id stem was derived purely from
  `Path(source_file).stem`, so two chunks at the same relative
  filename across different domains (e.g. `domain-a/doc.md` and
  `domain-b/doc.md`) collided on a shared store and the second
  ingest upserted over the first. Post-fix, the chunk-id prefix
  becomes `f"{domain_id}\x1f{stem}"` whenever `domain_id` is in the
  caller-supplied metadata (which `KnowledgeIngestionManager`
  threads automatically). The record-separator (`\x1f`) between
  `domain_id` and `stem` rules out snake_case-domain collisions
  (`my` + `team_doc` vs `my_team` + `doc` would otherwise both
  produce `my_team_doc` under `_`). Single-domain consumers
  (no `domain_id` threaded) see **no change** — chunk IDs keep the
  historical `f"{stem}_{index}"` form, so re-ingest into existing
  populated stores remains idempotent.

- **`RAGKnowledgeBase.ingest_from_backend` no longer threads the
  redundant `source` and `filename` keys** that
  `KnowledgeBaseConfig.get_metadata` adds, into
  `_embed_and_store_chunks`. The chunk-build step already receives
  the more-precise `source_file` (display URI) and `source_path`
  (relative path) explicitly; dropping the redundant copies stops
  the new immutable-key enforcer from emitting a spurious warning
  on every legitimate ingest.

### Changed

- **`VectorMemory.clear()` semantics on tenant-scoped instances.**
  When `default_filter` is set, `clear()` (no args) now removes
  only the matching tenant's vectors, not the entire store. The
  pre-fix unscoped behavior was a documented gap (Brief 118
  sub-issue 8b); the docs steered consumers away from production
  `clear()` because it could not respect tenant scoping. This is
  a behavior change for tenant-scoped instances — consumers who
  genuinely want to wipe all tenants from a shared store should
  call `mem.vector_store.clear()` directly (bypassing the
  `VectorMemory` wrapper).

- **`VectorMemory.clear(filter_metadata=...)` now AND-composes
  with `default_filter` instead of replacing it.** Pre-fix, an
  explicit `filter_metadata` argument took full precedence over
  the memory's `default_filter`, allowing a tenant-scoped instance
  to wipe other tenants' rows in a shared store via an explicit
  override (e.g. tenant-A's memory could call
  `clear(filter_metadata={"user_id": "B"})` and remove tenant B's
  data). Post-fix the filters AND-compose, so explicit filters
  narrow WITHIN the tenant scope and never escape it. On key
  collision (caller passes a key that conflicts with the default)
  the merged filter contains contradictory clauses and matches
  nothing — the clear is a no-op rather than a cross-tenant wipe.

- **`KnowledgeBase` ABC now declares `clear(filter=...)`** with a
  default `NotImplementedError`. `RAGKnowledgeBase` overrides it
  with the filter-aware delete path. Subclasses that don't support
  deletion get a clean error rather than being silently
  mis-driven by managers like `KnowledgeIngestionManager`.

### Fixed

- **`MarkdownChunker.ChunkMetadata.to_dict()` no longer lets
  `custom` overwrite structured fields.** Pre-fix, `to_dict` ended
  with `**self.custom`, so a custom entry sharing a key with a
  structured field (`headings`, `chunk_index`, `chunk_size`,
  etc.) silently overwrote the structured value in the serialized
  dict — same vulnerability class as the pre-118 `_create_chunk`
  `node_type` defense, but covering the entire system-field
  surface. Post-fix, `**self.custom` is unpacked first so
  structured fields win.

- **`RAGKnowledgeBase._embed_and_store_chunks` chunk-id separator
  switched from `_` to `\x1f` (ASCII unit separator)** to
  eliminate snake-case-domain collisions. Pre-fix, the
  underscore-joined prefix caused
  `domain_id="my"`+file `team_doc.md` to collide with
  `domain_id="my_team"`+file `doc.md` (both produced
  `my_team_doc_0`). The unit-separator character cannot appear in
  domain IDs or file stems, so collisions are structurally
  impossible. Chunk IDs are not part of any documented public
  surface, so this is a safe internal change.

- **`RAGKnowledgeBase._embed_and_store_chunks` strips redundant
  `source` / `filename` keys from caller metadata at the shared
  layer.** Pre-fix, the strip lived only in
  `ingest_from_backend`, so direct callers of
  `load_markdown_text(metadata={"source": "..."})` still
  triggered a spurious immutable-key warning even though the
  caller's `source` was a redundant copy of the explicit
  `source_file` argument (different views of the same file). Now
  every entry point benefits.

- **Immutable-key warnings are emitted once per offense, not once
  per chunk.** Pre-fix, the per-chunk loop in
  `_embed_and_store_chunks` invoked `enforce_immutable_keys` on
  every chunk, so an N-chunk document with one bad metadata blob
  emitted N identical warnings. Post-fix, the helper is invoked
  with `caller=metadata` on the first chunk only (warning
  emission) and `caller=None` on subsequent chunks (silent
  enforcement) — one warning per offense.

### Migration

- Callers who currently rely on `default_metadata` for tenant
  scoping should add `immutable_metadata_keys=[...]` matching the
  scoping keys. Existing callers who do not set
  `immutable_metadata_keys` see no behavior change for
  `add_message` — caller metadata still wins on every key (the
  pre-fix default).
- Callers who relied on `VectorMemory.clear(filter_metadata=...)`
  as a "broader" wipe than `default_filter` (e.g. a tenant-A memory
  passing `filter_metadata={"category": "X"}` expecting to wipe
  category X across ALL tenants in the shared store) must update
  their code: the explicit filter now narrows WITHIN the tenant
  scope. For an all-tenants wipe, drop down to the underlying
  vector store: `mem.vector_store.clear(filter={"category": "X"})`.
- Callers of `RAGKnowledgeBase` ingest methods who passed
  caller-`metadata` containing `text`/`source`/`chunk_index`/
  `document_type`/`source_path` (a bug-shaped pattern) must update
  their code: those keys are now system-controlled and caller
  values are logged as warnings and discarded.
- **`VectorMemory.clear()` on tenant-scoped instances now
  auto-applies `default_filter`.** Code that called `clear()` to
  wipe an entire shared store (regardless of tenant scoping) will
  now wipe only the calling tenant's slice. Consumers who meant
  the all-tenants wipe should call `mem.vector_store.clear()`
  directly.
- **`KnowledgeIngestionManager.ingest(clear_existing=True)` is now
  safe in shared stores.** Workarounds that flipped
  `clear_existing` to `False` to avoid cross-domain wipes can be
  reverted on upgrade.

### Security
- Bumped minimum `jinja2` requirement from `>=3.1.0` to `>=3.1.6`
  to exclude versions affected by GHSA-cpwx-vrp4-4pq7,
  GHSA-gmj6-6f8f-6699, GHSA-h75v-3vvj-5mfj, and GHSA-q2x7-8rv6-6q7h.

### Added
- `EnsureIngestionResult.duration_seconds` property — counterpart
  to `IngestionResult.duration_seconds`. Computes
  `completed_at - started_at` in seconds. Returns `float` (not
  `float | None`): `EnsureIngestionResult.completed_at` is typed
  as `datetime` with a construction-time default factory, so a
  terminal result's duration is always defined.
- `RegistryBackend.peek_config(bot_id)` — non-mutating sibling of
  `get_config`. Returns the stored config dict without updating
  `last_accessed_at`, for inspection / audit / bookkeeping reads
  that should not register as user activity. Implemented on
  `InMemoryBackend`, `DataKnobsRegistryAdapter`, and
  `HTTPRegistryBackend`. The HTTP backend has no client-side
  activity state, so its `peek_config` delegates to `get_config`;
  servers that want to distinguish a non-touching peek may define
  their own contract (header, query parameter, or sibling
  endpoint) — this client deliberately does not impose one.

### Changed
- `BotRegistry.get_config()` now routes through
  `RegistryBackend.peek_config` rather than `get_config`.
  Inspection-style reads no longer bump `last_accessed_at`;
  consumers needing the touching behavior should use
  `BotRegistry.get_bot()`, which is the user-facing resolution
  path.
- `BotRegistry.get_bot()` now touches the backend on every call
  (cache hit and miss alike) so `last_accessed_at` reliably
  reflects user activity. Previously the backend `get_config`
  was issued only on the cache-miss branch, which produced an
  inverted activity signal — hot bots (always cache hits) never
  updated, cold bots updated only on TTL expiry. The change adds
  one backend read per `get_bot` call; for the HTTP backend that
  is one extra round trip per call, for the
  `DataKnobsRegistryAdapter` it is one extra `db.read` plus the
  pre-existing `db.update` that `get_config` already performed.
- `ConfigCachingManager.get_raw_config()` now routes through
  `RegistryBackend.peek_config`. Bypassing the cache also bypasses
  the activity bump, matching the inspection-path role the method
  already documents.
- `CachingRegistryManager.get_or_create()` cache-miss reads now
  route through `RegistryBackend.peek_config`. Previously
  `last_accessed_at` was bumped only on cache misses (cache hits
  bypass the backend), producing an inverted activity signal —
  hot bots never updated, cold bots updated only on TTL expiry.
  Storage timestamps now reflect direct backend reads only;
  user-activity tracking for `CachingRegistryManager` consumers
  belongs at the `get_or_create` caller (or higher) — if your
  deployment relied on cache-miss-as-activity, call
  `backend.get_config()` directly in the request-handling path,
  or migrate the call site to `BotRegistry.get_bot()` (which now
  bumps unconditionally).
- Non-UTF-8 backend bytes for a knowledge-base config raise
  `IngestionConfigError` from
  `RAGKnowledgeBase._load_kb_config_from_backend`. Previously a
  stray `UnicodeDecodeError` could escape this path.
- `EnsureIngestionResult.completed_at` is typed as `datetime`
  (non-optional) with a construction-time default factory. Every
  terminal state — skip, error, success — produced by
  `KnowledgeIngestionService.ensure_ingested`,
  `KnowledgeIngestionService.ingest_from_config`, and
  `AutoIngestionMixin._ensure_knowledge_base_ingested` carries a
  real timestamp; consumers that serialize via `to_dict()` see a
  consistent `"completed_at"` on every result. The
  ``IngestionResult`` → ``EnsureIngestionResult`` boundary in
  `from_ingestion_result` coalesces a not-yet-completed source
  (`IngestionResult.completed_at is None`) to
  `datetime.now(timezone.utc)` rather than weakening the
  invariant.
- `EnsureIngestionResult.to_dict()` now serializes `started_at`
  (ISO format), `completed_at` (ISO format), and
  `duration_seconds` — bringing it into shape parity with
  `IngestionResult.to_dict()`. Strict superset of prior keys; no
  removed keys.

### Internal
- `RAGKnowledgeBase._load_kb_config_from_backend` uses
  `dataknobs_common.config_loading.parse_yaml_or_json` for the
  bytes → dict parse. Surface is `IngestionConfigError`.

## v0.6.18 - 2026-05-06

## v0.6.17 - 2026-04-29

### Added
- `RAGKnowledgeBase.ingest_from_backend(backend, domain_id,
  config=None, progress_callback=None, extra_metadata=None)` —
  unified ingest for any `KnowledgeResourceBackend` (file, memory,
  S3) with full `KnowledgeBaseConfig` support: patterns, exclude
  patterns, per-pattern chunking overrides, streaming JSON/JSONL.
  When `config` is `None`, auto-loads
  `knowledge_base.(yaml|yml|json)` from the domain root (falling
  back to `_metadata/knowledge_base.*`); a malformed config raises
  `IngestionConfigError`. `extra_metadata` is merged onto every
  chunk — `KnowledgeIngestionManager` uses this to thread
  `{"domain_id": domain_id}` onto chunks so multi-tenant queries
  can filter on it.
- `IngestOrchestrator` (`dataknobs_bots.knowledge.orchestration`) —
  subscriber-side primitive that listens on an `EventBus` trigger
  topic and dispatches to
  `KnowledgeIngestionManager.ingest_if_changed`. Concurrent triggers
  for the same `domain_id` are serialized via a per-domain
  `asyncio.Lock`; different domains proceed in parallel. Stateless
  across restarts; trigger adapters (S3/SQS/cron → bus) remain
  consumer responsibility.
- `BackendDocumentSource` (re-exported from
  `dataknobs_xization.ingestion`) — adapts any
  `KnowledgeResourceBackend` to the `DocumentSource` protocol.
  Derives a common literal prefix from configured patterns and
  passes it to `backend.list_files(prefix=...)` so S3 (and any
  other prefix-aware backend) can avoid listing the whole bucket.
- `KnowledgeIngestionManager.ingest_if_changed(domain_id,
  last_version=None)` returning `IngestionResult | None` —
  returns `None` (and skips the ingest) when `last_version` is
  supplied and the backend reports no changes.
- `S3KnowledgeBackend` accepts a pre-built
  `session_config: S3SessionConfig` kwarg for sharing a single S3
  configuration across multiple backends.

### Changed
- `KnowledgeIngestionManager.ingest()` delegates to
  `RAGKnowledgeBase.ingest_from_backend` and threads
  `{"domain_id": domain_id}` into per-chunk metadata so downstream
  queries can filter by tenant.
- `S3KnowledgeBackend` `region` default flipped from `"us-east-1"`
  to `None`; client routes through `create_boto3_s3_client`. See
  `dataknobs-data` notes above for the behavior-change details and
  migration guidance.
- `S3KnowledgeBackend.from_config` accepts both `region` and
  `region_name` keys (parity with `SyncS3Database` /
  `AsyncS3Database`).
