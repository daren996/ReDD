# ReDD TODO

This document tracks the project work needed to make ReDD:

- a clean Python package installable with `pip`
- a public GitHub open-source repository
- a web demo that can evolve into a deployed service
- a productized pipeline that incrementally absorbs the efficiency-oriented modules from `ReDD_Dev`

The checklist is ordered roughly by priority.

## Current P0 Status Snapshot

This snapshot reflects the current repository state after the package/runtime boundary cleanup.

### Estimated status

- [x] `src/redd/` package root exists
- [x] `pyproject.toml` exists
- [x] installable CLI entry points exist through `pyproject.toml`
- [x] public stage entry points exist:
  - `preprocessing`
  - `schema_refinement`
  - `data_extraction`
  - `create_data_loader`
  - `run_pipeline`
- [x] query-independent and query-specific schema entry points now exist explicitly as:
  - `schema_global`
  - `schema_refine`
- [x] backward-compatible alias modules exist for old schema-facing names during the transition:
  - `global_schema`
  - `schema_refinement`
  - `schema_tailoring`
- [x] public package imports work at a basic level
- [x] provider resolution is centralized behind a registry/factory
- [x] prompt loading supports packaged resources instead of relying on the current working directory
- [x] repository scripts are thin wrappers over package CLI entry points
- [x] `scripts/_bootstrap.py` is gone
- [x] active package/library paths now raise exceptions instead of calling `exit()` in the main package flow
- [x] runtime/config/output-path responsibilities are centralized behind `src/redd/config.py` and `src/redd/runtime.py`
- [x] the first stable migration wave from `ReDD_Dev` is exposed and documented as wave 1, not full parity
- [x] targeted package-boundary tests cover config normalization, runtime helpers, prompt resources, provider routing, and API contracts

### Current assessment

- [x] Treat the current P0 status as complete for the package/runtime boundary
- [x] Treat the absorbed `ReDD_Dev` functionality as wave 1, not full migration parity
- [x] Prioritize controlled module promotion over broad new surface area

### Architectural debt still visible inside `src/redd/core/`

- [ ] Treat the current package boundary cleanup as complete, but do **not** treat the internal module layout as final
- [ ] Consolidate internals by **pipeline responsibility** rather than by provider, API vendor, or experiment lineage
- [ ] Use `core/llm/` as the main home for provider-specific differences instead of scattering them across stage folders
- [ ] Use backend adapters to separate `cloud` vs `local` execution, rather than separate stage implementations
- [ ] Convert optional optimizations such as doc filtering, proxy execution, alpha allocation, and schema tailoring into composable strategies/plugins instead of alternate stage classes
- [ ] Treat `ReDD_Dev` as the reference for useful algorithms, but not as the directory structure to copy literally
- [ ] Treat `core/` as a transitional internal container, not the final long-term package layout

## P0 Core Refactor

### Productized pipeline shape

- [x] Treat `ReDD_Dev` as the feature incubator for efficiency modules, and migrate selected modules into the package incrementally
- [x] Keep the package architecture centered on stable pipeline stages instead of one-off experiment scripts
- [x] Design the public pipeline around the following stage entry points:
  - `preprocessing`
  - `schema_refinement`
  - `data_extraction`
  - `create_data_loader`
  - `run_pipeline`
- [x] Map the user-facing stage names to the internal research terms:
  - `PREPROCESSING`
  - `SCHEMA REFINEMENT`
  - `DATA EXTRACTION`
- [x] Ensure each stage can run independently for debugging, benchmarking, and future web demo usage
- [x] Ensure `run_pipeline` can orchestrate the full end-to-end flow across all stages

### P0 blockers found during repository audit

- [x] Eliminate duplicated runtime responsibilities across `config.py`, `runtime.py`, `api.py`, and internal provider helpers
- [x] Finish the first stable migration wave before importing many more advanced modules from `ReDD_Dev`
- [x] Strengthen test coverage so the current package boundary is reliable before the next migration wave

### Project structure

- [x] Decide on one config directory name and use it consistently: `configs/`
- [x] Create `src/redd/` as the official package root
- [x] Separate reusable library code from experiment-only code
- [x] Keep `scripts/` for CLI and experiments, not core business logic
- [x] Keep `notebooks/`, `papers/`, `outputs/`, and raw datasets out of the package API surface
- [x] Remove `sys.path.append(...)` style imports from scripts
- [x] Remove the temporary `scripts/_bootstrap.py` `src/` path injection once packaging is in place

### Public API

- [x] Define a stable top-level API for external users
- [x] Expose a small number of supported entry points such as:
  - `SchemaGenerator`
  - `DataPopulator`
  - `preprocessing`
  - `schema_global`
  - `schema_refine`
  - `schema_refinement`
  - `data_extraction`
  - `create_data_loader`
  - `run_pipeline`
- [x] Mark legacy/internal modules as internal-only
- [x] Decide which objects are public API vs. internal building blocks
- [x] Keep class/function names Pythonic even if the product-facing stage labels stay uppercase in docs/UI

### Config and runtime

- [x] Unify config normalization across the packaged stage entry points
- [x] Support shared config plus module-specific config sections
- [x] Centralize API key loading
- [x] Centralize logging setup
- [x] Centralize output path construction and stage-owned output layout rules
- [x] Centralize schema-artifact source resolution and data-loader filemap injection
- [x] Ensure core modules do not call `exit()` directly in the main package flow
- [x] Ensure core modules raise exceptions instead of terminating the process in the main package flow

### Remaining cleanup targets

- [x] Reduce duplication between config loading, runtime setup, and pipeline orchestration helpers
- [x] Audit remaining lower-level/internal modules for stray process-terminating behavior
- [x] Ensure output-path resolution has one obvious home instead of being spread across multiple helper layers

### Providers and prompts

- [x] Replace scattered `if/elif` provider branching with a registry/factory
- [x] Normalize provider interface for OpenAI, DeepSeek, Together, SiliconFlow, local models, and future providers
- [x] Move prompt loading to package-safe resource loading
- [x] Stop relying on current working directory for prompt/template paths

### Domain boundaries

- [x] Keep cloud/provider selection behind adapters and provider factories
- [x] Keep pipeline orchestration in the package API layer instead of provider/model adapters
- [x] Keep data access behind loader/repository helpers instead of stage orchestration code
- [x] Keep evaluation/correction workflows outside the stable stage API surface unless explicitly promoted later

### Architecture consolidation audit: provider split vs stage composition

This section captures the next internal cleanup wave. The public package surface is already stage-oriented, but several internal folders are still organized in a legacy way. The target state is:

- one implementation per **stage**
- optional optimizations as **composable submodules**
- provider differences isolated in `core/llm/`
- dataset/storage differences isolated in `core/data_loader/`
- evaluation/correction kept outside the primary runtime unless explicitly promoted

### Target package layout after `core/` de-flattening

The current `core/` tree is acceptable as a transition state, but it should not remain the permanent home for every internal module. The preferred direction is:

- `src/redd/optimizations/`
  - `doc_filtering`
  - `adaptive_sampling`
  - `alpha_allocation`
- `src/redd/proxy/`
  - `predicate_proxy`
  - `join_resolution`
  - `proxy_runtime`
- `src/redd/correction/`
  - correction and reliability-oriented modules
- `src/redd/exp/`
  - `evaluation`
  - `experiments`

- [x] Add forward-looking namespace packages for `src/redd/optimizations/`, `src/redd/proxy/`, `src/redd/correction/`, and `src/redd/exp/` so new work stops defaulting to `src/redd/core/`

Layout rules:

- [x] Do **not** turn `optimizations/` into a new catch-all bucket
- [x] Keep only genuinely optional efficiency/performance modules in `optimizations/`
- [x] Keep runtime proxy capabilities out of `optimizations/`; they belong under `proxy/`
- [ ] Keep correction/reliability workflows out of the main runtime path unless explicitly promoted
- [ ] Keep evaluation and experiment code under `exp/` rather than mixing it into runtime product modules
- [ ] Migrate toward this structure incrementally instead of performing a large blind directory move

### Cross-cutting consolidation rules

- [x] Standardize on the rule: `one stage = one main orchestrator module`
- [x] Stop creating separate pipeline modules just because the LLM provider changes
- [x] Stop creating separate stage modules just because the backend is `local` vs `API`
- [ ] Stop encoding legacy research branches as peer stage implementations when they are really optional strategies inside a stage
- [ ] Keep stage orchestrators thin: they should coordinate loaders, LLM backends, prompt runners, and optimization strategies instead of implementing backend-specific logic inline
- [x] Move provider-specific request formatting, auth, retries, and model naming into `core/llm/`
- [ ] Keep future external integrations behind adapters, similar to `text_to_sql.py`, instead of letting third-party frameworks shape the package architecture

### Folder-by-folder consolidation plan

#### `src/redd/core/data_population/`

Current issue:

- [x] Treat `data_population/` as the clearest example of legacy over-splitting that has now been substantially cleaned up
- [x] Acknowledge that the folder is now centered on:
  - one unified stage implementation: `datapop.py`
  - a thin factory: `factory.py`
  - a minimal internal export surface: `__init__.py`
  - support utilities and sub-strategies rather than provider-specific stage files
- [ ] Keep the remaining work focused on making unified datapop thinner and more strategy-oriented, rather than continuing file proliferation

Desired direction:

- [x] Make `datapop.py` the single canonical internal implementation of the data-extraction stage
- [x] Keep `DataPopulator` and `data_extraction()` wired to that one orchestrator rather than conditionally selecting different stage classes
- [x] Move provider choice entirely behind `core/llm/` and backend adapters
- [x] Treat `local` as a backend mode of unified datapop, not a separate `datapop_local.py` stage
- [x] Treat legacy proxy execution as an optional extraction strategy, not a separate legacy branch file such as `datapop_ccg.py`
- [ ] Treat `doc_filter`, `alpha_allocation`, and similar optimizations as optional pluggable strategies inside unified datapop
- [x] Reduce `factory.py` so it always returns the unified datapop orchestrator plus configured strategies/backends

Concrete cleanup tasks:

- [x] Collapse provider-specific datapop files into backend/provider configuration handled by `core/llm/`
- [x] Fold local execution into the unified datapop backend selection path
- [x] Fold the legacy proxy branch into optional predicate-proxy / join-resolution / execution-policy hooks used by `datapop.py`
- [x] Remove old base/API/provider datapop variants after the unified path stabilized
- [x] Shrink `src/redd/core/data_population/__init__.py` so it no longer exports many legacy/populator-specific classes
- [x] Treat `res_to_db.py` and other support helpers as utilities, not alternate datapop implementations
- [ ] Continue extracting optional runtime features into explicit strategy modules under datapop instead of growing `datapop.py` further

Desired end state:

- [x] One canonical datapop orchestrator
- [x] Zero provider-specific datapop files
- [x] Zero local-vs-cloud datapop file split
- [x] Zero legacy proxy-branch-as-a-peer-datapop-implementation split

#### `src/redd/core/schema_gen/`

Current issue:

- [ ] Treat `schema_gen/` as a smaller version of the same problem seen in `data_population/`
- [ ] Acknowledge that the folder still contains provider-split modules:
  - one canonical implementation: `schemagen.py`
  - one remaining compatibility alias: `schemagen_gpt.py`
  - plus `schemagen_basic.py` and `factory.py`

Desired direction:

- [x] Converge on one canonical schema-generation orchestrator, for example `schemagen.py` or a unified successor to the current main implementation
- [x] Move provider-specific differences into `core/llm/` instead of `schema_gen/`
- [ ] Keep adaptive semantic sampling as a composable optimization under schema generation, not a reason to fork provider-specific stage files
- [x] Let `factory.py` resolve one generator implementation plus backend configuration, not one file per provider

Concrete cleanup tasks:

- [ ] Reduce schema-generation compatibility aliases to the minimum necessary set, then remove them after deprecation
- [ ] Decide whether `schemagen_basic.py` is still needed as a reusable base class or should be replaced by a thinner internal protocol/base
- [x] Simplify `src/redd/core/schema_gen/__init__.py` so it stops exporting legacy/provider-specific generator classes
- [x] Make the schema-generation folder match the same architectural rule as datapop: one stage orchestrator, many optional helpers

Desired end state:

- [x] One canonical schema-generation implementation
- [x] Provider differences isolated in `core/llm/`
- [ ] Adaptive sampling remains available without creating more generator variants

#### Legacy proxy-module replacement plan

Current issue:

- [x] Treat the old `ccg/` area as legacy naming that should disappear from the product architecture entirely
- [x] Acknowledge that the old proxy area has already been decomposed into runtime-oriented modules, but still leaves legacy terminology in comments, config compatibility, and docs:
  - predicate filtering
  - proxy execution/runtime wiring
  - proxy ordering and calibration
  - join handling
  - oracle/experimental helpers
  - pretraining data utilities

Desired direction:

- [ ] Remove the old `ccg` concept from public and internal architecture names
- [ ] Remove the word `guard` from the product/runtime vocabulary
- [x] Split the legacy proxy area into three clearer concepts:
  - `predicate_proxy`
  - `join_resolution`
  - `proxy_runtime`
- [ ] Keep lower-level experimental components such as oracle/pretraining helpers internal until they justify promotion
- [x] Stop representing the old proxy stack as a monolithic peer of datapop; instead compose `predicate_proxy`, `join_resolution`, and `proxy_runtime` into data extraction

Concrete cleanup tasks:

- [x] Rename or replace the old `src/redd/core/ccg/` area so the surviving runtime code lives under proxy-oriented names
- [x] Move predicate filtering/proxy logic under `predicate_proxy`
- [x] Move join-aware logic under `join_resolution`
- [x] Move execution ordering, calibration, orchestration, and runtime coordination under `proxy_runtime`
- [x] Normalize `proxy_runtime` internals toward proxy naming and centralize legacy guard-key compatibility reads
- [ ] Decide which legacy files are:
  - promotable into `predicate_proxy`
  - promotable into `join_resolution`
  - promotable into `proxy_runtime`
  - experiment-only and deletable from the package runtime path
- [ ] Ensure public users think in terms of `predicate_proxy`, `join_resolution`, and `proxy_runtime`, never the old legacy label
- [x] Remove datapop-facing coupling to the old monolithic proxy branch path
- [ ] Remove remaining legacy compatibility keys and comments such as `ccg`, `use_ccg`, `guard_threshold`, `use_embedding_guards`, and similar transitional wording
- [x] Keep `gliclass_pretrain_data.py` and similar assets out of the stable runtime path unless explicitly needed for packaged training workflows

Desired end state:

- [ ] No `ccg` terminology remains in the product architecture
- [ ] No `guard` terminology remains in the product architecture
- [x] `predicate_proxy` owns predicate-level proxy logic
- [x] `join_resolution` owns join-aware resolution logic
- [x] `proxy_runtime` owns proxy orchestration, ordering, calibration, and runtime composition
- [x] Data extraction composes proxy modules without splitting into a separate legacy branch
- [x] The proxy stack lives under `src/redd/proxy/`, not under `optimizations/`

#### `src/redd/core/doc_filtering/`

Current status:

- [x] Treat `doc_filtering/` as closer to the desired architecture than `data_population/` or `schema_gen/`
- [x] Acknowledge that it already has:
  - a base abstraction
  - runtime wiring
  - concrete strategies
  - a composite filter

Follow-up cleanup:

- [ ] Use `doc_filtering/` as the model for other optimization folders: one clear interface plus pluggable strategies
- [ ] Keep runtime/config wiring separate from actual filter logic
- [ ] Avoid letting stage-specific orchestration leak deeply into filter implementations
- [ ] Ensure doc filters plug into `schema_refine` or `data_extraction` through a stable strategy interface rather than ad hoc conditionals

#### Future `src/redd/optimizations/`

- [x] Add a compatibility namespace for `src/redd/optimizations/` before the physical move so future module placement is explicit
- [x] Group optimization-oriented modules under `src/redd/optimizations/`
- [x] Move `doc_filtering`, `adaptive_sampling`, and `alpha_allocation` into this optimization-oriented area when import stability and compatibility shims are ready
- [x] Keep the scope of `optimizations/` narrow: optional efficiency/performance modules only
- [x] Do not place `predicate_proxy`, `join_resolution`, or `proxy_runtime` under `optimizations/`

#### `src/redd/core/data_loader/`

Current status:

- [x] Treat `data_loader/` differently from provider-split folders because separate loaders can be valid when the storage/dataset model genuinely differs
- [x] Acknowledge that `data_loader/` is already closer to a legitimate registry/factory architecture

Required cleanup:

- [ ] Distinguish clearly between:
  - storage-backend differences
  - dataset-layout differences
  - dataset-specific quirks that may not deserve a dedicated loader class
- [ ] Decide whether loaders such as `data_loader_cuad.py` should remain standalone or become profiles/adapters over a more general loader
- [ ] Keep `create_data_loader()` as the stable factory boundary
- [ ] Avoid letting loader proliferation turn into the same anti-pattern seen in provider-specific stage modules
- [ ] Document which loaders are truly distinct and which exist only because of historical experiment setup

Desired end state:

- [ ] One registry-driven loader family with only genuinely different loader implementations
- [ ] Dataset-specific details encoded through config/profiles where possible

#### `src/redd/core/schema_tailor/`

- [x] Treat `schema_tailor/` as a sub-strategy of query-specific schema refinement, not as a parallel stage family
- [ ] Keep `schema_tailor.py` focused on refinement-specific transformations
- [ ] Avoid expanding `schema_tailor/` into another provider/backend split area
- [x] Route new public usage through `schema_refine` while keeping `schema_refinement` and `schema_tailoring` as backward-compatible compatibility layers
- [ ] Remove compatibility forwarding layers once downstream imports have been migrated and documented deprecation windows have passed

#### `src/redd/core/doc_chunking/`

Current status:

- [x] Rename the legacy `chunking/` module family to `doc_chunking/`
- [x] Update package-internal imports from `redd.core.chunking` to `redd.core.doc_chunking`
- [x] Keep document chunking terminology aligned with the rest of the package naming scheme

Follow-up cleanup:

- [ ] Audit the repository for remaining user-facing docs or comments that still use the old `chunking` module path when they really mean `doc_chunking`
- [ ] Decide whether doc chunking should stay a pure internal utility family or gain a stable public façade in a later wave
- [ ] Keep chunking-specific storage/output assumptions from leaking into unrelated stage APIs

#### `src/redd/core/embedding/` and `src/redd/core/llm/`

Desired role:

- [ ] Treat `embedding/` and `llm/` as the right places for backend/provider normalization
- [ ] Keep model/provider naming, auth, client creation, retry policy, and backend dispatch here instead of in stage folders
- [ ] Use these folders to absorb the complexity that should be removed from `data_population/` and `schema_gen/`

Concrete tasks:

- [ ] Expand `core/llm/` so stage modules can depend on one normalized model-execution interface
- [ ] Decide whether embeddings and LLMs should share common backend/runtime helpers
- [ ] Keep the public API provider-agnostic even if internal adapters vary significantly

#### `src/redd/core/evaluation/` and `src/redd/core/correction/`

- [ ] Keep evaluation and correction modules out of the primary pipeline stage surface for now
- [ ] Treat them as separate workflow families until they earn promotion into the stable package API
- [ ] Prevent these folders from leaking research/training assumptions into runtime package dependencies
- [ ] When promoting anything from these folders later, expose it as a clearly separate subsystem rather than bloating `preprocessing`, `schema_refinement`, or `data_extraction`
- [x] Add forward-looking `src/redd/correction/` and `src/redd/exp/` namespaces so future migrations do not continue targeting `src/redd/core/`
- [x] Move the main evaluation implementation behind `src/redd/exp/` and leave `src/redd/core/evaluation/` as a compatibility layer
- [x] Move foundational correction modules toward `src/redd/correction/` and leave legacy `src/redd/core/correction/` imports as compatibility paths
- [x] Move the main correction training/evaluation loaders behind `src/redd/correction/` and leave `src/redd/core/correction/` as compatibility paths
- [x] Move ensemble-analysis helpers behind `src/redd/correction/` and leave `src/redd/core/correction/` as compatibility paths
- [ ] Continue moving remaining correction-oriented modules toward `src/redd/correction/`
- [ ] Move evaluation and experiment-oriented modules toward `src/redd/exp/`

### Consolidation principles to preserve during future migration

- [ ] Migrate algorithms from `ReDD_Dev`, not folder sprawl
- [ ] Prefer `one stage + many strategies` over `many sibling stage implementations`
- [ ] Prefer `backend adapters` over `provider-named stage files`
- [ ] Prefer `public stage façades` over exposing internal research namespaces directly
- [ ] Prefer deleting alias wrappers after stabilization instead of keeping them forever as dead architectural weight

### Incremental module migration from `ReDD_Dev`

- [x] Create a migration plan that brings in `ReDD_Dev` modules in priority order instead of copying the whole repo blindly
- [x] Migrate modules only after assigning each one to a stable package boundary and entry point
- [x] Keep research-only code, ablation code, and package-ready code clearly separated during migration
- [x] Treat the current migration as "wave 1 only" rather than full parity with `ReDD_Dev`
- [x] Only migrate additional efficiency modules after the package/runtime boundary is stable
- [x] Record, for each migrated module, whether it is:
  - public API
  - internal runtime-only
  - experiment-only

### Target package modules

- [x] Add a preprocessing module for query-independent work
- [x] Add a schema refinement module for query-conditioned schema updates
- [x] Add a data extraction module for final table/attribute extraction
- [x] Add a loader module exposing `create_data_loader`
- [x] Add a pipeline orchestrator exposing `run_pipeline`
- [x] Add explicit schema-stage helper entry points:
  - `schema_global`
  - `schema_refine`

### Wave-1 efficiency surfaces

- [x] Add an embedding and retrieval layer for preprocessing
- [x] Prefer an adapter boundary so embeddings/retrieval can use an external retrieval or RAG-style library without coupling the whole package to that framework
- [x] Rename chunking internals to `doc_chunking` for clearer document-centric terminology
- [x] Add a global schema discovery module
- [x] Include adaptive semantic sampling in the global schema discovery step
- [x] Expose a doc filtering module for query-aware pruning
- [x] Expose schema-tailoring support for refinement-specific optimizations
- [x] Expose predicate-level proxy support
- [x] Expose join handling modules
- [x] Expose parameter tuning / calibration / optimization modules where they are stable enough to expose
- [x] Expose a text-to-SQL adapter layer instead of implementing text-to-SQL from scratch inside the core package
- [x] Define stable package names for these modules:
  - `doc_chunking`
  - `embedding`
  - `retrieval`
  - `global_schema`
  - `adaptive_sampling`
  - `doc_filtering`
  - `schema_tailoring`
  - `predicate_proxy`
  - `join_resolution`
  - `proxy_runtime`
  - `parameter_optimization`
  - `text_to_sql`

### Current migration status of efficiency modules

- [x] Stable public package surfaces now exist for:
  - `embedding`
  - `retrieval`
  - `global_schema`
  - `adaptive_sampling`
  - `doc_filtering`
  - `schema_tailoring`
  - `predicate_proxy`
  - `join_resolution`
  - `proxy_runtime`
  - `parameter_optimization`
  - `text_to_sql`
- [x] `schema_global` exists as the explicit query-independent schema entry point
- [x] `schema_refine` exists as the explicit query-specific schema entry point
- [x] `schema_refinement` remains available as a backward-compatible alias during the transition
- [x] `data_extraction` is the stable stage that consumes execution-side optimization modules
- [x] `ReDD_Dev` should still be treated as the source of future migration candidates, not as something already fully absorbed

### Stage-by-stage responsibilities

- [x] Define `preprocessing` as the query-independent stage responsible for:
  - embeddings
  - doc chunking support where preprocessing-time document splitting is needed
  - retrieval index preparation
  - global schema discovery
  - adaptive semantic sampling support
- [x] Define `schema_global` as the direct helper API for query-independent schema extraction
- [x] Define `schema_refine` as the direct helper API for query-aware schema extraction
- [x] Keep `schema_refinement` as a compatibility alias over `schema_refine` during migration
- [x] Define query-aware schema refinement as responsible for:
  - query-conditioned schema generation
  - query-conditioned schema artifacts
  - optional `schema_tailor` refinement support
- [x] Define `data_extraction` as the stage responsible for:
  - optional doc filtering
  - table assignment
  - attribute extraction
  - result assembly
  - predicate proxy execution
  - optional join-aware execution
  - proxy runtime orchestration where predicate proxies and join resolution need coordinated execution
  - optional alpha-allocation tuning
  - adapter-style text-to-SQL integration where needed

### Rules for future migration waves

- [x] For the current wave, imported modules have documented classification and stable entry points where appropriate
- [x] External libraries are wrapped behind adapters or factories where they enter the stable package surface
- [x] The package remains centered on ReDD concepts, not on any single third-party framework
- [ ] Before promoting more `ReDD_Dev` modules, ensure each new optimization has:
  - a config surface
  - tests
  - a clear on/off switch
  - a fallback path

### Follow-up after P0

- [ ] Decide which evaluation and correction components should become package-ready in wave 2
- [ ] Deepen end-to-end regression coverage against representative datasets instead of only unit-level package contracts
- [ ] Decide whether the future web/demo layer should consume the stable package directly or sit behind a thinner service adapter

## P1 Packaging

### Packaging foundation

- [x] Add `pyproject.toml`
- [x] Choose a build backend
- [x] Decide the public package name
- [x] Decide minimum supported Python version
- [x] Add package metadata:
  - name
  - version
  - description
  - readme
  - license
  - authors
  - repository/homepage URLs

### Source layout

- [x] Move package code under `src/redd/`
- [ ] Ensure imports work after installation, not only from repo root
- [x] Include prompts/templates/resources as package data

### CLI

- [x] Add a supported CLI entry point such as `redd`
- [ ] Support subcommands for schema generation, data population, evaluation, and config inspection
- [x] Ensure scripts become thin wrappers around library calls

### Dependency strategy

- [ ] Minimize core dependencies
- [ ] Split optional dependencies into extras:
  - `.[dev]`
  - `.[web]`
  - `.[gpu]`
  - `.[retrieval]`
  - `.[text2sql]`
- [ ] Avoid shipping heavy web/GPU dependencies in the default install
- [ ] Keep retrieval/RAG dependencies optional
- [ ] Keep text-to-SQL dependencies optional

### Packaging validation

- [ ] Verify `pip install -e .`
- [ ] Verify `python -m build`
- [ ] Verify wheel install in a clean environment
- [ ] Verify package resources are available after installation

## P1 Quality and Tests

### Test baseline

- [ ] Create a consistent `tests/` layout
- [x] Add `pytest`
- [x] Add a minimal smoke test for the main pipeline

### Priority test areas

- [ ] Data loader tests
- [ ] Config parsing tests
- [ ] Prompt/resource loading tests
- [ ] Provider factory tests
- [ ] Output format tests
- [ ] Error handling tests
- [ ] Preprocessing stage tests
- [ ] Schema refinement stage tests
- [ ] Data extraction stage tests
- [ ] Module migration regression tests for components imported from `ReDD_Dev`

### Tooling

- [ ] Add formatting and linting tooling
- [ ] Add basic type checking
- [ ] Add CI to run lint, tests, and build checks

## P1 Open Source Readiness

### Repository health

- [ ] Add `LICENSE`
- [ ] Rewrite `README.md`
- [ ] Add `CONTRIBUTING.md`
- [ ] Add `SECURITY.md`
- [ ] Add issue templates
- [ ] Add pull request template
- [ ] Add `.gitignore` cleanup if needed

### README content

- [ ] Explain what ReDD does
- [ ] Add quick start instructions
- [ ] Add installation instructions
- [ ] Add a minimal usage example
- [ ] Document supported providers and datasets
- [ ] Document the pipeline stages and their responsibilities
- [ ] Document which advanced efficiency modules are included from `ReDD_Dev`
- [ ] Document limitations and roadmap

### Open-source cleanup

- [ ] Audit the repository for secrets, local absolute paths, and personal environment assumptions
- [ ] Decide what datasets, checkpoints, outputs, and generated artifacts can be public
- [ ] Remove or relocate files that should not be shipped or published
- [ ] Decide on the license before making the repository public

## P2 Web Demo

### Backend

- [ ] Build a clean service layer that imports `redd` instead of calling experiment scripts directly
- [ ] Add a minimal FastAPI backend
- [ ] Define stable request/response schemas
- [ ] Add endpoints for:
  - health check
  - config inspection
  - preprocessing job submission
  - schema refinement job submission
  - data extraction job submission
  - end-to-end pipeline job submission
  - result lookup

### Long-running jobs

- [ ] Design for async job execution rather than long blocking HTTP requests
- [ ] Introduce a worker/job abstraction if needed
- [ ] Plan how logs and progress updates are surfaced to the frontend

### Frontend

- [ ] Start with a minimal demo UI
- [ ] Focus first on one happy path:
  - choose dataset/task
  - run pipeline
  - inspect result
- [ ] Later expose advanced controls for optional efficiency modules without making the basic demo too complex
- [ ] Avoid frontend-specific business logic duplication

## P2 Deployment Preparation

### Containerization

- [ ] Add a backend `Dockerfile`
- [ ] Add a frontend `Dockerfile`
- [ ] Add a `docker-compose.yml`
- [ ] Ensure environment-variable-based configuration
- [ ] Add health checks
- [ ] Send logs to stdout/stderr

### Storage and ops

- [ ] Decide how outputs are stored in demo mode
- [ ] Design storage so it can later move from local files to DB/object storage
- [ ] Keep the deployment target `docker compose` first
- [ ] Do not introduce Kubernetes until the service shape justifies it

## P3 Publishing and Release Flow

### Versioning and release

- [ ] Define versioning policy
- [ ] Add GitHub Actions for test/build
- [ ] Add GitHub Actions for release
- [ ] Test publishing on TestPyPI
- [ ] Configure PyPI Trusted Publishing
- [ ] Verify install and CLI behavior from published artifacts

## Suggested Immediate Work

These are the best next steps to execute first:

- [x] Decide final package name
- [x] Create `pyproject.toml`
- [x] Create `src/redd/`
- [x] Define the package stage entry points: `preprocessing`, `schema_refinement`, `data_extraction`, `create_data_loader`, `run_pipeline`
- [x] Introduce explicit schema helper entry points: `schema_global`, `schema_refine`
- [x] Rename `chunk_filtering` to `doc_filtering`
- [x] Rename `chunking` to `doc_chunking`
- [x] Create a provider factory
- [x] Create a unified config loader
- [x] Add baseline tests
- [ ] Create the migration order for `ReDD_Dev` modules
- [ ] Remove compatibility forwarding layers after the new schema/doc naming has fully propagated:
  - `global_schema.py`
  - `schema_refinement.py`
  - `schema_tailoring.py`
- [ ] Audit `src/redd/core/data_population/` and define the deletion/consolidation path toward one canonical `datapop.py`
- [ ] Rewrite `src/redd/core/data_population/factory.py` so backend/strategy composition replaces provider-specific stage files
- [ ] Shrink `src/redd/core/data_population/__init__.py` and stop exporting provider-specific/legacy datapop classes
- [ ] Audit `src/redd/core/schema_gen/` and define the deletion/consolidation path toward one canonical schema-generation orchestrator
- [ ] Rewrite `src/redd/core/schema_gen/factory.py` so provider resolution stays in `core/llm/`
- [ ] Remove `ccg` naming entirely and split the remaining runtime into:
  - `predicate_proxy`
  - `join_resolution`
  - `proxy_runtime`
- [ ] Decide which `data_loader` implementations are truly distinct and which should become profiles over a shared loader
- [ ] Consolidate runtime/config/output-path helpers
- [ ] Add deeper stage/module regression tests
- [ ] Tighten CLI surface and packaging validation
- [ ] Continue the next controlled migration wave from `ReDD_Dev`
- [ ] Rewrite `README.md` for open-source readiness rather than internal transition status

## Notes

- LangChain is not a current priority.
- Kubernetes is not a current priority.
- Focus first on a clean, packageable, testable Python core.
- The web app should consume the package, not duplicate its logic.
- External frameworks for retrieval/RAG and text-to-SQL should be integrated through adapters, not used as the package architecture itself.
