const form = document.querySelector("#run-form");
const configPath = document.querySelector("#config-path");
const experiment = document.querySelector("#experiment");
const apiKey = document.querySelector("#api-key");
const apiKeyStatus = document.querySelector("#api-key-status");
const configProjectName = document.querySelector("#config-project-name");
const configProjectSeed = document.querySelector("#config-project-seed");
const configExperimentId = document.querySelector("#config-experiment-id");
const configOutputDir = document.querySelector("#config-output-dir");
const configLogDir = document.querySelector("#config-log-dir");
const configArtifactId = document.querySelector("#config-artifact-id");
const configOutputLayout = document.querySelector("#config-output-layout");
const configConsoleLogLevel = document.querySelector("#config-console-log-level");
const forceRerun = document.querySelector("#force-rerun");
const configLlmEnabled = document.querySelector("#config-llm-enabled");
const configLlmProvider = document.querySelector("#config-llm-provider");
const configLlmModel = document.querySelector("#config-llm-model");
const configLlmCustomWrap = document.querySelector("#config-llm-custom-wrap");
const configLlmCustomModel = document.querySelector("#config-llm-custom-model");
const configLlmApiKeyEnv = document.querySelector("#config-llm-api-key-env");
const configLlmBaseUrl = document.querySelector("#config-llm-base-url");
const configLlmStructuredBackend = document.querySelector("#config-llm-structured-backend");
const configLlmMaxRetries = document.querySelector("#config-llm-max-retries");
const configLlmWaitTime = document.querySelector("#config-llm-wait-time");
const configLlmTemperature = document.querySelector("#config-llm-temperature");
const configLlmTopP = document.querySelector("#config-llm-top-p");
const configLlmMaxTokens = document.querySelector("#config-llm-max-tokens");
const configLlmLocalModelPath = document.querySelector("#config-llm-local-model-path");
const configEmbeddingEnabled = document.querySelector("#config-embedding-enabled");
const configEmbeddingProvider = document.querySelector("#config-embedding-provider");
const configEmbeddingModel = document.querySelector("#config-embedding-model");
const configEmbeddingCustomWrap = document.querySelector("#config-embedding-custom-wrap");
const configEmbeddingCustomModel = document.querySelector("#config-embedding-custom-model");
const configEmbeddingApiKeyEnv = document.querySelector("#config-embedding-api-key-env");
const configEmbeddingBaseUrl = document.querySelector("#config-embedding-base-url");
const configEmbeddingBatchSize = document.querySelector("#config-embedding-batch-size");
const configEmbeddingStorageFile = document.querySelector("#config-embedding-storage-file");
const configOptDocFilterEnabled = document.querySelector("#config-opt-doc-filter-enabled");
const configOptSchemaAdaptiveEnabled = document.querySelector("#config-opt-schema-adaptive-enabled");
const configOptSchemaAdaptiveThreshold = document.querySelector("#config-opt-schema-adaptive-threshold");
const configOptSchemaAdaptiveStreak = document.querySelector("#config-opt-schema-adaptive-streak");
const configOptSchemaAdaptiveMinDocs = document.querySelector("#config-opt-schema-adaptive-min-docs");
const configOptProxyEnabled = document.querySelector("#config-opt-proxy-enabled");
const configOptProxyMode = document.querySelector("#config-opt-proxy-mode");
const configOptProxyModel = document.querySelector("#config-opt-proxy-model");
const configOptProxyLearned = document.querySelector("#config-opt-proxy-learned");
const configOptProxyEmbedding = document.querySelector("#config-opt-proxy-embedding");
const configOptProxyFallback = document.querySelector("#config-opt-proxy-fallback");
const configOptProxyEpochs = document.querySelector("#config-opt-proxy-epochs");
const configOptJoinEnabled = document.querySelector("#config-opt-join-enabled");
const configStagesJson = document.querySelector("#config-stages-json");
const configDatasetList = document.querySelector("#config-dataset-list");
const configDatasetCount = document.querySelector("#config-dataset-count");
const configSectionButtons = document.querySelectorAll("[data-config-section-button]");
const configSectionPages = document.querySelectorAll("[data-config-section-page]");
const runButton = document.querySelector("#run-button");
const loadConfig = document.querySelector("#load-config");
const runTitle = document.querySelector("#run-title");
const viewEyebrow = document.querySelector("#view-eyebrow");
const statusPill = document.querySelector("#status-pill");
const errorPanel = document.querySelector("#error-panel");
const stageResults = document.querySelector("#stage-results");
const rawJson = document.querySelector("#raw-json");
const refreshButton = document.querySelector("#refresh-button");
const refreshLabel = document.querySelector("#refresh-label");
const resultsLibrary = document.querySelector("#results-library");
const resultsCountLabel = document.querySelector("#results-count-label");
const architectureFunnelPanel = document.querySelector(".architecture-funnel-panel");
const architectureFunnelBody = document.querySelector("#architecture-funnel-body");
const architectureFunnelToggle = document.querySelector("#architecture-funnel-toggle");
const consoleStateLabel = document.querySelector("#console-state-label");
const consoleEmptyState = document.querySelector("#console-empty-state");
const consoleActive = document.querySelector("#console-active");
const consoleRunId = document.querySelector("#console-run-id");
const consoleRunStatus = document.querySelector("#console-run-status");
const consoleRunElapsed = document.querySelector("#console-run-elapsed");
const consoleCurrentStep = document.querySelector("#console-current-step");
const consoleStepper = document.querySelector("#console-stepper");
const optimizationCards = document.querySelector("#optimization-cards");
const optimizationFeed = document.querySelector("#optimization-feed");
const evaluationCards = document.querySelector("#evaluation-cards");
const consoleProgressList = document.querySelector("#console-progress-list");
const consoleLogs = document.querySelector("#console-logs");
const clearConsoleLogs = document.querySelector("#clear-console-logs");
const copyConsoleLogs = document.querySelector("#copy-console-logs");
const generatedConfig = document.querySelector("#generated-config");
const copyConfig = document.querySelector("#copy-config");
const configSummary = document.querySelector("#config-summary");
const configWorkspace = document.querySelector("#config-workspace");
const configEmptyState = document.querySelector("#config-empty-state");
const summaryExperiment = document.querySelector("#summary-experiment");
const summaryDatasets = document.querySelector("#summary-datasets");
const summaryStages = document.querySelector("#summary-stages");
const summaryElapsed = document.querySelector("#summary-elapsed");
const datasetBrowser = document.querySelector("#dataset-browser");
const datasetDetail = document.querySelector("#dataset-detail");
const detailTitle = document.querySelector("#detail-title");
const datasetSelect = document.querySelector("#dataset-select");
const datasetPickerTrigger = document.querySelector("#dataset-picker-trigger");
const datasetPickerLabel = document.querySelector("#dataset-picker-label");
const datasetPickerBadge = document.querySelector("#dataset-picker-badge");
const datasetPickerCount = document.querySelector("#dataset-picker-count");
const datasetPickerList = document.querySelector("#dataset-picker-list");
const addDataset = document.querySelector("#add-dataset");
const datasetSourceLabel = document.querySelector("#dataset-source-label");
const selectedDatasets = document.querySelector("#selected-datasets");
const selectedCount = document.querySelector("#selected-count");
const targetDatasetCount = document.querySelector("#target-dataset-count");
const targetQueryCount = document.querySelector("#target-query-count");
const selectedQueryCount = document.querySelector("#selected-query-count");
const querySelectTrigger = document.querySelector("#query-select-trigger");
const querySelectLabel = document.querySelector("#query-select-label");
const querySelectBadge = document.querySelector("#query-select-badge");
const overviewState = document.querySelector("#overview-state");
const overviewSelectedDatasets =
  document.querySelector("#overview-selected-datasets") || targetDatasetCount;
const overviewSelectedQueries =
  document.querySelector("#overview-selected-queries") || targetQueryCount;
const overviewRunDatasets = document.querySelector("#overview-run-datasets");
const overviewRunQueries = document.querySelector("#overview-run-queries");
const overviewOracleCost = document.querySelector("#overview-oracle-cost");
const overviewSystemCost = document.querySelector("#overview-system-cost");
const overviewSavings = document.querySelector("#overview-savings");
const overviewRecall = document.querySelector("#overview-recall");
const relationPreviewBody = document.querySelector("#relation-preview-body");
const queryList = document.querySelector("#query-list");
const configSourcePanel = document.querySelector("#config-source-panel");
const allSourcePanel = document.querySelector("#all-source-panel");
const uploadSourcePanel = document.querySelector("#upload-source-panel");
const themeToggle = document.querySelector("#theme-toggle");
const themeLabel = document.querySelector("#theme-label");
const paperViewerMount = document.querySelector("#paper-viewer-mount");
const paperTabs = document.querySelectorAll(".paper-tab");
const paperExperimentList = document.querySelector("#paper-experiment-list");
const paperExperimentRefresh = document.querySelector("#paper-experiment-refresh");
const paperExperimentRunAll = document.querySelector("#paper-experiment-run-all");
const CUSTOM_MODEL_VALUE = "__custom__";
const SEARCH_DATASET_ALLOWLIST = new Set(["spider.college_demo", "bird.schools_demo"]);
const MAX_CONSOLE_LOGS = 180;
const CONSOLE_LOG_RENDER_LIMIT = 90;
const RESULTS_RENDER_BATCH = 24;
const THEME_STORAGE_KEY = "redd-theme-fastredd";
const DEFAULT_MODELS = {
  llm: {
    provider: "none",
    model: "ground_truth",
    api_key_env: null,
    base_url: null,
    structured_backend: "auto",
    max_retries: 5,
    wait_time: 10,
    temperature: null,
    top_p: null,
    max_tokens: null,
    local_model_path: null,
  },
  embedding: {
    provider: "local",
    model: "local-hash-embedding",
    enabled: true,
    api_key_env: null,
    base_url: null,
    batch_size: 100,
    storage_file: "embeddings.sqlite3",
  },
};

const state = {
  defaults: null,
  configInfo: null,
  configDraft: null,
  registryDatasets: [],
  configDatasetIds: new Set(),
  datasetDrafts: {},
  selectedDatasetIds: new Set(),
  selectedQueryIds: new Set(),
  pendingDatasetId: "",
  pendingDatasetLoadedId: "",
  pendingQueryIds: new Set(),
  datasetPickerOpen: false,
  queryDropdownOpen: false,
  pendingDatasetLoading: false,
  activeDatasetQueries: [],
  datasetSource: "config",
  activeView: "config",
  activePage: "workbench",
  activeConfigSection: "project",
  activeDatasetId: null,
  configMaterialized: false,
  selectedPaper: "paper",
  paperExperiments: [],
  paperExperimentOutputRoot: "",
  lastPayload: {},
  resultLibrary: [],
  resultsUnavailable: "",
  resultsRenderLimit: RESULTS_RENDER_BATCH,
  currentRunOutputsReady: false,
  runInFlight: false,
  pendingDeleteResultPath: "",
  consoleRun: null,
  consoleEventSource: null,
  consoleElapsedTimer: null,
};

const COMMON_RUN_STEPS = [
  { id: "validate_request", label: "Validate request" },
  { id: "materialize_config", label: "Materialize config" },
  { id: "resolve_datasets", label: "Resolve datasets" },
  { id: "prepare_runtime", label: "Prepare runtime" },
];

const STAGE_RUN_STEPS = {
  preprocessing: "Run preprocessing",
  schema_refinement: "Run schema refinement",
  data_extraction: "Run data extraction",
};

const TAIL_RUN_STEPS = [
  { id: "evaluation", label: "Run evaluation" },
  { id: "collect_optimization_metrics", label: "Collect optimization metrics" },
  { id: "refresh_outputs", label: "Refresh outputs" },
];

const DEFAULT_OPTIMIZATIONS = [
  {
    id: "extraction",
    title: "Extraction",
    status: "waiting",
    message: "Waiting for extraction artifacts.",
    metrics: {},
  },
];

const paperCatalog = {
  paper: {
    title: "ReDD Paper",
    description: "The main ReDD paper describing relational data extraction over document collections.",
    href: "/assets/papers/2025_ReDD.pdf?preview=1",
    downloadHref: "/assets/papers/2025_ReDD.pdf",
  },
  "tech-report": {
    title: "ReDD Technical Report",
    description: "A longer technical report with additional context and implementation detail.",
    href: "/assets/papers/2025_ReDD_TechReport.pdf?preview=1",
    downloadHref: "/assets/papers/2025_ReDD_TechReport.pdf",
  },
};

function selectedStages() {
  return Array.from(document.querySelectorAll('input[name="stages"]:checked')).map(
    (item) => item.value,
  );
}

function selectedDatasetIds() {
  return Array.from(state.selectedDatasetIds);
}

function selectedQueryIds() {
  return Array.from(state.selectedQueryIds);
}

function hasPendingRunDataset() {
  return (
    state.datasetSource === "all" &&
    Boolean(state.pendingDatasetId) &&
    state.pendingDatasetLoadedId === state.pendingDatasetId
  );
}

function pendingDatasetRequiresQueries() {
  return hasPendingRunDataset() && state.activeDatasetQueries.length > 0;
}

function runDatasetIds() {
  if (state.datasetSource === "all") {
    if (!hasPendingRunDataset()) {
      return [];
    }
    if (pendingDatasetRequiresQueries() && !state.pendingQueryIds.size) {
      return [];
    }
    return [state.pendingDatasetId];
  }
  return selectedDatasetIds();
}

function runQueryIds() {
  if (state.datasetSource === "all") {
    return hasPendingRunDataset() ? Array.from(state.pendingQueryIds) : [];
  }
  return configuredQueryIds();
}

function runQueryLabel() {
  const datasetIds = runDatasetIds();
  if (!datasetIds.length) {
    return "-";
  }
  const queryIds = runQueryIds();
  return queryIds.length ? String(queryIds.length) : "all";
}

function configuredQueryIds() {
  return Array.from(
    new Set(
      selectedDatasetIds().flatMap((datasetId) => {
        const queryIds = ensureDatasetDraft(datasetId).query_ids;
        return Array.isArray(queryIds) ? queryIds : [];
      }),
    ),
  );
}

function configDatasetIds() {
  return Array.from(state.configDatasetIds);
}

function clonePlain(value, fallback) {
  return JSON.parse(JSON.stringify(value == null ? fallback : value));
}

function fieldString(value, fallback = "") {
  return value == null ? fallback : String(value);
}

function nullableText(control) {
  const value = control.value.trim();
  return value ? value : null;
}

function controlTitleValue(control) {
  if (!control || control.type === "password" || control.type === "checkbox") {
    return "";
  }
  if (control.tagName === "SELECT") {
    return control.selectedOptions?.[0]?.textContent?.trim() || control.value || "";
  }
  return String(control.value || "").trim();
}

function syncControlTitle(control) {
  const value = controlTitleValue(control);
  if (value) {
    control.title = value;
  } else {
    control.removeAttribute("title");
  }
}

function syncControlTitles(root = document) {
  root.querySelectorAll("input, select, textarea").forEach((control) => {
    syncControlTitle(control);
    if (control.tagName === "TEXTAREA") {
      autoGrowTextarea(control);
    }
  });
}

function textareaAutoGrowLimit(textarea) {
  const viewportHeight = Math.max(window.innerHeight || 0, 320);
  const datasetField = textarea.dataset?.datasetField || "";
  if (textarea.id === "config-stages-json") {
    return Math.min(Math.max(viewportHeight * 0.78, 320), 720);
  }
  if (datasetField === "loader_options" || datasetField === "split") {
    return Math.min(Math.max(viewportHeight * 0.66, 260), 560);
  }
  if (datasetField === "root" || datasetField === "query_ids") {
    return Math.min(Math.max(viewportHeight * 0.52, 180), 440);
  }
  return Math.min(Math.max(viewportHeight * 0.46, 150), 380);
}

function autoGrowTextarea(textarea) {
  if (!textarea || textarea.tagName !== "TEXTAREA" || !textarea.isConnected) {
    return;
  }
  if (textarea.readOnly && textarea.style.position === "fixed") {
    return;
  }
  if (window.getComputedStyle(textarea).display === "none") {
    return;
  }
  const limit = textareaAutoGrowLimit(textarea);
  textarea.style.height = "auto";
  textarea.style.height = `${Math.min(textarea.scrollHeight + 2, limit)}px`;
  textarea.style.overflowY = textarea.scrollHeight > limit ? "auto" : "hidden";
}

function autoGrowTextareas(root = document) {
  root.querySelectorAll("textarea").forEach(autoGrowTextarea);
}

function scheduleAutoGrowTextareas(root = document) {
  autoGrowTextareas(root);
  requestAnimationFrame(() => autoGrowTextareas(root));
  window.setTimeout(() => autoGrowTextareas(root), 0);
}

async function copyTextToClipboard(text) {
  if (navigator.clipboard?.writeText) {
    try {
      await navigator.clipboard.writeText(text);
      return true;
    } catch (_) {
      // Fall through to the legacy copy path used by restricted browser shells.
    }
  }

  const textarea = document.createElement("textarea");
  textarea.value = text;
  textarea.setAttribute("readonly", "");
  textarea.style.position = "fixed";
  textarea.style.left = "-9999px";
  textarea.style.top = "0";
  document.body.appendChild(textarea);
  textarea.select();
  try {
    return document.execCommand("copy");
  } catch (_) {
    return false;
  } finally {
    textarea.remove();
  }
}

function showCopyFeedback(button, copied) {
  const previous = button.textContent;
  button.textContent = copied ? "Copied" : "Copy unavailable";
  window.setTimeout(() => {
    button.textContent = previous || "Copy";
  }, 1400);
}

function optionalInteger(control, fallback = null) {
  const value = control.value.trim();
  if (!value) {
    return fallback;
  }
  const parsed = Number.parseInt(value, 10);
  return Number.isNaN(parsed) ? fallback : parsed;
}

function optionalNumber(control, fallback = null) {
  const value = control.value.trim();
  if (!value) {
    return fallback;
  }
  const parsed = Number.parseFloat(value);
  return Number.isNaN(parsed) ? fallback : parsed;
}

function defaultConfigDraft(info = null) {
  return {
    project: clonePlain(info?.project, { name: "redd-web-demo", seed: 42 }),
    runtime: clonePlain(info?.runtime, {
      output_dir: "outputs/demo",
      log_dir: "logs",
      output_layout: "dataset_stage",
      artifact_id: "web-demo-v1",
      console_log_level: "WARNING",
      force_rerun: false,
    }),
    models: clonePlain(info?.models || state.defaults?.models, DEFAULT_MODELS),
    stageConfigs: clonePlain(info?.stage_configs, {
      data_extraction: {
        enabled: true,
        schema_source: "ground_truth",
        oracle: "ground_truth",
      },
    }),
    experimentId: info?.experiment || "demo",
  };
}

function prettyJson(value) {
  return JSON.stringify(value, null, 2);
}

function parseJsonTextarea(textarea, fallback) {
  const text = textarea.value.trim();
  if (!text) {
    textarea.classList.remove("invalid");
    return clonePlain(fallback, {});
  }
  try {
    const parsed = JSON.parse(text);
    textarea.classList.remove("invalid");
    return parsed;
  } catch {
    textarea.classList.add("invalid");
    return clonePlain(fallback, {});
  }
}

function modelCatalog() {
  return (
    state.configInfo?.model_catalog ||
    state.defaults?.model_catalog || { llm: {}, embedding: {} }
  );
}

function modelControls(kind) {
  if (kind === "embedding") {
    return {
      enabled: configEmbeddingEnabled,
      provider: configEmbeddingProvider,
      model: configEmbeddingModel,
      customWrap: configEmbeddingCustomWrap,
      customModel: configEmbeddingCustomModel,
      apiKeyEnv: configEmbeddingApiKeyEnv,
      baseUrl: configEmbeddingBaseUrl,
      batchSize: configEmbeddingBatchSize,
      storageFile: configEmbeddingStorageFile,
    };
  }
  return {
    enabled: configLlmEnabled,
    provider: configLlmProvider,
    model: configLlmModel,
    customWrap: configLlmCustomWrap,
    customModel: configLlmCustomModel,
    apiKeyEnv: configLlmApiKeyEnv,
    baseUrl: configLlmBaseUrl,
    structuredBackend: configLlmStructuredBackend,
    maxRetries: configLlmMaxRetries,
    waitTime: configLlmWaitTime,
    temperature: configLlmTemperature,
    topP: configLlmTopP,
    maxTokens: configLlmMaxTokens,
    localModelPath: configLlmLocalModelPath,
  };
}

function catalogProviders(kind) {
  return Object.keys(modelCatalog()?.[kind] || {});
}

function catalogModels(kind, provider) {
  return modelCatalog()?.[kind]?.[provider] || [];
}

function firstCatalogProvider(kind) {
  return catalogProviders(kind)[0] || "openai";
}

function firstCatalogModel(kind, provider) {
  return catalogModels(kind, provider)[0]?.id || "";
}

function modelLabel(entry) {
  return entry.label && entry.label !== entry.id ? `${entry.label} (${entry.id})` : entry.id;
}

function fillProviderSelect(kind, selectedProvider) {
  const controls = modelControls(kind);
  const providers = catalogProviders(kind);
  const provider = selectedProvider || firstCatalogProvider(kind);
  if (provider && !providers.includes(provider)) {
    providers.push(provider);
  }
  controls.provider.innerHTML = providers
    .map(
      (item) =>
        `<option value="${escapeHtml(item)}" ${item === provider ? "selected" : ""}>${escapeHtml(
          item,
        )}</option>`,
    )
    .join("");
  controls.provider.value = provider;
}

function fillModelSelect(kind, provider, selectedModel = "") {
  const controls = modelControls(kind);
  const entries = catalogModels(kind, provider);
  const knownModels = entries.map((entry) => entry.id);
  const hasSelected = selectedModel && knownModels.includes(selectedModel);
  const selectedValue = hasSelected
    ? selectedModel
    : selectedModel
      ? CUSTOM_MODEL_VALUE
      : firstCatalogModel(kind, provider);
  controls.model.innerHTML = [
    ...entries.map(
      (entry) =>
        `<option value="${escapeHtml(entry.id)}" ${entry.id === selectedValue ? "selected" : ""}>${escapeHtml(
          modelLabel(entry),
        )}</option>`,
    ),
    `<option value="${CUSTOM_MODEL_VALUE}" ${
      selectedValue === CUSTOM_MODEL_VALUE ? "selected" : ""
    }>Custom...</option>`,
  ].join("");
  controls.model.value = selectedValue || CUSTOM_MODEL_VALUE;
  controls.customModel.value = hasSelected ? "" : selectedModel || "";
  updateModelCustomField(kind);
}

function updateModelCustomField(kind) {
  const controls = modelControls(kind);
  const usesCustomModel = controls.model.value === CUSTOM_MODEL_VALUE;
  controls.customWrap.hidden = !usesCustomModel;
  controls.customModel.disabled =
    controls.customWrap.hidden ||
    controls.customModel.closest("article").classList.contains("disabled");
}

function updateModelControlState(kind) {
  const controls = modelControls(kind);
  const enabled = controls.enabled.checked;
  const card = controls.enabled.closest(".model-config-card");
  card.classList.toggle("disabled", !enabled);
  [
    controls.provider,
    controls.model,
    controls.apiKeyEnv,
    controls.baseUrl,
    controls.structuredBackend,
    controls.maxRetries,
    controls.waitTime,
    controls.temperature,
    controls.topP,
    controls.maxTokens,
    controls.localModelPath,
    controls.batchSize,
    controls.storageFile,
  ]
    .filter(Boolean)
    .forEach((control) => {
      control.disabled = !enabled;
    });
  updateModelCustomField(kind);
}

function setModelAdvancedControls(kind, config) {
  const controls = modelControls(kind);
  if (kind === "embedding") {
    const defaults = DEFAULT_MODELS.embedding;
    controls.apiKeyEnv.value = fieldString(config?.api_key_env, defaults.api_key_env);
    controls.baseUrl.value = fieldString(config?.base_url, "");
    controls.batchSize.value = fieldString(config?.batch_size, defaults.batch_size);
    controls.storageFile.value = fieldString(config?.storage_file, defaults.storage_file);
    return;
  }

  const defaults = DEFAULT_MODELS.llm;
  controls.apiKeyEnv.value = fieldString(config?.api_key_env, defaults.api_key_env);
  controls.baseUrl.value = fieldString(config?.base_url, "");
  controls.structuredBackend.value = config?.structured_backend || defaults.structured_backend;
  controls.maxRetries.value = fieldString(config?.max_retries, defaults.max_retries);
  controls.waitTime.value = fieldString(config?.wait_time, defaults.wait_time);
  controls.temperature.value = fieldString(config?.temperature, "");
  controls.topP.value = fieldString(config?.top_p, "");
  controls.maxTokens.value = fieldString(config?.max_tokens, "");
  controls.localModelPath.value = fieldString(config?.local_model_path, "");
}

function setModelControlsFromDraft(kind, config) {
  const controls = modelControls(kind);
  const isEnabled = Boolean(config && config.provider !== "none" && config.enabled !== false);
  const provider = config?.provider || firstCatalogProvider(kind);
  const model = config?.model || firstCatalogModel(kind, provider);
  controls.enabled.checked = isEnabled;
  fillProviderSelect(kind, provider);
  fillModelSelect(kind, provider, model);
  setModelAdvancedControls(kind, config);
  updateModelControlState(kind);
}

function readModelConfigFromControls(kind, previousConfig) {
  const controls = modelControls(kind);
  if (!controls.enabled.checked) {
    return null;
  }
  const provider = controls.provider.value || firstCatalogProvider(kind);
  const model =
    controls.model.value === CUSTOM_MODEL_VALUE
      ? controls.customModel.value.trim()
      : controls.model.value;
  const base = clonePlain(previousConfig, {});
  const next = {
    ...base,
    provider,
    model: model || firstCatalogModel(kind, provider),
  };
  if (kind === "embedding") {
    next.api_key_env = nullableText(controls.apiKeyEnv);
    next.base_url = nullableText(controls.baseUrl);
    next.batch_size = optionalInteger(controls.batchSize, DEFAULT_MODELS.embedding.batch_size);
    next.storage_file = controls.storageFile.value.trim() || DEFAULT_MODELS.embedding.storage_file;
    next.enabled = true;
  } else {
    next.api_key_env = nullableText(controls.apiKeyEnv);
    next.base_url = nullableText(controls.baseUrl);
    next.structured_backend = controls.structuredBackend.value || DEFAULT_MODELS.llm.structured_backend;
    next.max_retries = optionalInteger(controls.maxRetries, DEFAULT_MODELS.llm.max_retries);
    next.wait_time = optionalNumber(controls.waitTime, DEFAULT_MODELS.llm.wait_time);
    next.temperature = optionalNumber(controls.temperature, null);
    next.top_p = optionalNumber(controls.topP, null);
    next.max_tokens = optionalInteger(controls.maxTokens, null);
    next.local_model_path = nullableText(controls.localModelPath);
  }
  return next;
}

function handleModelProviderChange(kind) {
  const controls = modelControls(kind);
  fillModelSelect(kind, controls.provider.value, firstCatalogModel(kind, controls.provider.value));
  updateConfigDraftFromEditor();
}

function handleModelEnabledChange(kind) {
  const controls = modelControls(kind);
  if (controls.enabled.checked && !controls.model.value) {
    fillModelSelect(kind, controls.provider.value, firstCatalogModel(kind, controls.provider.value));
  }
  updateModelControlState(kind);
  updateConfigDraftFromEditor();
}

function setConfigEditorFromDraft() {
  const draft = state.configDraft || defaultConfigDraft(state.configInfo);
  configProjectName.value = draft.project?.name || "";
  configProjectSeed.value = draft.project?.seed ?? "";
  configExperimentId.value = draft.experimentId || "";
  configOutputDir.value = draft.runtime?.output_dir || "";
  configLogDir.value = draft.runtime?.log_dir || "";
  configArtifactId.value = draft.runtime?.artifact_id || "";
  configOutputLayout.value = draft.runtime?.output_layout || "dataset_stage";
  configConsoleLogLevel.value = draft.runtime?.console_log_level || "WARNING";
  forceRerun.checked = Boolean(draft.runtime?.force_rerun);
  setModelControlsFromDraft("llm", draft.models?.llm);
  setModelControlsFromDraft("embedding", draft.models?.embedding);
  configStagesJson.value = prettyJson(draft.stageConfigs || {});
  setOptimizationControlsFromStageConfigs(draft.stageConfigs || {});
  syncControlTitle(configPath);
  syncControlTitle(experiment);
  syncControlTitles(configWorkspace);
}

function apiKeyStatusLabel(status) {
  if (apiKey.value.trim()) {
    return "Override";
  }
  if (status?.configured) {
    return "Ready";
  }
  return "";
}

function renderApiKeyStatus(status = state.configInfo?.api_key_status || state.defaults?.api_key_status) {
  if (!apiKeyStatus) {
    return;
  }
  const configured = Boolean(apiKey.value.trim()) || Boolean(status?.configured);
  const label = apiKeyStatusLabel(status);
  apiKeyStatus.textContent = label;
  apiKeyStatus.hidden = !label;
  apiKeyStatus.classList.toggle("configured", configured);
}

function readConfigDraftFromEditor() {
  const previous = state.configDraft || defaultConfigDraft(state.configInfo);
  const seedValue = Number.parseInt(configProjectSeed.value, 10);
  return {
    project: {
      name: configProjectName.value.trim() || "redd-web-demo",
      seed: Number.isNaN(seedValue) ? 42 : seedValue,
    },
    runtime: {
      output_dir: configOutputDir.value.trim() || "outputs/demo",
      log_dir: configLogDir.value.trim() || "logs",
      output_layout: configOutputLayout.value || "dataset_stage",
      artifact_id: configArtifactId.value.trim() || "web-demo-v1",
      console_log_level: configConsoleLogLevel.value || "WARNING",
      force_rerun: Boolean(forceRerun?.checked),
    },
    models: {
      llm: readModelConfigFromControls("llm", previous.models?.llm),
      embedding: readModelConfigFromControls("embedding", previous.models?.embedding),
    },
    stageConfigs: parseJsonTextarea(configStagesJson, previous.stageConfigs || {}),
    experimentId: configExperimentId.value.trim() || experiment.value.trim() || "demo",
  };
}

function updateConfigDraftFromEditor() {
  state.configDraft = readConfigDraftFromEditor();
  state.configMaterialized = true;
  renderGeneratedConfig();
  renderOverview();
}

function yamlKey(value) {
  const text = String(value == null ? "" : value);
  return /^[A-Za-z0-9_.\/-]+$/.test(text) ? text : JSON.stringify(text);
}

function yamlScalar(value) {
  if (value == null) {
    return "null";
  }
  if (typeof value === "number" || typeof value === "boolean") {
    return String(value);
  }
  const text = String(value);
  return /^[A-Za-z0-9_.\/-]+$/.test(text) ? text : JSON.stringify(text);
}

function toYaml(value, indent = 0) {
  const pad = " ".repeat(indent);
  if (Array.isArray(value)) {
    if (!value.length) {
      return "[]";
    }
    return value
      .map((item) => {
        if (item && typeof item === "object") {
          return `${pad}-\n${toYaml(item, indent + 2)}`;
        }
        return `${pad}- ${yamlScalar(item)}`;
      })
      .join("\n");
  }
  if (value && typeof value === "object") {
    const entries = Object.entries(value);
    if (!entries.length) {
      return "{}";
    }
    return entries
      .map(([key, nested]) => {
        if (nested && typeof nested === "object") {
          if (Array.isArray(nested) && !nested.length) {
            return `${pad}${yamlKey(key)}: []`;
          }
          if (!Array.isArray(nested) && !Object.keys(nested).length) {
            return `${pad}${yamlKey(key)}: {}`;
          }
          return `${pad}${yamlKey(key)}:\n${toYaml(nested, indent + 2)}`;
        }
        return `${pad}${yamlKey(key)}: ${yamlScalar(nested)}`;
      })
      .join("\n");
  }
  return `${pad}${yamlScalar(value)}`;
}

function normalizeDatasetRoot(path) {
  const text = String(path || "").replace(/\\/g, "/");
  const marker = "/dataset/";
  const index = text.indexOf(marker);
  if (index >= 0) {
    return `dataset/${text.slice(index + marker.length)}`;
  }
  return text;
}

function stageConfigValue(stageConfigs, stageId, key, fallback = null) {
  const value = stageConfigs?.[stageId]?.[key];
  return value && typeof value === "object" ? value : fallback;
}

function setOptimizationControlsFromStageConfigs(stageConfigs = {}) {
  const schemaFilter =
    stageConfigValue(stageConfigs, "schema_refinement", "doc_filter") ||
    stageConfigValue(stageConfigs, "data_extraction", "doc_filter") ||
    {};
  configOptDocFilterEnabled.checked = Boolean(schemaFilter.enabled);

  const adaptive =
    stageConfigValue(stageConfigs, "schema_refinement", "adaptive_sampling") ||
    stageConfigValue(stageConfigs, "preprocessing", "adaptive_sampling") ||
    {};
  configOptSchemaAdaptiveEnabled.checked = Boolean(adaptive.enabled);
  configOptSchemaAdaptiveThreshold.value = fieldString(
    adaptive.entropy_threshold ?? adaptive.theta,
    "0.05",
  );
  configOptSchemaAdaptiveStreak.value = fieldString(adaptive.streak_limit ?? adaptive.m, "3");
  configOptSchemaAdaptiveMinDocs.value = fieldString(adaptive.min_docs ?? adaptive.n_min, "3");

  const proxy = stageConfigValue(stageConfigs, "data_extraction", "proxy_runtime") || {};
  configOptProxyEnabled.checked = Boolean(proxy.enabled);
  configOptProxyMode.value = proxy.predicate_proxy_mode || "pretrained";
  configOptProxyModel.value =
    proxy.finetuned_model || "knowledgator/gliclass-small-v1.0";
  configOptProxyLearned.checked = proxy.use_learned_proxies !== false;
  configOptProxyEmbedding.checked = proxy.use_embedding_proxies !== false;
  configOptProxyFallback.checked = Boolean(proxy.allow_embedding_fallback);
  configOptProxyEpochs.value = fieldString(proxy.finetuned_epochs, "0");
  configOptJoinEnabled.checked = proxy.use_join_resolution !== false && Boolean(proxy.enabled);
  updateOptimizationControlState();
}

function updateOptimizationControlState() {
  const stageIds = selectedStages();
  const canUseDocFilter =
    stageIds.includes("schema_refinement") || stageIds.includes("data_extraction");
  const canUseSchemaAdaptive = stageIds.includes("schema_refinement");
  const canUseProxy = stageIds.includes("data_extraction");

  if (!canUseDocFilter) {
    configOptDocFilterEnabled.checked = false;
  }
  if (!canUseSchemaAdaptive) {
    configOptSchemaAdaptiveEnabled.checked = false;
  }
  if (!canUseProxy) {
    configOptProxyEnabled.checked = false;
  }

  configOptDocFilterEnabled.disabled = !canUseDocFilter;
  configOptSchemaAdaptiveEnabled.disabled = !canUseSchemaAdaptive;
  configOptProxyEnabled.disabled = !canUseProxy;

  document
    .querySelector('[data-optimization-card="doc_filter"]')
    ?.classList.toggle("unavailable", !canUseDocFilter);
  document
    .querySelector('[data-optimization-card="schema_adaptive"]')
    ?.classList.toggle("unavailable", !canUseSchemaAdaptive);
  document
    .querySelector('[data-optimization-card="proxy_runtime"]')
    ?.classList.toggle("unavailable", !canUseProxy);

  const docFilterEnabled = canUseDocFilter && configOptDocFilterEnabled.checked;
  document
    .querySelector('[data-optimization-card="doc_filter"]')
    ?.classList.toggle("active", docFilterEnabled);
  const adaptiveEnabled = canUseSchemaAdaptive && configOptSchemaAdaptiveEnabled.checked;
  document
    .querySelector('[data-optimization-card="schema_adaptive"]')
    ?.classList.toggle("active", adaptiveEnabled);
  [
    configOptSchemaAdaptiveThreshold,
    configOptSchemaAdaptiveStreak,
    configOptSchemaAdaptiveMinDocs,
  ].forEach((control) => {
    control.disabled = !adaptiveEnabled;
  });
  const proxyEnabled = canUseProxy && configOptProxyEnabled.checked;
  document
    .querySelector('[data-optimization-card="proxy_runtime"]')
    ?.classList.toggle("active", proxyEnabled);
  [
    configOptProxyMode,
    configOptProxyModel,
    configOptProxyLearned,
    configOptProxyEmbedding,
    configOptProxyFallback,
    configOptProxyEpochs,
    configOptJoinEnabled,
  ].forEach((control) => {
    control.disabled = !proxyEnabled;
  });
  if (!proxyEnabled) {
    configOptJoinEnabled.checked = false;
  }
}

function removeOptimizationKeys(stages, stageIds) {
  ["preprocessing", "schema_refinement"].forEach((stageId) => {
    if (!stageIds.includes(stageId) || !stages[stageId]) {
      return;
    }
    delete stages[stageId].adaptive_sampling;
    delete stages[stageId].doc_filter;
  });
  if (stageIds.includes("data_extraction") && stages.data_extraction) {
    delete stages.data_extraction.doc_filter;
    delete stages.data_extraction.proxy_runtime;
    delete stages.data_extraction.alpha_allocation;
  }
}

function applyOptimizationControlsToStages(stages, stageIds) {
  const existingDocFilter =
    stageConfigValue(stages, "schema_refinement", "doc_filter") ||
    stageConfigValue(stages, "data_extraction", "doc_filter") ||
    {};
  const existingProxy = stageConfigValue(stages, "data_extraction", "proxy_runtime") || {};
  removeOptimizationKeys(stages, stageIds);
  if (configOptDocFilterEnabled.checked) {
    const fullDocFilter = {
      ...existingDocFilter,
      enabled: true,
      filter_type: existingDocFilter.filter_type || "schema_relevance",
    };
    if (stageIds.includes("schema_refinement") && stages.schema_refinement) {
      stages.schema_refinement.doc_filter = fullDocFilter;
    }
    if (stageIds.includes("data_extraction") && stages.data_extraction) {
      stages.data_extraction.doc_filter = stageIds.includes("schema_refinement")
        ? { enabled: true }
        : fullDocFilter;
    }
  }
  if (configOptSchemaAdaptiveEnabled.checked) {
    const adaptive = {
      enabled: true,
      algorithm: "entropy",
      entropy_threshold: optionalNumber(configOptSchemaAdaptiveThreshold, 0.05),
      streak_limit: optionalInteger(configOptSchemaAdaptiveStreak, 3),
      min_docs: optionalInteger(configOptSchemaAdaptiveMinDocs, 3),
    };
    ["preprocessing", "schema_refinement"].forEach((stageId) => {
      if (stageIds.includes(stageId) && stages[stageId]) {
        stages[stageId].adaptive_sampling = adaptive;
      }
    });
  }
  if (configOptProxyEnabled.checked && stageIds.includes("data_extraction") && stages.data_extraction) {
    stages.data_extraction.proxy_runtime = {
      ...existingProxy,
      enabled: true,
      predicate_proxy_mode: configOptProxyMode.value || "pretrained",
      use_embedding_proxies: Boolean(configOptProxyEmbedding.checked),
      use_learned_proxies: Boolean(configOptProxyLearned.checked),
      use_finetuned_learned_proxies: Boolean(configOptProxyLearned.checked),
      use_join_resolution: Boolean(configOptJoinEnabled.checked),
      finetuned_model:
        configOptProxyModel.value.trim() ||
        "knowledgator/gliclass-small-v1.0",
      finetuned_epochs: optionalInteger(configOptProxyEpochs, 0),
      allow_embedding_fallback: Boolean(configOptProxyFallback.checked),
    };
    stages.data_extraction.alpha_allocation = { enabled: false };
  }
}

function enabledOptimizationPlaceholders() {
  const items = [];
  if (configOptDocFilterEnabled.checked) {
    items.push({
      id: "doc_filter",
      title: "Document Filter",
      status: "waiting",
      message: "Waiting for filter artifacts.",
      metrics: {},
    });
  }
  if (configOptSchemaAdaptiveEnabled.checked) {
    items.push({
      id: "schema_adaptive",
      title: "Schema Adaptive Sampling",
      status: "waiting",
      message: "Waiting for schema adaptive sampling stats.",
      metrics: {},
    });
  }
  if (configOptProxyEnabled.checked) {
    items.push({
      id: "proxy_runtime",
      title: "Proxy Runtime",
      status: "waiting",
      message: "Waiting for proxy decisions.",
      metrics: {},
    });
    if (configOptJoinEnabled.checked) {
      items.push({
        id: "join_proxy",
        title: "Join-aware Filtering",
        status: "waiting",
        message: "Waiting for join-aware proxy decisions.",
        metrics: {},
      });
    }
  }
  items.push(...clonePlain(DEFAULT_OPTIMIZATIONS, []));
  return items;
}

function defaultStageConfig(stageId, selectedStagesForRun) {
  if (stageId === "schema_refinement") {
    return {
      enabled: true,
      source_stage: selectedStagesForRun.includes("preprocessing") ? "preprocessing" : null,
      oracle: "llm",
    };
  }
  if (stageId === "data_extraction") {
    return {
      enabled: true,
      schema_source: selectedStagesForRun.includes("schema_refinement")
        ? "schema_refinement"
        : "ground_truth",
      oracle: "ground_truth",
    };
  }
  return { enabled: true, oracle: "llm" };
}

function sourceDatasetForId(datasetId) {
  const configDataset = (state.configInfo?.datasets || []).find((item) => item.id === datasetId);
  const registryDataset = state.registryDatasets.find((item) => item.id === datasetId);
  return configDataset || registryDataset || { id: datasetId };
}

function ensureDatasetDraft(datasetId) {
  if (state.datasetDrafts[datasetId]) {
    return state.datasetDrafts[datasetId];
  }
  const source = sourceDatasetForId(datasetId);
  const config = {
    loader: source.loader || "hf_manifest",
    root: normalizeDatasetRoot(source.root || source.path || ""),
    loader_options: clonePlain(source.loader_options, { manifest: "manifest.yaml" }),
    split: clonePlain(source.split, { train_count: 0 }),
  };
  if (Array.isArray(source.query_ids) && source.query_ids.length) {
    config.query_ids = [...source.query_ids];
  }
  state.datasetDrafts[datasetId] = config;
  return config;
}

function datasetConfigForId(datasetId) {
  const draft = clonePlain(ensureDatasetDraft(datasetId), {});
  if (Array.isArray(draft.query_ids) && !draft.query_ids.length) {
    delete draft.query_ids;
  }
  if (!draft.loader_options) {
    draft.loader_options = { manifest: "manifest.yaml" };
  }
  if (!draft.split) {
    draft.split = { train_count: 0 };
  }
  draft.root = normalizeDatasetRoot(draft.root || "");
  draft.loader = draft.loader || "hf_manifest";
  return draft;
}

function datasetConfigForRun(datasetId) {
  const draft = datasetConfigForId(datasetId);
  if (hasPendingRunDataset() && datasetId === state.pendingDatasetId) {
    if (state.pendingQueryIds.size) {
      draft.query_ids = Array.from(state.pendingQueryIds);
    } else {
      delete draft.query_ids;
    }
  }
  return draft;
}

function generatedConfigObject(options = {}) {
  const draft = state.configDraft || defaultConfigDraft(state.configInfo);
  const forRun = Boolean(options.forRun);
  const datasetIds = forRun ? runDatasetIds() : selectedDatasetIds();
  const configuredDatasetIds = forRun && hasPendingRunDataset() ? datasetIds : configDatasetIds();
  const stageIds = selectedStages();
  const experimentId = draft.experimentId || experiment.value.trim() || state.configInfo?.experiment || "demo";
  const stages = {};
  stageIds.forEach((stageId) => {
    stages[stageId] = clonePlain(
      draft.stageConfigs?.[stageId] ||
      state.configInfo?.stage_configs?.[stageId] ||
      defaultStageConfig(stageId, stageIds),
      {},
    );
  });
  applyOptimizationControlsToStages(stages, stageIds);
  const datasets = {};
  configuredDatasetIds.forEach((datasetId) => {
    datasets[datasetId] = forRun ? datasetConfigForRun(datasetId) : datasetConfigForId(datasetId);
  });
  return {
    config_version: "2.1.1",
    project: draft.project,
    runtime: draft.runtime,
    models: draft.models,
    datasets,
    stages,
    experiments: {
      [experimentId]: {
        datasets: datasetIds,
        stages: stageIds,
      },
    },
  };
}

function setStatus(label, mode) {
  statusPill.textContent = label;
  statusPill.className = `status-pill ${mode}`;
}

function revealErrorPanel() {
  window.requestAnimationFrame(() => {
    errorPanel.scrollIntoView({ block: "center", inline: "nearest" });
    errorPanel.focus({ preventScroll: true });
  });
}

function setError(message) {
  if (!message) {
    errorPanel.hidden = true;
    errorPanel.textContent = "";
    return;
  }
  errorPanel.hidden = false;
  errorPanel.textContent = message;
  revealErrorPanel();
}

function escapeHtml(value) {
  return String(value == null ? "" : value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function truncate(value, length = 170) {
  const text = String(value == null ? "" : value);
  return text.length > length ? `${text.slice(0, length)}...` : text;
}

async function fetchJson(url, options) {
  const response = await fetch(url, options);
  const text = await response.text();
  let data = null;
  if (text) {
    try {
      data = JSON.parse(text);
    } catch (error) {
      if (response.ok) {
        throw new Error(`Expected JSON response from ${url}. ${error.message}`);
      }
    }
  }
  if (!response.ok) {
    const fallback = text ? `: ${truncate(text, 500)}` : "";
    throw new Error(data ? errorMessage(data) : `HTTP ${response.status} ${response.statusText}${fallback}`);
  }
  return data;
}

function stageCount(value) {
  if (Array.isArray(value)) {
    return value.length;
  }
  if (value && typeof value === "object") {
    return Object.keys(value).length;
  }
  return value == null ? 0 : 1;
}

function formatMetric(value, { percent = false } = {}) {
  if (!percent && typeof value === "string") {
    return value;
  }
  if (value == null || Number.isNaN(Number(value))) {
    return percent ? "--%" : "-";
  }
  const number = Number(value);
  if (percent) {
    const percentValue = Math.abs(number) <= 1 ? number * 100 : number;
    return `${percentValue.toFixed(1)}%`;
  }
  return Number.isInteger(number) ? String(number) : number.toFixed(2);
}

function formatRunDuration(seconds) {
  const safeSeconds = Math.max(0, Number(seconds) || 0);
  const minutes = Math.floor(safeSeconds / 60);
  const remainder = Math.floor(safeSeconds % 60);
  return `${String(minutes).padStart(2, "0")}:${String(remainder).padStart(2, "0")}`;
}

function buildConsoleSteps() {
  const stageSteps = selectedStages().map((stage) => ({
    id: stage,
    label: STAGE_RUN_STEPS[stage] || `Run ${stage}`,
  }));
  return [...COMMON_RUN_STEPS, ...stageSteps, ...TAIL_RUN_STEPS];
}

function resetConsoleRun(stepsOverride = null) {
  if (state.consoleEventSource) {
    state.consoleEventSource.close();
    state.consoleEventSource = null;
  }
  if (state.consoleElapsedTimer) {
    window.clearInterval(state.consoleElapsedTimer);
    state.consoleElapsedTimer = null;
  }
  const steps = (stepsOverride || buildConsoleSteps()).map((step) => ({
    ...step,
    status: "pending",
    message: "",
  }));
  state.consoleRun = {
    id: "",
    status: "queued",
    currentStep: "-",
    startedAt: Date.now(),
    elapsedSeconds: 0,
    steps,
    logs: [],
    progresses: {},
    optimizations: enabledOptimizationPlaceholders(),
    optimizationActivities: [],
    optimizationActivityKeys: new Set(),
    optimizationLogKeys: new Set(),
    logDedupeKeys: new Set(),
    droppedLogCount: 0,
    evaluation: null,
    error: null,
    lastHeartbeatAt: 0,
  };
  renderConsoleRun();
  renderConsoleLogs();
  state.consoleElapsedTimer = window.setInterval(() => {
    if (!state.consoleRun || !["queued", "running"].includes(state.consoleRun.status)) {
      return;
    }
    state.consoleRun.elapsedSeconds = (Date.now() - state.consoleRun.startedAt) / 1000;
    consoleRunElapsed.textContent = formatRunDuration(state.consoleRun.elapsedSeconds);
    if (
      state.consoleRun.status === "running" &&
      state.consoleRun.currentStep &&
      state.consoleRun.currentStep !== "-" &&
      state.consoleRun.elapsedSeconds - state.consoleRun.lastHeartbeatAt >= 20
    ) {
      state.consoleRun.lastHeartbeatAt = state.consoleRun.elapsedSeconds;
      appendConsoleLog(
        `Still running ${state.consoleRun.currentStep}. Elapsed ${formatRunDuration(
          state.consoleRun.elapsedSeconds,
        )}.`,
      );
    }
  }, 500);
}

function clearCurrentRunPayload() {
  state.lastPayload = {};
  state.currentRunOutputsReady = false;
  summaryExperiment.textContent = "-";
  summaryDatasets.textContent = "-";
  summaryStages.textContent = "-";
  summaryElapsed.textContent = "-";
  rawJson.textContent = "";
  renderStageResults({});
  renderRelationPreview();
  renderOutputResults();
  renderOverview();
}

function renderConsoleRun() {
  const run = state.consoleRun;
  if (!run) {
    consoleEmptyState.hidden = false;
    consoleActive.hidden = true;
    consoleStateLabel.textContent = "No run yet";
    return;
  }
  consoleEmptyState.hidden = true;
  consoleActive.hidden = false;
  consoleStateLabel.textContent = statusLabel(run.status);
  consoleRunId.textContent = run.id || "pending";
  consoleRunStatus.textContent = statusLabel(run.status);
  consoleRunStatus.dataset.status = run.status;
  consoleRunElapsed.textContent = formatRunDuration(run.elapsedSeconds);
  consoleCurrentStep.textContent = run.currentStep || "-";
  renderConsoleSteps();
  renderOptimizationCards();
  renderOptimizationActivityFeed();
  renderEvaluationCards();
  renderConsoleProgress();
}

function statusLabel(status) {
  const labels = {
    queued: "Queued",
    running: "Running",
    completed: "Completed",
    failed: "Failed",
    pending: "Pending",
    started: "Started",
    done: "Done",
    measured: "Measured",
    not_enabled: "Not enabled",
    no_metrics: "No metrics",
    waiting: "Waiting",
  };
  return labels[status] || status || "-";
}

function renderConsoleSteps() {
  const run = state.consoleRun;
  consoleStepper.innerHTML = run.steps
    .map(
      (step) => `
        <article class="console-step ${escapeHtml(step.status)}">
          <span class="console-step-dot" aria-hidden="true"></span>
          <div>
            <strong>${escapeHtml(step.label)}</strong>
            <small>${escapeHtml(step.message || statusLabel(step.status))}</small>
          </div>
          <em>${escapeHtml(statusLabel(step.status))}</em>
        </article>
      `,
    )
    .join("");
}

function renderOptimizationCards() {
  const run = state.consoleRun;
  const visibleOptimizations = run.optimizations.filter(
    (item) => item.status !== "not_enabled" && item.id !== "extraction",
  );
  if (!visibleOptimizations.length) {
    optimizationCards.innerHTML = '<div class="optimization-empty">No enabled optimizations for this run</div>';
    return;
  }
  const total = optimizationPipelineSummary(visibleOptimizations);
  const totalCard = total
    ? `
      <article class="optimization-card optimization-total-card measured">
        <header>
          <div>
            <h5>FastReDD Total</h5>
            <p>${escapeHtml(
              total.offlineOnly
                ? "Offline-only benchmark ablation; GT guard is not deployable."
                : "End-to-end LLM-doc call reduction across enabled optimizers.",
            )}</p>
          </div>
          <span>Measured</span>
        </header>
        <dl>
          <div><dt>BF calls</dt><dd>${escapeHtml(formatMetric(total.before))}</dd></div>
          <div><dt>Fast calls</dt><dd>${escapeHtml(formatMetric(total.after))}</dd></div>
          <div><dt>Saved</dt><dd>${escapeHtml(formatMetric(total.saved))}</dd></div>
          <div><dt>Reduction</dt><dd>${escapeHtml(formatMetric(total.reduction, { percent: true }))}</dd></div>
          ${total.offlineOnly ? '<div><dt>Mode</dt><dd>Offline-only</dd></div>' : ""}
        </dl>
      </article>
    `
    : "";
  optimizationCards.innerHTML = totalCard + visibleOptimizations
    .map((item) => {
      const metrics = formatOptimizationMetrics(item);
      return `
        <article class="optimization-card ${escapeHtml(item.status || "waiting")}">
          <header>
            <div>
              <h5>${escapeHtml(item.title)}</h5>
              <p>${escapeHtml(item.message || "")}</p>
            </div>
            <span>${escapeHtml(statusLabel(item.status))}</span>
          </header>
          <dl>${metrics}</dl>
        </article>
      `;
    })
    .join("");
}

function optimizationPipelineSummary(optimizations) {
  const byId = new Map(optimizations.map((item) => [item.id, item.metrics || {}]));
  const docFilter = byId.get("doc_filter") || {};
  const proxyRuntime = byId.get("proxy_runtime") || {};
  const before = Number(docFilter.llm_doc_calls_before || proxyRuntime.llm_doc_calls_before);
  const after = Number(proxyRuntime.llm_doc_calls_after || docFilter.llm_doc_calls_after);
  if (!Number.isFinite(before) || !Number.isFinite(after) || before <= 0) {
    return null;
  }
  const saved = Math.max(before - after, 0);
  return {
    before,
    after,
    saved,
    reduction: saved / before,
    offlineOnly: Boolean(proxyRuntime.offline_only_gt_guard),
  };
}

function renderOptimizationActivityFeed() {
  const run = state.consoleRun;
  if (!optimizationFeed) {
    return;
  }
  const activities = Array.isArray(run?.optimizationActivities)
    ? run.optimizationActivities.slice(-18)
    : [];
  if (!activities.length) {
    optimizationFeed.innerHTML =
      '<div class="optimization-feed-empty">Per-query savings will appear here while optimizations run.</div>';
    return;
  }
  optimizationFeed.innerHTML = activities.map(renderOptimizationActivityItem).join("");
}

function renderOptimizationActivityItem(activity) {
  const percent = Math.max(0, Math.min(1, Number(activity.reduction) || 0));
  const before = activity.before ?? 0;
  const after = activity.after ?? 0;
  const saved = activity.saved ?? Math.max(before - after, 0);
  const scope = [activity.dataset, activity.queryId, activity.table].filter(Boolean).join(" · ");
  const docPreview = Array.isArray(activity.docIds) && activity.docIds.length
    ? `
      <details class="optimization-feed-docs">
        <summary>${escapeHtml(String(activity.docTotal || activity.docIds.length))} doc ids</summary>
        ${activity.docIds.map((docId) => `<span>${escapeHtml(String(docId))}</span>`).join("")}
        ${activity.docTotal > activity.docIds.length ? `<em>+${activity.docTotal - activity.docIds.length}</em>` : ""}
      </details>
    `
    : "";
  return `
    <article class="optimization-feed-item ${escapeHtml(activity.kind)}">
      <div class="optimization-feed-badge">${escapeHtml(activity.badge)}</div>
      <div class="optimization-feed-main">
        <div class="optimization-feed-title">
          <strong>${escapeHtml(activity.title)}</strong>
          <span>${escapeHtml(scope || "optimization")}</span>
        </div>
        <div class="optimization-feed-stats">
          <span><b>${escapeHtml(String(before))}</b> to <b>${escapeHtml(String(after))}</b></span>
          <em>${escapeHtml(String(saved))} saved</em>
          <strong>${escapeHtml(formatMetric(percent, { percent: true }))}</strong>
        </div>
        <div class="optimization-feed-track" aria-hidden="true">
          <span style="width: ${percent * 100}%"></span>
        </div>
        ${docPreview}
      </div>
    </article>
  `;
}

function renderEvaluationCards() {
  if (!evaluationCards) {
    return;
  }
  const evaluation = state.consoleRun?.evaluation || state.lastPayload?.evaluation;
  if (!evaluation || evaluation.status === "not_enabled") {
    evaluationCards.innerHTML = '<div class="evaluation-empty">Evaluation will appear after extraction finishes.</div>';
    return;
  }
  if (evaluation.status === "no_metrics") {
    evaluationCards.innerHTML = `<div class="evaluation-empty">${escapeHtml(evaluation.message || "No evaluation metrics yet.")}</div>`;
    return;
  }
  const summary = evaluation.summary || {};
  const queries = Array.isArray(evaluation.queries) ? evaluation.queries : [];
  evaluationCards.innerHTML = `
    <article class="evaluation-summary-card">
      <header>
        <div>
          <h5>Query Recall</h5>
          <p>${escapeHtml(evaluation.message || "Evaluation measured.")}</p>
        </div>
        <span>${escapeHtml(statusLabel(evaluation.status))}</span>
      </header>
      <dl>
        <div><dt>Can answer</dt><dd>${escapeHtml(String(summary.can_answer ?? 0))}/${escapeHtml(String(summary.queries ?? 0))}</dd></div>
        <div><dt>SQL answer</dt><dd>${formatRecallValue(summary.answer_covered, summary.answer_total, summary.sql_answer_recall)}</dd></div>
        <div><dt>Cell</dt><dd>${formatRecallValue(summary.cell_covered, summary.cell_total, summary.cell_recall)}</dd></div>
        <div><dt>Table</dt><dd>${formatRecallValue(summary.table_covered, summary.table_total, summary.table_recall)}</dd></div>
      </dl>
    </article>
    ${queries.length ? renderEvaluationQueryTable(queries) : '<div class="evaluation-empty">No query rows.</div>'}
  `;
}

function renderEvaluationQueryTable(queries) {
  return `
    <div class="evaluation-query-table" role="table" aria-label="Per-query evaluation">
      <div class="evaluation-query-head" role="row">
        <span>Query</span>
        <span>Status</span>
        <span>SQL</span>
        <span>Cell</span>
        <span>Table</span>
      </div>
      ${queries.map(renderEvaluationQueryRow).join("")}
    </div>
  `;
}

function renderEvaluationQueryRow(query) {
  const status = query.can_answer_query ? "ok" : "warn";
  return `
    <div class="evaluation-query-row" role="row">
      <strong>${escapeHtml([query.dataset, query.query_id].filter(Boolean).join(" · ") || "query")}</strong>
      <span class="${status}">${query.can_answer_query ? "OK" : "Partial"}</span>
      <b>${formatRecallMetric(query.sql_answer_recall)}</b>
      <b>${formatRecallMetric(query.cell_recall)}</b>
      <b>${formatRecallMetric(query.table_recall)}</b>
    </div>
  `;
}

function formatRecallMetric(metric) {
  if (!metric || typeof metric !== "object") {
    return "-";
  }
  return formatRecallValue(metric.covered, metric.total, metric.recall);
}

function formatRecallValue(covered, total, recall) {
  const coveredValue = Number(covered);
  const totalValue = Number(total);
  const prefix = Number.isFinite(coveredValue) && Number.isFinite(totalValue)
    ? `${coveredValue}/${totalValue}`
    : "-";
  return `${prefix} ${formatMetric(recall, { percent: true })}`;
}

function renderConsoleProgress() {
  const run = state.consoleRun;
  const progressItems = Object.values(run.progresses || {});
  if (!progressItems.length) {
    consoleProgressList.innerHTML = '<div class="console-log-empty">Progress bars will appear during stage loops.</div>';
    return;
  }
  consoleProgressList.innerHTML = progressItems
    .map((item) => {
      const total = Number(item.total) || 0;
      const current = Number(item.current) || 0;
      const percent = item.percent == null && total ? current / total : Number(item.percent) || 0;
      const boundedPercent = Math.max(0, Math.min(1, percent));
      const countText = total ? `${current}/${total}` : String(current);
      return `
        <article class="console-progress-item ${escapeHtml(item.status || "running")}">
          <div class="console-progress-head">
            <strong>${escapeHtml(item.label || "Progress")}</strong>
            <span>${escapeHtml(countText)} · ${escapeHtml(formatMetric(boundedPercent, { percent: true }))}</span>
          </div>
          <div class="console-progress-track" aria-hidden="true">
            <span style="width: ${boundedPercent * 100}%"></span>
          </div>
          <small>${escapeHtml(statusLabel(item.status))} · ${escapeHtml(formatMetric(item.rate))}/s</small>
        </article>
      `;
    })
    .join("");
}

function formatOptimizationMetrics(item) {
  const metrics = item.metrics || {};
  const preferredOrder = [
    "llm_doc_calls_before",
    "llm_doc_calls_after",
    "llm_doc_calls_saved",
    "llm_doc_call_reduction",
    "table_assignment_calls_before",
    "table_assignment_calls_after",
    "table_assignment_calls_saved",
    "table_assignment_call_reduction",
    "offline_only_gt_guard",
    "gt_guard_rejected_doc_calls",
    "source_table_metadata_hits",
    "source_table_metadata_misses",
    "cache_hits",
    "cache_misses",
    "total_conflicts",
    "text_pass_gt_fail",
    "gt_pass_text_fail",
    "affected_docs",
    "affected_queries",
    "checked_doc_predicates",
    "excluded_docs",
    "rejected",
    "pass_rate",
    "queries",
    "tables",
    "input_docs",
    "kept_docs",
  ];
  const entries = Object.entries(metrics).filter(([, value]) => value !== null && value !== undefined);
  if (!entries.length) {
    return '<div class="optimization-empty">No metrics emitted</div>';
  }
  const visibleMetricCount = item?.id === "table_assignment_cache" ? 8 : 6;
  return entries
    .sort(([left], [right]) => {
      const leftIndex = preferredOrder.indexOf(left);
      const rightIndex = preferredOrder.indexOf(right);
      return (leftIndex < 0 ? 999 : leftIndex) - (rightIndex < 0 ? 999 : rightIndex);
    })
    .slice(0, visibleMetricCount)
    .map(([key, value]) => {
      const label = key.replace(/_/g, " ");
      const formatted = /rate|recall|precision|reduction/.test(key)
        ? formatMetric(value, { percent: true })
        : formatMetric(value);
      return `<div><dt>${escapeHtml(label)}</dt><dd>${escapeHtml(formatted)}</dd></div>`;
    })
    .join("");
}

function formatOptimizationDetails(item) {
  const details = Array.isArray(item.details) ? item.details.slice(-5).reverse() : [];
  if (!details.length) {
    return "";
  }
  const rows = details
    .map((detail) => {
      if (item?.id === "dataset_consistency_audit" || detail.kind === "dataset_consistency_audit") {
        const ref = detail.conflict_ref || [detail.dataset, detail.query_id, detail.doc_id, detail.attribute].filter(Boolean).join(" · ");
        const type = detail.conflict_type || "conflict";
        const textValues = Array.isArray(detail.text_values) ? detail.text_values.join(", ") : "";
        const gtValues = Array.isArray(detail.ground_truth_values) ? detail.ground_truth_values.join(", ") : "";
        const values = textValues || gtValues
          ? `text [${textValues || "-"}] · gt [${gtValues || "-"}]`
          : "value mismatch";
        return `
          <li>
            <strong>${escapeHtml(ref || "dataset conflict")}</strong>
            <span>${escapeHtml(type)} · ${escapeHtml(values)}</span>
          </li>
        `;
      }
      const prefix = [detail.dataset, detail.query_id, detail.table].filter(Boolean).join(" · ");
      const saved = detail.llm_doc_calls_saved ?? detail.table_assignment_calls_saved ?? detail.excluded_docs ?? detail.rejected;
      const total = detail.input_docs ?? detail.evaluated ?? detail.llm_doc_calls_before ?? detail.table_assignment_calls_before;
      const kept = detail.kept_docs ?? detail.passed ?? detail.llm_doc_calls_after ?? detail.table_assignment_calls_after;
      const rate = detail.reduction ?? detail.pass_rate;
      const docIds = detail.excluded_doc_ids_preview || detail.rejected_doc_ids_preview || [];
      const docTotal = detail.excluded_doc_ids_total ?? detail.rejected_doc_ids_total;
      const docText = docIds.length
        ? ` · docs ${docIds.map(String).join(", ")}${docTotal > docIds.length ? ` +${docTotal - docIds.length}` : ""}`
        : "";
      const guardText = detail.offline_only_gt_guard ? " · offline-only GT guard" : "";
      const counts = total == null
        ? `${saved ?? 0} saved`
        : `${kept ?? 0}/${total} kept · ${saved ?? 0} saved`;
      return `
        <li>
          <strong>${escapeHtml(prefix || detail.kind || "update")}</strong>
          <span>${escapeHtml(counts)}${rate == null ? "" : ` · ${escapeHtml(formatMetric(rate, { percent: true }))}`}${escapeHtml(guardText)}${escapeHtml(docText)}</span>
        </li>
      `;
    })
    .join("");
  return `<details class="optimization-detail-list"><summary>Recent updates</summary><ul>${rows}</ul></details>`;
}

function optimizationDetailLogMessages(item) {
  const details = Array.isArray(item?.details) ? item.details : [];
  return details.map((detail, index) => ({
    key: optimizationDetailKey(item.id, detail, index),
    message: formatOptimizationDetailLogMessage(item, detail),
  }));
}

function optimizationDetailKey(optimizationId, detail, index) {
  return [
    optimizationId || detail.kind || "optimization",
    detail.dataset || "",
    detail.query_id || "",
    detail.table || "",
    detail.doc_id || "",
    detail.attribute || "",
    detail.conflict_type || "",
    detail.input_docs ?? detail.evaluated ?? detail.llm_doc_calls_before ?? "",
    detail.kept_docs ?? detail.passed ?? detail.llm_doc_calls_after ?? "",
    detail.excluded_docs ?? detail.rejected ?? detail.llm_doc_calls_saved ?? "",
  ].join("|");
}

function formatOptimizationDetailLogMessage(item, detail) {
  const dataset = detail.dataset ? `${detail.dataset} ` : "";
  const query = detail.query_id ? `${detail.query_id}` : "";
  if (item?.id === "doc_filter" || detail.kind === "doc_filter") {
    const inputDocs = Number(detail.input_docs) || 0;
    const excluded = Number(detail.excluded_docs) || 0;
    const reduction = detail.reduction ?? (inputDocs ? excluded / inputDocs : null);
    const docIds = detail.excluded_doc_ids_preview || [];
    const total = detail.excluded_doc_ids_total ?? excluded;
    const preview = docIds.length
      ? `; filtered docs ${docIds.map(String).join(", ")}${total > docIds.length ? ` +${total - docIds.length}` : ""}`
      : "";
    return `Doc Filter ${dataset}${query}: excluded ${excluded}/${inputDocs} docs, saved ${formatMetric(reduction, { percent: true })}${preview}`;
  }
  if (item?.id === "proxy_runtime" || detail.kind === "proxy_runtime") {
    const before = Number(detail.llm_doc_calls_before ?? detail.evaluated) || 0;
    const after = Number(detail.llm_doc_calls_after ?? detail.passed) || 0;
    const saved = Number(detail.llm_doc_calls_saved ?? detail.rejected) || Math.max(before - after, 0);
    const table = detail.table ? `${detail.table} ` : "";
    const docIds = detail.rejected_doc_ids_preview || [];
    const total = detail.rejected_doc_ids_total ?? saved;
    const preview = docIds.length
      ? `; rejected docs ${docIds.map(String).join(", ")}${total > docIds.length ? ` +${total - docIds.length}` : ""}`
      : "";
    const guard = detail.offline_only_gt_guard ? "; offline-only GT guard active" : "";
    return `Proxy ${table}${dataset}${query}: passed ${after}/${before} docs, saved ${saved} LLM-doc calls${guard}${preview}`;
  }
  if (item?.id === "table_assignment_cache" || detail.kind === "table_assignment_cache") {
    const before = Number(detail.table_assignment_calls_before ?? detail.input_docs) || 0;
    const after = Number(detail.table_assignment_calls_after ?? detail.cache_misses) || 0;
    const saved = Number(detail.table_assignment_calls_saved ?? detail.cache_hits) || Math.max(before - after, 0);
    const reduction = detail.reduction ?? (before ? saved / before : null);
    return `Table Cache ${dataset}${query}: reused ${saved}/${before} assignments, ${after} actual calls, saved ${formatMetric(reduction, { percent: true })}`;
  }
  if (item?.id === "dataset_consistency_audit" || detail.kind === "dataset_consistency_audit") {
    const ref = detail.conflict_ref || [detail.dataset, detail.query_id, detail.doc_id, detail.attribute].filter(Boolean).join(" ");
    const textValues = Array.isArray(detail.text_values) ? detail.text_values.join(", ") : "";
    const gtValues = Array.isArray(detail.ground_truth_values) ? detail.ground_truth_values.join(", ") : "";
    return `Dataset audit ${ref}: ${detail.conflict_type || "conflict"}; text [${textValues || "-"}] vs GT [${gtValues || "-"}]`;
  }
  if (item?.id === "schema_adaptive" || detail.kind === "schema_adaptive") {
    const processed = Number(detail.documents_processed) || 0;
    const filtered = Number(detail.filtered_documents ?? detail.total_documents) || 0;
    const saved = Number(detail.documents_saved) || Math.max(filtered - processed, 0);
    return `Schema Adaptive ${dataset}${query}: processed ${processed}/${filtered} docs, saved ${saved}`;
  }
  return `${item?.title || "Optimization"} ${dataset}${query}: ${item?.message || "metric updated"}`;
}

function renderConsoleLogs() {
  const run = state.consoleRun;
  if (!run.logs.length) {
    consoleLogs.innerHTML = '<div class="console-log-empty">Logs will appear when the run starts.</div>';
    return;
  }
  const visibleLogs = run.logs.slice(-CONSOLE_LOG_RENDER_LIMIT);
  const hiddenCount =
    (run.droppedLogCount || 0) + Math.max(0, run.logs.length - visibleLogs.length);
  const truncatedNotice = hiddenCount
    ? `<div class="console-log-truncated">Showing latest ${visibleLogs.length} logs. ${hiddenCount} older logs hidden.</div>`
    : "";
  consoleLogs.innerHTML =
    truncatedNotice +
    visibleLogs
      .map(
        (entry) => `
        <div class="console-log-line ${escapeHtml(entry.level || "info")}">
          <time>${escapeHtml(entry.time)}</time>
          <span>${escapeHtml(entry.message)}</span>
        </div>
      `,
      )
      .join("");
  consoleLogs.scrollTop = consoleLogs.scrollHeight;
}

function appendConsoleLog(message, level = "info") {
  if (!state.consoleRun) {
    return;
  }
  if (!state.consoleRun.logDedupeKeys) {
    state.consoleRun.logDedupeKeys = new Set();
  }
  const logKey = message;
  if (state.consoleRun.logDedupeKeys.has(logKey)) {
    return;
  }
  state.consoleRun.logDedupeKeys.add(logKey);
  const date = new Date();
  state.consoleRun.logs.push({
    time: date.toLocaleTimeString([], { hour12: false }),
    message,
    level,
  });
  if (state.consoleRun.logs.length > MAX_CONSOLE_LOGS) {
    const overflow = state.consoleRun.logs.length - MAX_CONSOLE_LOGS;
    state.consoleRun.logs.splice(0, overflow);
    state.consoleRun.droppedLogCount = (state.consoleRun.droppedLogCount || 0) + overflow;
  }
  renderConsoleLogs();
}

function appendOptimizationConsoleLog(key, message) {
  if (!state.consoleRun || !message) {
    return;
  }
  if (!state.consoleRun.optimizationLogKeys) {
    state.consoleRun.optimizationLogKeys = new Set();
  }
  if (state.consoleRun.optimizationLogKeys.has(key)) {
    return;
  }
  state.consoleRun.optimizationLogKeys.add(key);
  appendConsoleLog(message);
}

function updateConsoleStep(stepId, status, message = "") {
  if (!state.consoleRun || !stepId) {
    return;
  }
  const step = state.consoleRun.steps.find((item) => item.id === stepId);
  if (!step) {
    return;
  }
  step.status = status;
  if (message) {
    step.message = message;
  }
  if (status === "running") {
    state.consoleRun.currentStep = step.label;
  }
  renderConsoleSteps();
  consoleCurrentStep.textContent = state.consoleRun.currentStep || "-";
}

function handleConsoleEvent(event) {
  if (!state.consoleRun) {
    resetConsoleRun();
  }
  if (event.run_id) {
    state.consoleRun.id = event.run_id;
  }
  if (event.status) {
    state.consoleRun.status = event.status === "completed" ? "completed" : event.status;
  }
  if (event.elapsed_seconds != null) {
    state.consoleRun.elapsedSeconds = event.elapsed_seconds;
  }

  if (event.type === "run_started") {
    state.consoleRun.status = "running";
    appendConsoleLog(event.message || "Run started.");
  } else if (event.type === "step_started") {
    updateConsoleStep(event.step, "running", event.message);
    appendConsoleLog(event.message || `Started ${event.step}.`);
  } else if (event.type === "step_completed") {
    updateConsoleStep(event.step, "done", event.message);
    appendConsoleLog(event.message || `Completed ${event.step}.`);
  } else if (event.type === "step_failed") {
    updateConsoleStep(event.step, "failed", event.message);
    appendConsoleLog(event.message || `Failed ${event.step}.`, "error");
  } else if (event.type === "optimization_update") {
    const optimization = event.optimization || {};
    upsertOptimization(optimization);
    upsertOptimizationActivities(optimization);
    const detailMessages = optimizationDetailLogMessages(optimization);
    if (optimization.partial && detailMessages.length) {
      detailMessages.forEach((detail, index) => {
        appendOptimizationConsoleLog(detail.key, index === 0 ? event.message || detail.message : detail.message);
      });
    } else {
      appendConsoleLog(event.message || "Optimization metric updated.");
      detailMessages.forEach((detail) => appendOptimizationConsoleLog(detail.key, detail.message));
    }
  } else if (event.type === "evaluation_update") {
    state.consoleRun.evaluation = event.evaluation || null;
    appendConsoleLog(event.message || "Evaluation updated.", "success");
  } else if (event.type === "progress_update") {
    upsertProgress(event.progress);
    appendConsoleLog(event.message || "Progress updated.");
  } else if (event.type === "run_completed") {
    state.consoleRun.status = "completed";
    state.consoleRun.currentStep = "Complete";
    appendConsoleLog(event.message || "Run completed.", "success");
  } else if (event.type === "run_failed") {
    state.consoleRun.status = "failed";
    state.consoleRun.error = event.error || { detail: event.message };
    const activeStep = state.consoleRun.steps.find((step) => step.status === "running");
    if (activeStep) {
      activeStep.status = "failed";
      activeStep.message = event.message || "Failed";
      state.consoleRun.currentStep = activeStep.label;
    }
    appendConsoleLog(event.message || "Run failed.", "error");
  } else if (event.type === "run_queued") {
    appendConsoleLog(event.message || "Run queued.");
  }
  renderConsoleRun();
}

function upsertOptimization(item) {
  if (!state.consoleRun || !item || !item.id) {
    return;
  }
  if (item.status === "not_enabled") {
    state.consoleRun.optimizations = state.consoleRun.optimizations.filter(
      (current) => current.id !== item.id,
    );
    renderOptimizationCards();
    return;
  }
  const index = state.consoleRun.optimizations.findIndex((current) => current.id === item.id);
  if (index >= 0) {
    state.consoleRun.optimizations[index] = mergeOptimizationItem(state.consoleRun.optimizations[index], item);
  } else {
    state.consoleRun.optimizations.push(item);
  }
  renderOptimizationCards();
}

function upsertOptimizationActivities(item) {
  if (!state.consoleRun || !item || !item.id || !Array.isArray(item.details)) {
    return;
  }
  if (!state.consoleRun.optimizationActivityKeys) {
    state.consoleRun.optimizationActivityKeys = new Set();
  }
  if (!Array.isArray(state.consoleRun.optimizationActivities)) {
    state.consoleRun.optimizationActivities = [];
  }
  item.details.forEach((detail, index) => {
    const activity = optimizationDetailActivity(item, detail, index);
    if (!activity || state.consoleRun.optimizationActivityKeys.has(activity.key)) {
      return;
    }
    state.consoleRun.optimizationActivityKeys.add(activity.key);
    state.consoleRun.optimizationActivities.push(activity);
  });
  state.consoleRun.optimizationActivities = state.consoleRun.optimizationActivities.slice(-80);
  renderOptimizationActivityFeed();
}

function optimizationDetailActivity(item, detail, index) {
  const key = optimizationDetailKey(item.id, detail, index);
  if (item.id === "doc_filter" || detail.kind === "doc_filter") {
    const before = Number(detail.input_docs ?? detail.llm_doc_calls_before) || 0;
    const after = Number(detail.kept_docs ?? detail.llm_doc_calls_after) || 0;
    const saved = Number(detail.excluded_docs ?? detail.llm_doc_calls_saved) || Math.max(before - after, 0);
    return {
      key,
      kind: "doc-filter",
      badge: "DF",
      title: "Doc Filter",
      dataset: detail.dataset || "",
      queryId: detail.query_id || "",
      table: "",
      before,
      after,
      saved,
      reduction: detail.reduction ?? (before ? saved / before : 0),
      docIds: detail.excluded_doc_ids_preview || [],
      docTotal: detail.excluded_doc_ids_total ?? saved,
    };
  }
  if (item.id === "proxy_runtime" || detail.kind === "proxy_runtime") {
    const before = Number(detail.llm_doc_calls_before ?? detail.evaluated) || 0;
    const after = Number(detail.llm_doc_calls_after ?? detail.passed) || 0;
    const saved = Number(detail.llm_doc_calls_saved ?? detail.rejected) || Math.max(before - after, 0);
    return {
      key,
      kind: "proxy-runtime",
      badge: "PX",
      title: "Predicate Proxy",
      dataset: detail.dataset || "",
      queryId: detail.query_id || "",
      table: detail.table || "",
      before,
      after,
      saved,
      reduction: before ? saved / before : 0,
      docIds: detail.rejected_doc_ids_preview || [],
      docTotal: detail.rejected_doc_ids_total ?? saved,
      offlineOnly: Boolean(detail.offline_only_gt_guard),
    };
  }
  if (item.id === "table_assignment_cache" || detail.kind === "table_assignment_cache") {
    const before = Number(detail.table_assignment_calls_before ?? detail.input_docs) || 0;
    const after = Number(detail.table_assignment_calls_after ?? detail.cache_misses) || 0;
    const saved = Number(detail.table_assignment_calls_saved ?? detail.cache_hits) || Math.max(before - after, 0);
    return {
      key,
      kind: "table-cache",
      badge: "TC",
      title: "Table Cache",
      dataset: detail.dataset || "",
      queryId: detail.query_id || "",
      table: "",
      before,
      after,
      saved,
      reduction: before ? saved / before : 0,
      docIds: [],
      docTotal: saved,
    };
  }
  if (item.id === "schema_adaptive" || detail.kind === "schema_adaptive") {
    const before = Number(detail.filtered_documents ?? detail.total_documents) || 0;
    const after = Number(detail.documents_processed) || 0;
    const saved = Number(detail.documents_saved) || Math.max(before - after, 0);
    return {
      key,
      kind: "schema-adaptive",
      badge: "AS",
      title: "Schema Adaptive",
      dataset: detail.dataset || "",
      queryId: detail.query_id || "",
      table: "",
      before,
      after,
      saved,
      reduction: before ? saved / before : 0,
      docIds: [],
      docTotal: 0,
    };
  }
  return null;
}

function mergeOptimizationItem(existing, incoming) {
  if (!incoming.partial) {
    return {
      ...incoming,
      details: Array.isArray(incoming.details) && incoming.details.length
        ? incoming.details
        : existing.details,
    };
  }
  const metrics = mergeOptimizationMetrics(existing.metrics || {}, incoming.metrics || {});
  const details = [
    ...(Array.isArray(existing.details) ? existing.details : []),
    ...(Array.isArray(incoming.details) ? incoming.details : []),
  ].slice(-12);
  return {
    ...existing,
    ...incoming,
    metrics,
    details,
  };
}

function mergeOptimizationMetrics(existing, incoming) {
  const metrics = { ...existing };
  const summable = new Set([
    "queries",
    "tables",
    "input_docs",
    "kept_docs",
    "excluded_docs",
    "evaluated",
    "passed",
    "rejected",
    "llm_doc_calls_before",
    "llm_doc_calls_after",
    "llm_doc_calls_saved",
    "table_assignment_calls_before",
    "table_assignment_calls_after",
    "table_assignment_calls_saved",
    "source_table_metadata_hits",
    "source_table_metadata_misses",
    "cache_hits",
    "cache_misses",
  ]);
  Object.entries(incoming).forEach(([key, value]) => {
    if (typeof value === "number" && summable.has(key)) {
      metrics[key] = (Number(metrics[key]) || 0) + value;
    } else {
      metrics[key] = value;
    }
  });
  if (metrics.llm_doc_calls_before) {
    metrics.llm_doc_call_reduction = (Number(metrics.llm_doc_calls_saved) || 0) / Number(metrics.llm_doc_calls_before);
  }
  if (metrics.table_assignment_calls_before) {
    metrics.table_assignment_call_reduction =
      (Number(metrics.table_assignment_calls_saved) || 0) / Number(metrics.table_assignment_calls_before);
  }
  if (metrics.evaluated) {
    metrics.pass_rate = (Number(metrics.passed) || 0) / Number(metrics.evaluated);
  }
  return metrics;
}

function upsertProgress(item) {
  if (!state.consoleRun || !item || !item.id) {
    return;
  }
  state.consoleRun.progresses[item.id] = item;
  renderConsoleProgress();
}

async function startStreamingRun(payload, options = {}) {
  resetConsoleRun(options.steps || null);
  clearCurrentRunPayload();
  const run = await fetchJson(options.endpoint || "/api/runs", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  state.consoleRun.id = run.run_id;
  state.consoleRun.status = run.status || "queued";
  renderConsoleRun();

  return new Promise((resolve, reject) => {
    const source = new EventSource(`/api/runs/${encodeURIComponent(run.run_id)}/events`);
    state.consoleEventSource = source;
    source.onmessage = async (message) => {
      try {
        const event = JSON.parse(message.data);
        handleConsoleEvent(event);
        if (event.type === "run_completed") {
          source.close();
          state.consoleEventSource = null;
          state.consoleRun.status = "running";
          updateConsoleStep("refresh_outputs", "running", "Refreshing saved output cards.");
          renderConsoleRun();
          state.currentRunOutputsReady = false;
          renderPayload(event.payload);
          state.currentRunOutputsReady = true;
          await loadOutputResults();
          updateConsoleStep("refresh_outputs", "done", "Saved outputs refreshed.");
          state.consoleRun.status = "completed";
          state.consoleRun.currentStep = "Complete";
          renderConsoleRun();
          resolve(event.payload);
        }
        if (event.type === "run_failed") {
          source.close();
          state.consoleEventSource = null;
          reject(new Error(errorMessage({ detail: event.error }) || event.message || "Run failed."));
        }
      } catch (error) {
        source.close();
        state.consoleEventSource = null;
        reject(error);
      }
    };
    source.onerror = () => {
      source.close();
      state.consoleEventSource = null;
      reject(new Error("Run event stream disconnected."));
    };
  });
}

function average(values) {
  const numeric = values.map(Number).filter((value) => !Number.isNaN(value));
  if (!numeric.length) {
    return null;
  }
  return numeric.reduce((sum, value) => sum + value, 0) / numeric.length;
}

function collectNumbers(value, patterns, results = []) {
  if (Array.isArray(value)) {
    value.forEach((item) => collectNumbers(item, patterns, results));
    return results;
  }
  if (!value || typeof value !== "object") {
    return results;
  }
  Object.entries(value).forEach(([key, nested]) => {
    const normalizedKey = key.toLowerCase();
    if (typeof nested === "number" && patterns.some((pattern) => pattern.test(normalizedKey))) {
      results.push(nested);
    }
    collectNumbers(nested, patterns, results);
  });
  return results;
}

function countRunDatasets(payload) {
  const datasets = new Set(payload.datasets || []);
  Object.values(payload.result || {}).forEach((stageValue) => {
    if (Array.isArray(stageValue)) {
      stageValue.forEach((item) => {
        if (item && typeof item === "object" && item.dataset) {
          datasets.add(String(item.dataset));
        }
      });
    }
  });
  return datasets.size || null;
}

function countRunQueries(payload) {
  const evaluationQueries = Number(payload.evaluation?.summary?.queries);
  if (Number.isFinite(evaluationQueries) && evaluationQueries > 0) {
    return evaluationQueries;
  }
  if (Array.isArray(payload.query_ids) && payload.query_ids.length) {
    return payload.query_ids.length;
  }
  let count = 0;
  Object.values(payload.result || {}).forEach((stageValue) => {
    if (!Array.isArray(stageValue)) {
      return;
    }
    stageValue.forEach((item) => {
      if (item && typeof item === "object" && Array.isArray(item.query_ids) && item.query_ids.length) {
        count += item.query_ids.length;
        return;
      }
      if (item && typeof item === "object" && item.queries && typeof item.queries === "object") {
        count += Object.keys(item.queries).length;
      }
    });
  });
  return count || null;
}

function setFunnelStatus(id, value) {
  const node = document.querySelector(`#${id}`);
  if (node) {
    node.textContent = value;
  }
}

function setArchitectureFunnelOpen(open) {
  const expanded = Boolean(open);
  if (architectureFunnelBody) {
    architectureFunnelBody.hidden = !expanded;
  }
  if (architectureFunnelPanel) {
    architectureFunnelPanel.dataset.funnelCollapsed = String(!expanded);
  }
  if (architectureFunnelToggle) {
    architectureFunnelToggle.textContent = expanded ? "-" : "+";
    architectureFunnelToggle.setAttribute("aria-expanded", String(expanded));
    architectureFunnelToggle.setAttribute(
      "aria-label",
      expanded ? "Hide architecture funnel" : "Show architecture funnel",
    );
    architectureFunnelToggle.title = expanded ? "Hide architecture funnel" : "Show architecture funnel";
  }
}

function renderArchitectureFunnel(context = {}) {
  const stages = selectedStages();
  const payload = context.payload || state.lastPayload || {};
  const hasRun = Boolean(context.hasRun);
  const runDatasets = countRunDatasets(payload);
  const runQueries = countRunQueries(payload);
  const selectedDatasetCount = context.selectedDatasetCount ?? state.selectedDatasetIds.size;
  const selectedQueryLabel = context.selectedQueryLabel || (selectedDatasetCount ? "all" : "-");
  const rawLabel = hasRun
    ? `${runDatasets || selectedDatasetCount || 0} datasets`
    : selectedDatasetCount
      ? `${selectedDatasetCount} datasets / ${selectedQueryLabel} queries`
      : "Waiting";
  const recallLabel = context.recallLabel || "alpha guard";

  setFunnelStatus("funnel-raw", rawLabel);
  setFunnelStatus("funnel-csf", configOptDocFilterEnabled?.checked ? "Pruning enabled" : "Full corpus");
  setFunnelStatus("funnel-asc", configOptSchemaAdaptiveEnabled?.checked ? "Adaptive" : "Static");
  setFunnelStatus("funnel-ccg", configOptProxyEnabled?.checked ? "Proxy on" : "Ground truth");
  setFunnelStatus("funnel-teo", hasRun ? `${runQueries || "-"} queries` : selectedDatasetCount ? "Ready" : "Waiting");
  setFunnelStatus("funnel-ddg", stages.includes("preprocessing") ? "Selected" : "Optional");
  setFunnelStatus("funnel-dep", stages.includes("schema_refinement") ? "Selected" : "Optional");
  setFunnelStatus("funnel-acc", hasRun ? recallLabel : "alpha guard");
  setFunnelStatus("funnel-jpo", configOptJoinEnabled?.checked ? "Join on" : "Optional");
}

function relationAttributes(data, fallback = "") {
  if (data && typeof data === "object" && Object.keys(data).length) {
    return Object.entries(data)
      .slice(0, 4)
      .map(([key, value]) => `${key}: ${truncate(value, 44)}`)
      .join(" · ");
  }
  return truncate(fallback || "-", 140);
}

function hasRelationContent(data, fallback = "") {
  if (data && typeof data === "object" && Object.keys(data).length) {
    return true;
  }
  if (fallback == null || typeof fallback === "object") {
    return false;
  }
  const text = String(fallback).trim();
  return Boolean(text && text !== "-" && text !== "None");
}

function collectRelationRowsFromValue(value, rows = [], path = []) {
  if (!value || rows.length >= 8) {
    return rows;
  }
  if (Array.isArray(value)) {
    value.forEach((item, index) => {
      collectRelationRowsFromValue(item, rows, path.concat(String(index)));
    });
    return rows;
  }
  if (typeof value !== "object") {
    return rows;
  }
  if ("data" in value && ("res" in value || "table" in value || "id" in value)) {
    const table = value.table || value.res || "-";
    const data = value.data && typeof value.data === "object" ? value.data : {};
    if (hasRelationContent(data, value.value)) {
      rows.push({
        id: value.id || path[path.length - 1] || "-",
        table,
        attributes: relationAttributes(data, value.value),
        status: Object.keys(data).length ? "extracted" : "empty",
      });
    }
    return rows;
  }
  Object.entries(value).forEach(([key, nested]) => {
    collectRelationRowsFromValue(nested, rows, path.concat(key));
  });
  return rows;
}

function savedRelationRows(results = []) {
  const rows = [];
  results.forEach((result) => {
    if (rows.length >= 8 || result.evaluation || !Array.isArray(result.preview)) {
      return;
    }
    result.preview.forEach((item) => {
      if (rows.length >= 8) {
        return;
      }
      const data = item.data && typeof item.data === "object" ? item.data : {};
      const table = item.table || result.tables?.[0] || "";
      if (!table || !hasRelationContent(data, item.value)) {
        return;
      }
      rows.push({
        id: item.id || result.query_id || "-",
        table,
        attributes: relationAttributes(data, item.value),
        status: result.stage || "saved",
      });
    });
  });
  return rows;
}

function currentRunQueryIds(payload = {}) {
  const ids = new Set();
  if (Array.isArray(payload.query_ids)) {
    payload.query_ids.forEach((id) => ids.add(String(id)));
  }
  Object.values(payload.dataset_roots || {}).forEach((item) => {
    if (Array.isArray(item?.query_ids)) {
      item.query_ids.forEach((id) => ids.add(String(id)));
    }
  });
  return ids;
}

function outputScopeFromRoot(outRoot = "") {
  const normalized = String(outRoot || "").replace(/\\/g, "/");
  const marker = "/outputs/";
  const parts = (normalized.includes(marker) ? normalized.split(marker).pop() : normalized)
    .split("/")
    .filter(Boolean);
  if (parts.length < 4) {
    return null;
  }
  return {
    project: parts[0],
    dataset_id: parts[1],
    stage: parts[2],
    artifact: parts.slice(3).join("/"),
  };
}

function currentRunOutputScopes(payload = {}) {
  const scopes = [];
  const extractionItems = payload.result?.data_extraction;
  if (Array.isArray(extractionItems)) {
    extractionItems.forEach((item) => {
      const scope = outputScopeFromRoot(item?.out_root);
      if (scope) {
        scopes.push({
          ...scope,
          dataset_id: item?.dataset || scope.dataset_id,
        });
      }
    });
  }
  return scopes;
}

function savedResultsForCurrentRun(results = []) {
  const payload = state.lastPayload || {};
  const datasetIds = new Set((payload.datasets || []).map(String));
  const queryIds = currentRunQueryIds(payload);
  const outputScopes = currentRunOutputScopes(payload);
  if (!datasetIds.size && !queryIds.size) {
    return [];
  }
  return results.filter((result) => {
    const datasetId = result.dataset_id || result.dataset || "";
    const queryId = result.query_id || "";
    const datasetMatches = !datasetIds.size || datasetIds.has(String(datasetId));
    const queryMatches = !queryIds.size || (queryId && queryIds.has(String(queryId)));
    const outputMatches =
      !outputScopes.length ||
      outputScopes.some((scope) => {
        const artifact = String(result.artifact || "");
        return (
          (!scope.project || result.project === scope.project) &&
          (!scope.dataset_id || String(datasetId) === String(scope.dataset_id)) &&
          (!scope.stage || result.stage === scope.stage) &&
          (!scope.artifact || artifact === scope.artifact)
        );
      });
    return datasetMatches && queryMatches && outputMatches;
  });
}

function renderRelationPreview() {
  if (!relationPreviewBody) {
    return;
  }
  const payloadRows = collectRelationRowsFromValue(state.lastPayload?.result || {});
  const hasRunResult = Boolean(state.lastPayload?.result);
  const activeRun = state.consoleRun && ["queued", "running"].includes(state.consoleRun.status);
  const savedRows = hasRunResult
    ? state.currentRunOutputsReady
      ? savedRelationRows(savedResultsForCurrentRun(state.resultLibrary || []))
      : []
    : activeRun
      ? []
    : savedRelationRows(state.resultLibrary || []);
  const rows = payloadRows.length ? payloadRows : savedRows;
  if (!rows.length) {
    const message = activeRun && !hasRunResult
      ? "Run in progress. Tuple preview will appear after completion."
      : hasRunResult && !state.currentRunOutputsReady
        ? "Refreshing tuple preview from saved outputs."
        : state.lastPayload?.result
      ? "No tuple preview rows were returned for the current run."
      : "Run a query pipeline to extract R_Q.";
    relationPreviewBody.innerHTML = `<tr><td colspan="4">${escapeHtml(message)}</td></tr>`;
    return;
  }
  relationPreviewBody.innerHTML = rows
    .map(
      (row) => `
        <tr>
          <td>${escapeHtml(row.id)}</td>
          <td><span class="relation-table-pill">${escapeHtml(row.table)}</span></td>
          <td>${escapeHtml(row.attributes)}</td>
          <td>${escapeHtml(row.status)}</td>
        </tr>
      `,
    )
    .join("");
}

function renderStageResults(payload) {
  const result = payload.result || {};
  const entries = Object.entries(result);
  if (!entries.length) {
    const activeRun = state.consoleRun && ["queued", "running"].includes(state.consoleRun.status);
    stageResults.innerHTML = activeRun
      ? '<div class="empty-state">Run in progress. Stage results will appear after completion.</div>'
      : state.resultLibrary.length
      ? '<div class="empty-state">No in-memory run. Saved outputs are loaded below.</div>'
      : '<div class="empty-state">No run yet.</div>';
    return;
  }

  stageResults.innerHTML = "";
  entries.forEach(([stage, value]) => {
    const card = document.createElement("article");
    card.className = "stage-card";
    card.innerHTML = `
      <header>
        <h4>${escapeHtml(stage)}</h4>
        <span class="badge">complete</span>
      </header>
      <dl>
        <div>
          <dt>Items</dt>
          <dd>${stageCount(value)}</dd>
        </div>
        <div>
          <dt>Type</dt>
          <dd>${Array.isArray(value) ? "list" : typeof value}</dd>
        </div>
        <div>
          <dt>Status</dt>
          <dd>ok</dd>
        </div>
      </dl>
    `;
    stageResults.appendChild(card);
  });
  if (payload.evaluation && payload.evaluation.status !== "not_enabled") {
    const card = document.createElement("article");
    card.className = "stage-card evaluation-stage-card";
    const summary = payload.evaluation.summary || {};
    card.innerHTML = `
      <header>
        <h4>evaluation</h4>
        <span class="badge">${escapeHtml(statusLabel(payload.evaluation.status))}</span>
      </header>
      <dl>
        <div>
          <dt>SQL answer</dt>
          <dd>${formatRecallValue(summary.answer_covered, summary.answer_total, summary.sql_answer_recall)}</dd>
        </div>
        <div>
          <dt>Cell</dt>
          <dd>${formatRecallValue(summary.cell_covered, summary.cell_total, summary.cell_recall)}</dd>
        </div>
        <div>
          <dt>Can answer</dt>
          <dd>${escapeHtml(String(summary.can_answer ?? 0))}/${escapeHtml(String(summary.queries ?? 0))}</dd>
        </div>
      </dl>
    `;
    stageResults.appendChild(card);
  }
}

function formatBytes(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) {
    return "-";
  }
  if (number < 1024) {
    return `${number} B`;
  }
  if (number < 1024 * 1024) {
    return `${(number / 1024).toFixed(1)} KB`;
  }
  return `${(number / (1024 * 1024)).toFixed(1)} MB`;
}

function formatModified(value) {
  if (!value) {
    return "-";
  }
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return "-";
  }
  return date.toLocaleString([], {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function renderResultPreview(result) {
  if (result.parse_error) {
    return `<div class="empty-state compact">${escapeHtml(result.parse_error)}</div>`;
  }
  if (result.evaluation) {
    const evaluation = result.evaluation;
    return `
      <div class="result-evaluation-preview">
        <span>${evaluation.can_answer_query ? "Can answer" : "Partial answer"}</span>
        <b>SQL ${formatRecallMetric(evaluation.sql_answer_recall)}</b>
        <b>Cell ${formatRecallMetric(evaluation.cell_recall)}</b>
        <b>Table ${formatRecallMetric(evaluation.table_recall)}</b>
      </div>
    `;
  }
  const rows = (result.preview || []).filter((row) =>
    hasRelationContent(row.data && typeof row.data === "object" ? row.data : {}, row.value),
  );
  if (!rows.length) {
    return '<div class="empty-state compact">No extracted attribute preview records.</div>';
  }
  return rows
    .map((row) => {
      const data = row.data && typeof row.data === "object" ? row.data : {};
      const values = Object.entries(data)
        .slice(0, 4)
        .map(([key, value]) => `<span>${escapeHtml(key)}: ${escapeHtml(truncate(value, 70))}</span>`)
        .join("");
      return `
        <div class="result-preview-row">
          <strong>${escapeHtml(row.id || "-")}</strong>
          <em>${escapeHtml(row.table || "-")}</em>
          <span>${values || escapeHtml(truncate(row.value, 120))}</span>
        </div>
      `;
    })
    .join("");
}

function renderOutputResults() {
  const results = state.resultLibrary || [];
  syncResultsRenderLimit(results.length);
  if (resultsCountLabel) {
    resultsCountLabel.textContent = state.resultsUnavailable
      ? "Results refresh unavailable"
      : results.length
      ? `${results.length} saved ${results.length === 1 ? "result" : "results"}`
      : "No saved results";
  }
  if (!resultsLibrary) {
    return;
  }
  if (state.resultsUnavailable) {
    resultsLibrary.innerHTML = `
      <div class="empty-state results-unavailable">
        ${escapeHtml(state.resultsUnavailable)}
      </div>
    `;
    renderRelationPreview();
    return;
  }
  if (!results.length) {
    resultsLibrary.innerHTML = "";
    renderRelationPreview();
    return;
  }
  const visibleResults = results.slice(0, visibleResultCount(results.length));
  const projects = groupResultsByProject(visibleResults);
  const projectTotals = new Map(
    groupResultsByProject(results).map((group) => [group.project, group.items.length]),
  );
  resultsLibrary.innerHTML = `
    <section class="saved-results-head">
      <div>
        <span class="eyebrow">Outputs</span>
        <h4>Saved Results by Project</h4>
      </div>
      <strong>${projectTotals.size} projects / ${visibleResults.length} of ${results.length} files</strong>
    </section>
    <div class="project-results-list">
      ${projects.map((group) => renderProjectResults(group, projectTotals)).join("")}
    </div>
    ${renderResultsPager(visibleResults.length, results.length)}
  `;
  resultsLibrary.querySelector("[data-results-show-more]")?.addEventListener("click", () => {
    state.resultsRenderLimit = visibleResultCount(results.length) + RESULTS_RENDER_BATCH;
    renderOutputResults();
  });
  resultsLibrary.querySelector("[data-results-show-all]")?.addEventListener("click", () => {
    state.resultsRenderLimit = results.length;
    renderOutputResults();
  });
  resultsLibrary.querySelector("[data-results-collapse]")?.addEventListener("click", () => {
    state.resultsRenderLimit = RESULTS_RENDER_BATCH;
    state.pendingDeleteResultPath = "";
    renderOutputResults();
    resultsLibrary.scrollTop = 0;
  });
  resultsLibrary.querySelectorAll("[data-result-delete]").forEach((button) => {
    button.addEventListener("click", () => {
      state.pendingDeleteResultPath = button.dataset.resultDelete || "";
      renderOutputResults();
    });
  });
  resultsLibrary.querySelectorAll("[data-result-delete-cancel]").forEach((button) => {
    button.addEventListener("click", () => {
      state.pendingDeleteResultPath = "";
      renderOutputResults();
    });
  });
  resultsLibrary.querySelectorAll("[data-result-delete-confirm]").forEach((button) => {
    button.addEventListener("click", () => {
      deleteResultFile(button.dataset.resultDeleteConfirm || "");
    });
  });
  revealPendingResultDeleteConfirm();
  renderRelationPreview();
}

function syncResultsRenderLimit(totalResults) {
  const count = Math.max(0, Number(totalResults) || 0);
  if (!Number.isFinite(state.resultsRenderLimit) || state.resultsRenderLimit < RESULTS_RENDER_BATCH) {
    state.resultsRenderLimit = RESULTS_RENDER_BATCH;
  }
  if (!count) {
    state.resultsRenderLimit = RESULTS_RENDER_BATCH;
    state.pendingDeleteResultPath = "";
    return;
  }
  state.resultsRenderLimit = Math.min(Math.max(state.resultsRenderLimit, RESULTS_RENDER_BATCH), count);
}

function visibleResultCount(totalResults) {
  syncResultsRenderLimit(totalResults);
  return Math.min(Number(totalResults) || 0, state.resultsRenderLimit);
}

function renderResultsPager(visibleCount, totalCount) {
  if (totalCount <= RESULTS_RENDER_BATCH) {
    return "";
  }
  if (visibleCount >= totalCount) {
    return `
      <section class="results-window-controls complete">
        <span>All ${totalCount} files visible</span>
        <button class="secondary-button" type="button" data-results-collapse>Collapse</button>
      </section>
    `;
  }
  const nextCount = Math.min(RESULTS_RENDER_BATCH, totalCount - visibleCount);
  return `
    <section class="results-window-controls">
      <span>${visibleCount} of ${totalCount} files visible</span>
      <div>
        <button class="secondary-button" type="button" data-results-show-more>Show ${nextCount} more</button>
        <button class="secondary-button" type="button" data-results-show-all>Show all</button>
      </div>
    </section>
  `;
}

function revealPendingResultDeleteConfirm() {
  if (!state.pendingDeleteResultPath) {
    return;
  }
  window.requestAnimationFrame(() => {
    const confirm = resultsLibrary?.querySelector(".result-delete-confirm");
    if (!confirm) {
      return;
    }
    confirm.scrollIntoView({ block: "nearest", inline: "nearest", behavior: "auto" });
  });
}

function groupResultsByProject(results) {
  const grouped = new Map();
  results.forEach((result) => {
    const project = result.project || (result.relative_path || "").split(/[\\/]/)[0] || "outputs";
    if (!grouped.has(project)) {
      grouped.set(project, []);
    }
    grouped.get(project).push(result);
  });
  return Array.from(grouped.entries()).map(([project, items]) => ({ project, items }));
}

function renderProjectResults(group, projectTotals) {
  const total = projectTotals.get(group.project) || group.items.length;
  const countLabel =
    total === group.items.length
      ? `${total} ${total === 1 ? "file" : "files"}`
      : `${group.items.length} of ${total} files`;
  return `
    <section class="project-results-group">
      <header>
        <div>
          <span class="eyebrow">Project</span>
          <h5>${escapeHtml(group.project)}</h5>
        </div>
        <strong>${escapeHtml(countLabel)}</strong>
      </header>
      <div class="saved-results-grid">
        ${group.items.map(renderResultFileCard).join("")}
      </div>
    </section>
  `;
}

function renderResultFileCard(result) {
  const tables = (result.tables || []).slice(0, 3);
  const columns = (result.columns || []).slice(0, 4);
  const isEvaluation = Boolean(result.evaluation);
  const relativePath = result.relative_path || "";
  const isConfirmingDelete = Boolean(relativePath && state.pendingDeleteResultPath === relativePath);
  const resultName = result.dataset_id || "output";
  const resultPath = result.relative_path || result.name || "";
  return `
    <article class="result-file-card ${isEvaluation ? "evaluation-result-card" : ""}">
      <header>
        <div>
          <h4 title="${escapeHtml(resultName)}">${escapeHtml(resultName)}</h4>
          <p title="${escapeHtml(resultPath)}">${escapeHtml(resultPath)}</p>
        </div>
        <span>${escapeHtml(isEvaluation ? "evaluation" : result.stage || "json")}</span>
      </header>
      <dl>
        <div><dt>${isEvaluation ? "SQL answer" : "Records"}</dt><dd>${
          isEvaluation ? formatRecallMetric(result.evaluation.sql_answer_recall) : result.records_count ?? "-"
        }</dd></div>
        <div><dt>Query</dt><dd>${escapeHtml(result.query_id || "-")}</dd></div>
        <div><dt>Size</dt><dd>${formatBytes(result.size_bytes)}</dd></div>
        <div><dt>Updated</dt><dd>${escapeHtml(formatModified(result.modified))}</dd></div>
      </dl>
      <div class="result-tags">
        ${
          isEvaluation
            ? `<b>${result.evaluation.can_answer_query ? "Can answer" : "Partial answer"}</b>`
            : tables.length
            ? tables.map((item) => `<b>${escapeHtml(item)}</b>`).join("")
            : "<b>No table</b>"
        }
        ${isEvaluation ? "" : columns.map((item) => `<span>${escapeHtml(item)}</span>`).join("")}
      </div>
      <div class="result-preview">${renderResultPreview(result)}</div>
      <footer class="result-file-actions">
        ${
          isConfirmingDelete
            ? `
              <div class="result-delete-confirm" role="group" aria-label="Confirm result deletion">
                <span>Delete this file?</span>
                <button class="secondary-button" type="button" data-result-delete-cancel>Cancel</button>
                <button class="danger-button" type="button" data-result-delete-confirm="${escapeHtml(
                  relativePath,
                )}">Delete</button>
              </div>
            `
            : `<button class="danger-button" type="button" data-result-delete="${escapeHtml(relativePath)}">Delete</button>`
        }
      </footer>
    </article>
  `;
}

async function deleteResultFile(relativePath) {
  if (!relativePath) {
    return;
  }
  setError("");
  try {
    await fetchJson("/api/results", {
      method: "DELETE",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ relative_path: relativePath }),
    });
    state.pendingDeleteResultPath = "";
    await loadOutputResults();
  } catch (error) {
    setError(error.message || String(error));
  }
}

function renderPayload(payload) {
  state.lastPayload = payload;
  if (state.consoleRun && payload.evaluation) {
    state.consoleRun.evaluation = payload.evaluation;
  }
  summaryExperiment.textContent = payload.experiment || "-";
  summaryDatasets.textContent = String((payload.datasets || []).length || "-");
  summaryStages.textContent = String((payload.stages || []).length || "-");
  summaryElapsed.textContent =
    payload.elapsed_seconds == null ? "-" : `${payload.elapsed_seconds}s`;
  rawJson.textContent = JSON.stringify(payload, null, 2);
  renderStageResults(payload);
  renderOutputResults();
  state.configMaterialized = true;
  renderGeneratedConfig();
  renderOverview();
}

function errorMessage(errorPayload) {
  const detail = errorPayload && errorPayload.detail;
  if (detail && typeof detail === "object") {
    return `${detail.error || "Error"}: ${detail.detail || ""}`.trim();
  }
  if (typeof detail === "string") {
    return detail;
  }
  return "Request failed.";
}

function datasetItems() {
  return configDatasetItems();
}

function datasetSelectItems() {
  if (state.datasetSource === "all") {
    return state.registryDatasets.filter((item) => SEARCH_DATASET_ALLOWLIST.has(item.id));
  }
  return configDatasetItems();
}

function configDatasetItems() {
  return configDatasetIds().map((datasetId) => {
    const draft = ensureDatasetDraft(datasetId);
    const configDataset = (state.configInfo?.datasets || []).find((item) => item.id === datasetId);
    const registryDataset = state.registryDatasets.find((item) => item.id === datasetId);
    const source = configDataset || registryDataset || { id: datasetId };
    return {
      id: datasetId,
      kind: state.selectedDatasetIds.has(datasetId) ? "used" : "disabled",
      path: draft.root || source.root || source.path || "",
      loader: draft.loader || source.loader,
      documents_count: registryDataset?.documents_count ?? null,
      queries_count: registryDataset?.queries_count ?? null,
      tables_count: registryDataset?.tables_count ?? null,
      isConfig: true,
      isUsed: state.selectedDatasetIds.has(datasetId),
    };
  });
}

function renderDatasets() {
  const items = configDatasetItems();
  datasetSourceLabel.textContent = "From Config";
  configSourcePanel.classList.toggle("active", state.datasetSource === "config");
  allSourcePanel.classList.toggle("active", state.datasetSource === "all");
  uploadSourcePanel.classList.toggle("active", state.datasetSource === "upload");

  renderDatasetSelect(datasetSelectItems());
  renderDatasetBrowser(items);
  renderSelectedDatasets();
  renderAddDatasetButton();
  renderGeneratedConfig();
}

function renderDatasetSelect(items) {
  const selectedValue = state.pendingDatasetId || datasetSelect.value;
  datasetSelect.innerHTML = `
    <option value="">Select dataset</option>
    ${items
      .map(
        (item) => `<option value="${escapeHtml(item.id)}" ${
          item.id === selectedValue ? "selected" : ""
        }>${escapeHtml(item.id)}</option>`,
      )
      .join("")}
  `;
  datasetSelect.value = selectedValue;
  renderDatasetPicker(items);
}

function scrollDropdownIntoView(menu) {
  if (!menu || menu.hidden) {
    return;
  }
  window.requestAnimationFrame(() => {
    if (getComputedStyle(menu).position === "fixed") {
      return;
    }
    const inlineDropdown = window.matchMedia("(max-width: 480px), (max-height: 520px)").matches;
    menu.scrollIntoView({ block: inlineDropdown ? "center" : "nearest", inline: "nearest" });
    if (!inlineDropdown) {
      return;
    }
    window.requestAnimationFrame(() => {
      const rect = menu.getBoundingClientRect();
      const margin = 12;
      const targetTop = Math.max(margin, (window.innerHeight - rect.height) / 2);
      if (rect.top < margin || rect.bottom > window.innerHeight - margin) {
        window.scrollBy({ top: rect.top - targetTop, left: 0, behavior: "auto" });
      }
    });
  });
}

const selectionMenuScrollLock = {
  locked: false,
  scrollY: 0,
  body: {},
  html: {},
};

function setSelectionMenuScrollLock(locked) {
  const shouldLock = Boolean(locked);
  if (selectionMenuScrollLock.locked === shouldLock) {
    return;
  }

  const body = document.body;
  const html = document.documentElement;
  if (shouldLock) {
    selectionMenuScrollLock.scrollY = window.scrollY || html.scrollTop || 0;
    selectionMenuScrollLock.body = {
      position: body.style.position,
      top: body.style.top,
      right: body.style.right,
      left: body.style.left,
      width: body.style.width,
      overflow: body.style.overflow,
    };
    selectionMenuScrollLock.html = {
      overflow: html.style.overflow,
    };
    body.style.position = "fixed";
    body.style.top = `-${selectionMenuScrollLock.scrollY}px`;
    body.style.left = "0";
    body.style.right = "0";
    body.style.width = "100%";
    body.style.overflow = "hidden";
    html.style.overflow = "hidden";
    selectionMenuScrollLock.locked = true;
    return;
  }

  const { body: bodyStyles, html: htmlStyles } = selectionMenuScrollLock;
  body.style.position = bodyStyles.position || "";
  body.style.top = bodyStyles.top || "";
  body.style.right = bodyStyles.right || "";
  body.style.left = bodyStyles.left || "";
  body.style.width = bodyStyles.width || "";
  body.style.overflow = bodyStyles.overflow || "";
  html.style.overflow = htmlStyles.overflow || "";
  selectionMenuScrollLock.locked = false;
  window.scrollTo({ top: selectionMenuScrollLock.scrollY, left: 0, behavior: "auto" });
}

function syncSelectionMenuScrollLock() {
  setSelectionMenuScrollLock(state.datasetPickerOpen || state.queryDropdownOpen);
}

function closeSelectionMenus() {
  state.datasetPickerOpen = false;
  state.queryDropdownOpen = false;
  datasetPickerList.hidden = true;
  queryList.hidden = true;
  datasetPickerTrigger.setAttribute("aria-expanded", "false");
  querySelectTrigger.setAttribute("aria-expanded", "false");
  syncSelectionMenuScrollLock();
}

function selectionMenuContainsEvent(event, trigger, menu) {
  const path = typeof event.composedPath === "function" ? event.composedPath() : [];
  if (path.includes(trigger) || path.includes(menu)) {
    return true;
  }
  const target = event.target;
  return target instanceof Node && (trigger.contains(target) || menu.contains(target));
}

function setDatasetPickerOpen(open) {
  if (open && state.queryDropdownOpen) {
    state.queryDropdownOpen = false;
    queryList.hidden = true;
    querySelectTrigger.setAttribute("aria-expanded", "false");
  }
  state.datasetPickerOpen = Boolean(open && datasetSelectItems().length);
  datasetPickerList.hidden = !state.datasetPickerOpen;
  datasetPickerTrigger.setAttribute("aria-expanded", String(state.datasetPickerOpen));
  syncSelectionMenuScrollLock();
  if (state.datasetPickerOpen) {
    scrollDropdownIntoView(datasetPickerList);
  }
}

function renderDatasetPicker(items) {
  const selectedItem = items.find(
    (item) => item.id === (state.pendingDatasetId || datasetSelect.value),
  );
  const availableCount = items.length;
  datasetPickerLabel.textContent = selectedItem?.id || "Select dataset";
  datasetPickerCount.textContent = `${availableCount} available`;
  datasetPickerBadge.textContent = selectedItem ? "Selected" : "Browse";
  datasetPickerTrigger.disabled = !availableCount;
  if (!availableCount) {
    setDatasetPickerOpen(false);
    datasetPickerList.innerHTML = '<div class="empty-state compact">No datasets available.</div>';
    return;
  }
  datasetPickerList.hidden = !state.datasetPickerOpen;
  datasetPickerTrigger.setAttribute("aria-expanded", String(state.datasetPickerOpen));
  syncSelectionMenuScrollLock();
  datasetPickerList.innerHTML = `
    <div class="selection-window-head">
      <div>
        <span>Dataset Selection</span>
        <strong>${selectedItem ? escapeHtml(selectedItem.id) : "Available Datasets"}</strong>
      </div>
      <button type="button" data-dataset-picker-action="close">Done</button>
    </div>
    <div class="selection-window-body">
      ${items
        .map(
          (item) => `
            <button
              class="dataset-picker-option ${item.id === selectedItem?.id ? "active" : ""}"
              type="button"
              data-dataset-picker-id="${escapeHtml(item.id)}"
            >
              <span>
                <strong>${escapeHtml(item.id)}</strong>
                <small>${escapeHtml(item.root || item.path || "")}</small>
              </span>
              <em>${item.queries_count ?? "-"} queries</em>
            </button>
          `,
        )
        .join("")}
    </div>
  `;
  datasetPickerList
    .querySelector("[data-dataset-picker-action='close']")
    ?.addEventListener("click", () => {
      setDatasetPickerOpen(false);
      datasetPickerTrigger.focus();
    });
  datasetPickerList.querySelectorAll("[data-dataset-picker-id]").forEach((button) => {
    button.addEventListener("click", () => {
      datasetSelect.value = button.dataset.datasetPickerId;
      datasetSelect.dispatchEvent(new Event("change"));
      setDatasetPickerOpen(false);
      datasetPickerTrigger.focus();
    });
  });
  if (state.datasetPickerOpen) {
    scrollDropdownIntoView(datasetPickerList);
  }
}

function renderDatasetBrowser(items) {
  if (!items.length) {
    datasetBrowser.innerHTML = '<div class="empty-state">No datasets in the current config.</div>';
    return;
  }

  datasetBrowser.innerHTML = "";
  items.forEach((item) => {
    const row = document.createElement("article");
    row.className = `dataset-row ${state.activeDatasetId === item.id ? "active" : ""}`;
    row.innerHTML = `
      <button type="button" class="dataset-row-main">
        <span>
          <strong>${escapeHtml(item.id)}</strong>
          <small>${escapeHtml(item.path || "")}</small>
        </span>
        <span class="dataset-metrics">
          <b>${escapeHtml(item.kind || "dataset")}</b>
          <em>${item.documents_count ?? "-"} docs</em>
          <em>${item.queries_count ?? "-"} queries</em>
          <em>${item.tables_count ?? "-"} tables</em>
        </span>
      </button>
    `;
    row.querySelector("button").addEventListener("click", () => {
      loadDatasetDetail(item);
    });
    datasetBrowser.appendChild(row);
  });
}

function renderSelectedDatasets() {
  const ids = selectedDatasetIds();
  const configuredIds = configDatasetIds();
  const queryCountLabel =
    state.selectedQueryIds.size
      ? String(state.selectedQueryIds.size)
      : ids.length
        ? "all"
        : "-";
  selectedCount.textContent = `${ids.length} ${ids.length === 1 ? "dataset" : "datasets"}`;
  targetDatasetCount.textContent = String(ids.length);
  targetQueryCount.textContent = queryCountLabel;
  summaryDatasets.textContent = String(ids.length || "-");
  summaryStages.textContent = state.configMaterialized ? String(selectedStages().length || "-") : "-";
  renderConfigDatasetList();
  renderOverview();
  if (!selectedDatasets) {
    return;
  }
  if (!configuredIds.length) {
    selectedDatasets.innerHTML = '<div class="empty-state compact">No datasets in config.</div>';
    return;
  }
  selectedDatasets.innerHTML = configuredIds
    .map(
      (id) => `
        <article class="target-row" data-id="${escapeHtml(id)}">
          <span>
            <strong>${escapeHtml(id)}</strong>
            <small>${escapeHtml(datasetSourceForId(id))}</small>
          </span>
          <em>${state.selectedDatasetIds.has(id) ? "Used" : "Off"}</em>
        </article>
      `,
    )
    .join("");
  selectedDatasets.querySelectorAll(".target-row").forEach((row) => {
    row.addEventListener("click", () => {
      const item = datasetItems().find((dataset) => dataset.id === row.dataset.id);
      if (item) {
        loadDatasetDetail(item);
      }
    });
  });
}

function renderConfigDatasetList() {
  if (!configDatasetList || !configDatasetCount) {
    return;
  }
  const ids = configDatasetIds();
  const usedCount = selectedDatasetIds().length;
  configDatasetCount.textContent = `${usedCount}/${ids.length} used`;
  if (!ids.length) {
    configDatasetList.innerHTML = '<div class="empty-state compact">No datasets in config.</div>';
    return;
  }
  configDatasetList.innerHTML = ids
    .map((id) => {
      const draft = ensureDatasetDraft(id);
      const queryText = Array.isArray(draft.query_ids) ? draft.query_ids.join("\n") : "";
      return `
        <article class="config-dataset-card" data-dataset-id="${escapeHtml(id)}">
          <header>
            <label class="dataset-use-toggle">
              <input type="checkbox" data-dataset-action="use" ${state.selectedDatasetIds.has(id) ? "checked" : ""} />
              <span>Use</span>
            </label>
            <span>
            <strong>${escapeHtml(id)}</strong>
              <small>${escapeHtml(datasetSourceForId(id))}</small>
          </span>
            <button type="button" data-dataset-action="remove">Remove</button>
          </header>
          <div class="config-dataset-fields">
            <label>
              <span>Loader</span>
              <textarea data-dataset-field="loader" rows="2" title="${escapeHtml(
                draft.loader || "hf_manifest",
              )}" autocomplete="off">${escapeHtml(draft.loader || "hf_manifest")}</textarea>
            </label>
            <label>
              <span>Root</span>
              <textarea data-dataset-field="root" rows="3" autocomplete="off">${escapeHtml(draft.root || "")}</textarea>
            </label>
            <label>
              <span>Query IDs</span>
              <textarea data-dataset-field="query_ids" placeholder="All queries">${escapeHtml(queryText)}</textarea>
            </label>
            <label>
              <span>Loader Options JSON</span>
              <textarea data-dataset-field="loader_options" spellcheck="false">${escapeHtml(prettyJson(draft.loader_options || { manifest: "manifest.yaml" }))}</textarea>
            </label>
            <label>
              <span>Split JSON</span>
              <textarea data-dataset-field="split" spellcheck="false">${escapeHtml(prettyJson(draft.split || { train_count: 0 }))}</textarea>
            </label>
          </div>
        </article>
      `;
    })
    .join("");
  syncControlTitles(configDatasetList);
}

function datasetSourceForId(datasetId) {
  if ((state.configInfo?.datasets || []).some((dataset) => dataset.id === datasetId)) {
    return "config dataset";
  }
  if (state.registryDatasets.some((dataset) => dataset.id === datasetId)) {
    return "added to config";
  }
  return "config dataset";
}

function renderAddDatasetButton() {
  const detailLoaded = state.pendingDatasetLoadedId === state.pendingDatasetId;
  const requiresQuerySelection = state.activeDatasetQueries.length > 0;
  addDataset.disabled =
    !state.pendingDatasetId ||
    state.pendingDatasetLoading ||
    !detailLoaded ||
    (requiresQuerySelection && !state.pendingQueryIds.size);
}

function setQueryDropdownOpen(open) {
  if (open && state.datasetPickerOpen) {
    state.datasetPickerOpen = false;
    datasetPickerList.hidden = true;
    datasetPickerTrigger.setAttribute("aria-expanded", "false");
  }
  state.queryDropdownOpen = Boolean(open && state.activeDatasetQueries.length);
  queryList.hidden = !state.queryDropdownOpen;
  querySelectTrigger.setAttribute("aria-expanded", String(state.queryDropdownOpen));
  syncSelectionMenuScrollLock();
  if (state.queryDropdownOpen) {
    scrollDropdownIntoView(queryList);
  }
}

function querySelectCopy() {
  if (state.pendingDatasetLoading) {
    return {
      label: "Loading queries...",
      badge: "loading",
    };
  }
  if (!state.activeDatasetQueries.length) {
    return {
      label: state.pendingDatasetId ? "Default all schema attributes" : "Select dataset first",
      badge: state.pendingDatasetId ? "default" : "0 selected",
    };
  }
  const count = state.pendingQueryIds.size;
  const total = state.activeDatasetQueries.length;
  return {
    label: count ? `${count} of ${total} queries` : "Select queries",
    badge: `${count} selected`,
  };
}

function toggleDataset(datasetId, checked) {
  if (checked) {
    state.configDatasetIds.add(datasetId);
    ensureDatasetDraft(datasetId);
    state.selectedDatasetIds.add(datasetId);
  } else {
    state.selectedDatasetIds.delete(datasetId);
    if (!state.selectedDatasetIds.size) {
      state.selectedQueryIds.clear();
    }
  }
  state.selectedQueryIds = new Set(
    selectedDatasetIds().flatMap((id) => ensureDatasetDraft(id).query_ids || []),
  );
  renderDatasets();
}

function removeDatasetFromConfig(datasetId) {
  state.configDatasetIds.delete(datasetId);
  state.selectedDatasetIds.delete(datasetId);
  delete state.datasetDrafts[datasetId];
  state.selectedQueryIds = new Set(
    selectedDatasetIds().flatMap((id) => ensureDatasetDraft(id).query_ids || []),
  );
  if (state.activeDatasetId === datasetId) {
    state.activeDatasetId = null;
    detailTitle.textContent = "Dataset Detail";
    datasetDetail.innerHTML = '<div class="empty-state">Select a dataset to inspect it.</div>';
  }
  renderDatasets();
  renderOverview();
}

function updateDatasetDraftFromControl(control) {
  const card = control.closest("[data-dataset-id]");
  const datasetId = card?.dataset.datasetId;
  const field = control.dataset.datasetField;
  if (!datasetId || !field) {
    return;
  }
  const draft = ensureDatasetDraft(datasetId);
  if (field === "query_ids") {
    const queryIds = control.value
      .split(/\r?\n|,/)
      .map((item) => item.trim())
      .filter(Boolean);
    if (queryIds.length) {
      draft.query_ids = queryIds;
    } else {
      delete draft.query_ids;
    }
    state.selectedQueryIds = new Set(
      selectedDatasetIds().flatMap((id) => ensureDatasetDraft(id).query_ids || []),
    );
  } else if (field === "loader_options" || field === "split") {
    try {
      draft[field] = JSON.parse(control.value || "{}");
      control.classList.remove("invalid");
    } catch {
      control.classList.add("invalid");
      return;
    }
  } else {
    draft[field] = control.value.trim();
  }
  state.configMaterialized = true;
  renderGeneratedConfig();
  renderOverview();
}

function resetSelectionToConfig() {
  state.configDatasetIds = new Set(state.configInfo?.dataset_ids || []);
  state.datasetDrafts = {};
  state.selectedDatasetIds = new Set(state.configInfo?.dataset_ids || []);
  state.selectedQueryIds.clear();
  state.pendingDatasetId = "";
  state.pendingDatasetLoadedId = "";
  state.pendingQueryIds.clear();
  state.datasetPickerOpen = false;
  state.queryDropdownOpen = false;
  state.pendingDatasetLoading = false;
  state.activeDatasetQueries = [];
  state.activeDatasetId = null;
  renderQueriesSelector();
  renderDatasets();
}

async function loadConfigInfo({ resetSelection = true } = {}) {
  const resolvedExperiment = experiment.value.trim() || state.defaults?.experiment || "demo";
  if (!experiment.value.trim()) {
    experiment.value = resolvedExperiment;
  }
  const params = new URLSearchParams({
    config_path: configPath.value.trim(),
    experiment: resolvedExperiment,
  });
  const info = await fetchJson(`/api/config/inspect?${params.toString()}`);
  state.configInfo = info;
  state.configDraft = defaultConfigDraft(info);
  renderApiKeyStatus(info.api_key_status);
  experiment.value = info.experiment || "";
  experiment.innerHTML = (info.experiments || [])
    .map(
      (item) => `<option value="${escapeHtml(item.id)}" ${
        item.id === info.experiment ? "selected" : ""
      }>${escapeHtml(item.id)}</option>`,
    )
    .join("");
  document.querySelectorAll('input[name="stages"]').forEach((item) => {
    item.checked = (info.default_stages || []).includes(item.value);
  });
  summaryExperiment.textContent = info.experiment || "-";
  summaryDatasets.textContent = String((info.dataset_ids || []).length || "-");
  summaryStages.textContent = String((info.default_stages || []).length || "-");
  summaryElapsed.textContent = "-";
  setConfigEditorFromDraft();
  state.configMaterialized = true;
  if (resetSelection) {
    resetSelectionToConfig();
  } else {
    renderDatasets();
  }
  renderGeneratedConfig();
}

async function loadRegistryDatasets() {
  const payload = await fetchJson("/api/datasets");
  state.registryDatasets = payload.datasets || [];
  renderDatasets();
}

async function loadOutputResults() {
  try {
    const payload = await fetchJson("/api/results");
    state.resultLibrary = payload.results || [];
    state.resultsUnavailable = "";
  } catch (error) {
    if ((error.message || "").includes("Not Found")) {
      state.resultLibrary = [];
      state.resultsUnavailable =
        "Saved outputs need the latest web-demo backend. Restart the web demo server once, then click Refresh again.";
    } else {
      throw error;
    }
  }
  renderStageResults(state.lastPayload || {});
  renderOutputResults();
}

async function loadPendingDatasetQueries(item) {
  state.pendingDatasetLoading = true;
  state.pendingDatasetLoadedId = "";
  state.activeDatasetQueries = [];
  renderQueriesSelector();
  renderAddDatasetButton();
  renderOverview();

  try {
    const queries = await fetchJson(`/api/datasets/${encodeURIComponent(item.id)}/queries`);
    state.activeDatasetQueries = queries.queries || [];
    state.pendingQueryIds = new Set(
      state.activeDatasetQueries.map((query) => String(query.query_id)).filter(Boolean),
    );
    state.pendingDatasetLoadedId = item.id;
  } catch (error) {
    setError(error.message || String(error));
  } finally {
    state.pendingDatasetLoading = false;
    renderQueriesSelector();
    renderAddDatasetButton();
    renderOverview();
  }
}

async function refreshRuntimeResources() {
  setError("");
  const previousLabel = refreshLabel.textContent;
  refreshButton.disabled = true;
  refreshButton.setAttribute("aria-busy", "true");
  refreshLabel.textContent = "Refreshing";
  setStatus("Refreshing", "running");
  try {
    await Promise.all([
      loadRegistryDatasets(),
      loadOutputResults(),
      state.configMaterialized
        ? loadConfigInfo({ resetSelection: false })
        : Promise.resolve(),
    ]);
    refreshLabel.textContent = "Updated";
    setStatus("Ready", "idle");
    syncViewChrome();
  } catch (error) {
    refreshLabel.textContent = previousLabel || "Refresh";
    setStatus("Error", "error");
    setError(error.message || String(error));
  } finally {
    refreshButton.disabled = false;
    refreshButton.removeAttribute("aria-busy");
    window.setTimeout(() => {
      if (!refreshButton.disabled) {
        refreshLabel.textContent = "Refresh";
      }
    }, 1400);
  }
}

async function renderConfigDatasetDetail(item) {
  state.activeDatasetId = item.id;
  detailTitle.textContent = item.id;
  const itemPath = normalizeDatasetRoot(item.path || "");
  const registryItem = state.registryDatasets.find(
    (dataset) =>
      dataset.id === item.id ||
      normalizeDatasetRoot(dataset.path || dataset.root || "") === itemPath,
  );
  if (registryItem) {
    await loadDatasetDetail(registryItem, {
      activeId: item.id,
      displayId: item.id,
      sourceLabel: "config dataset",
    });
    return;
  }
  datasetDetail.innerHTML = `
    ${renderDatasetOverview(item, {}, [])}
    <div class="detail-grid">
      <div><span>Source</span><strong>config</strong></div>
      <div><span>Loader</span><strong>${escapeHtml(item.loader || "-")}</strong></div>
      <div><span>Root</span><strong>${escapeHtml(item.path || "-")}</strong></div>
    </div>
    <pre>${escapeHtml(JSON.stringify(item, null, 2))}</pre>
  `;
  renderDatasets();
}

async function loadDatasetDetail(item, options = {}) {
  if (item.isConfig) {
    await renderConfigDatasetDetail(item);
    return;
  }

  state.activeDatasetId = options.activeId || item.id;
  detailTitle.textContent = options.displayId || item.id;
  datasetDetail.innerHTML = '<div class="empty-state">Loading dataset detail...</div>';
  renderDatasets();

  try {
    const [detail, documents, schema, queries] = await Promise.all([
      fetchJson(`/api/datasets/${encodeURIComponent(item.id)}`),
      fetchJson(`/api/datasets/${encodeURIComponent(item.id)}/documents?limit=25`),
      fetchJson(`/api/datasets/${encodeURIComponent(item.id)}/schema`),
      fetchJson(`/api/datasets/${encodeURIComponent(item.id)}/queries`),
    ]);
    renderGeneratedConfig();
    datasetDetail.innerHTML = `
      ${renderDatasetOverview(detail, schema.schema || {}, queries.queries || [], options.sourceLabel)}
      <section class="detail-section">
        <h4>Documents</h4>
        ${renderRecordsTable(documents.documents || [], ["doc_id", "split", "doc_text"])}
      </section>
      <section class="detail-section">
        <h4>Schema</h4>
        ${renderSchema(schema.schema || {})}
      </section>
      <section class="detail-section">
        <h4>Queries</h4>
        ${renderQueries(queries.queries || [])}
      </section>
      <section class="detail-section">
        <h4>Manifest</h4>
        <pre>${escapeHtml(JSON.stringify(detail.manifest || {}, null, 2))}</pre>
      </section>
    `;
  } catch (error) {
    datasetDetail.innerHTML = `<div class="empty-state">${escapeHtml(error.message || String(error))}</div>`;
  }
}

function schemaStats(schema) {
  const tables = schema.tables || [];
  const fieldCount = tables.reduce((count, table) => count + (table.columns || []).length, 0);
  return {
    tables,
    tableCount: tables.length,
    fieldCount,
    tableNames: tables.map((table) => table.name || table.table_id || "table"),
  };
}

function renderDatasetOverview(detail, schema, queries, sourceLabel = null) {
  const stats = schemaStats(schema);
  const queryCount = detail.queries_count ?? queries.length;
  const tableLabel = stats.tableCount || detail.tables_count || "-";
  const fieldLabel = stats.fieldCount || "-";
  const tableNames = stats.tableNames.slice(0, 6);
  return `
    <section class="dataset-overview">
      <div class="overview-primary">
        <div>
          <span>Documents</span>
          <strong>${detail.documents_count ?? "-"}</strong>
        </div>
        <div>
          <span>Queries</span>
          <strong>${queryCount || "-"}</strong>
        </div>
        <div>
          <span>Schema</span>
          <strong>${tableLabel} tables</strong>
          <small>${fieldLabel} fields</small>
        </div>
        <div>
          <span>Source</span>
          <strong>${escapeHtml(sourceLabel || detail.kind || "dataset")}</strong>
        </div>
      </div>
      <div class="schema-overview">
        <span>Schema tables</span>
        <div>
          ${
            tableNames.length
              ? tableNames.map((name) => `<b>${escapeHtml(name)}</b>`).join("")
              : "<em>No schema tables loaded.</em>"
          }
        </div>
      </div>
    </section>
  `;
}

function renderRecordsTable(records, preferredColumns) {
  if (!records.length) {
    return '<div class="empty-state compact">No records.</div>';
  }
  const columns = preferredColumns.filter((column) => column in records[0]);
  if (!columns.length) {
    columns.push(...Object.keys(records[0]).slice(0, 4));
  }
  return `
    <div class="table-wrap">
      <table>
        <thead>
          <tr>${columns.map((column) => `<th>${escapeHtml(column)}</th>`).join("")}</tr>
        </thead>
        <tbody>
          ${records
            .map(
              (record) => `
                <tr>
                  ${columns
                    .map(
                      (column) =>
                        `<td data-column="${escapeHtml(column)}">${escapeHtml(truncate(record[column]))}</td>`,
                    )
                    .join("")}
                </tr>
              `,
            )
            .join("")}
        </tbody>
      </table>
    </div>
  `;
}

function renderSchema(schema) {
  const tables = schema.tables || [];
  if (!tables.length) {
    return '<div class="empty-state compact">No schema tables.</div>';
  }
  return tables
    .map(
      (table) => `
        <article class="schema-table">
          <strong>${escapeHtml(table.name || table.table_id || "table")}</strong>
          <p>${escapeHtml(table.description || "")}</p>
          <div>
            ${(table.columns || [])
              .map((column) => {
                const name = column.name || column.column_id || "field";
                const type = column.type ? `: ${column.type}` : "";
                return `<span>${escapeHtml(`${name}${type}`)}</span>`;
              })
              .join("")}
          </div>
        </article>
      `,
    )
    .join("");
}

function renderQueries(queries) {
  if (!queries.length) {
    return '<div class="empty-state compact">No explicit queries; extraction defaults to all schema attributes.</div>';
  }
  const items = queries
    .slice(0, 25)
    .map(
      (query) => `
        <article class="query-item">
          <strong>${escapeHtml(query.query_id || "query")}</strong>
          <p>${escapeHtml(query.question || "Default extraction: extract every attribute in the query-specific schema.")}</p>
          <code>${escapeHtml(truncate(query.sql || "", 220))}</code>
        </article>
      `,
    )
    .join("");
  return `<div class="query-detail-list">${items}</div>`;
}

function renderQueriesSelector() {
  if (state.datasetSource !== "all") {
    selectedQueryCount.textContent = "0";
    querySelectTrigger.disabled = true;
    querySelectLabel.textContent = "From loaded config";
    querySelectBadge.textContent = "config";
    setQueryDropdownOpen(false);
    queryList.innerHTML = '<div class="empty-state compact">Query selection is generated from the loaded config.</div>';
    return;
  }
  selectedQueryCount.textContent = String(state.pendingQueryIds.size);
  const copy = querySelectCopy();
  querySelectTrigger.disabled = state.pendingDatasetLoading || !state.activeDatasetQueries.length;
  querySelectLabel.textContent = copy.label;
  querySelectBadge.textContent = copy.badge;
  if (!state.activeDatasetQueries.length) {
    setQueryDropdownOpen(false);
    queryList.innerHTML = '<div class="empty-state compact">No explicit queries; extraction defaults to all schema attributes.</div>';
    return;
  }
  queryList.hidden = !state.queryDropdownOpen;
  querySelectTrigger.setAttribute("aria-expanded", String(state.queryDropdownOpen));
  syncSelectionMenuScrollLock();
  queryList.innerHTML = `
    <div class="selection-window-head">
      <div>
        <span>Query Selection</span>
        <strong>${escapeHtml(state.pendingDatasetId || "Dataset")}</strong>
      </div>
      <div class="selection-actions">
        <button type="button" data-query-action="clear">Clear</button>
        <button type="button" data-query-action="all">All</button>
        <button type="button" data-query-action="close">Done</button>
      </div>
    </div>
    <div class="selection-window-body">
      ${state.activeDatasetQueries
        .map((query) => {
          const queryId = String(query.query_id || "");
          return `
            <label class="query-option">
              <input type="checkbox" value="${escapeHtml(queryId)}" ${
                state.pendingQueryIds.has(queryId) ? "checked" : ""
              } />
              <span>
                <strong>${escapeHtml(queryId || "query")}</strong>
                <small>${escapeHtml(truncate(query.question || query.sql || "", 260))}</small>
              </span>
            </label>
          `;
        })
        .join("")}
    </div>
  `;
  queryList.querySelector("[data-query-action='close']")?.addEventListener("click", () => {
    setQueryDropdownOpen(false);
    querySelectTrigger.focus();
  });
  queryList.querySelector("[data-query-action='clear']")?.addEventListener("click", (event) => {
    event.stopPropagation();
    state.pendingQueryIds.clear();
    renderQueriesSelector();
    renderAddDatasetButton();
    renderOverview();
  });
  queryList.querySelector("[data-query-action='all']")?.addEventListener("click", (event) => {
    event.stopPropagation();
    state.pendingQueryIds = new Set(
      state.activeDatasetQueries.map((query) => String(query.query_id || "")).filter(Boolean),
    );
    renderQueriesSelector();
    renderAddDatasetButton();
    renderOverview();
  });
  queryList.querySelectorAll("input").forEach((input) => {
    input.addEventListener("change", (event) => {
      if (event.target.checked) {
        state.pendingQueryIds.add(event.target.value);
      } else {
        state.pendingQueryIds.delete(event.target.value);
      }
      renderQueriesSelector();
      renderAddDatasetButton();
      renderOverview();
    });
  });
  if (state.queryDropdownOpen) {
    scrollDropdownIntoView(queryList);
  }
}

function renderGeneratedConfig() {
  if (!state.configMaterialized) {
    configEmptyState.hidden = false;
    configWorkspace.hidden = true;
    configSummary.innerHTML = "";
    configSummary.hidden = true;
    generatedConfig.textContent = "";
    generatedConfig.hidden = true;
    copyConfig.disabled = true;
    return;
  }
  configEmptyState.hidden = true;
  configWorkspace.hidden = false;
  const config = generatedConfigObject();
  generatedConfig.textContent = toYaml(config);
  const basePath = state.configInfo?.resolved_config_path || configPath.value.trim() || "";
  const baseLabel = basePath ? basePath.split(/[\\/]/).pop() : "unsaved";
  const experimentId = config.experiments ? Object.keys(config.experiments)[0] : "-";
  const datasetCount = selectedDatasetIds().length;
  const stageCount = selectedStages().length;
  const configQueryCount = configuredQueryIds().length;
  const queryLabel =
    state.datasetSource === "config" ? configQueryCount || "all queries" : selectedQueryIds().length || "all queries";
  configSummary.innerHTML = `
    <span class="summary-chip" title="${escapeHtml(basePath || "Generated config")}">${escapeHtml(baseLabel)}</span>
    <span class="summary-chip">Exp ${escapeHtml(experimentId || "-")}</span>
    <span class="summary-chip">${datasetCount} ${datasetCount === 1 ? "dataset" : "datasets"}</span>
    <span class="summary-chip">${escapeHtml(String(queryLabel))}</span>
    <span class="summary-chip">${stageCount} ${stageCount === 1 ? "stage" : "stages"}</span>
  `;
  configSummary.hidden = false;
  generatedConfig.hidden = false;
  copyConfig.disabled = false;
}

function runButtonState() {
  if (state.runInFlight) {
    return {
      disabled: true,
      label: "Running Pipeline",
      title: "Run is in progress.",
    };
  }
  if (state.datasetSource === "config" && !state.configMaterialized) {
    return {
      disabled: true,
      label: "Load Config First",
      title: "Load the config before running this experiment.",
    };
  }
  if (!runDatasetIds().length) {
    if (pendingDatasetRequiresQueries() && !state.pendingQueryIds.size) {
      return {
        disabled: true,
        label: "Select Query",
        title: "Select at least one query for the chosen dataset.",
      };
    }
    return {
      disabled: true,
      label: "Select Dataset",
      title: "Choose or load at least one dataset before running.",
    };
  }
  if (!state.configMaterialized && !hasPendingRunDataset()) {
    return {
      disabled: true,
      label: "Load Config First",
      title: "Load the config or add a dataset before running.",
    };
  }
  return {
    disabled: false,
    label: "Execute Query Pipeline",
    title: "Run the selected datasets and queries.",
  };
}

function renderRunButtonState() {
  const stateInfo = runButtonState();
  runButton.disabled = stateInfo.disabled;
  runButton.textContent = stateInfo.label;
  runButton.title = stateInfo.title;
  runButton.dataset.runState = state.runInFlight ? "running" : stateInfo.disabled ? "blocked" : "ready";
  runButton.setAttribute("aria-disabled", String(stateInfo.disabled));
}

function renderOverview() {
  const selectedDatasetCount = runDatasetIds().length;
  const selectedQueryLabel = runQueryLabel();
  const payload = state.lastPayload || {};
  const hasRun = Boolean(payload.result);
  const resultRoot = payload.result || {};
  const optimizationSummary = optimizationPipelineSummary(payload.optimization_metrics || []);
  const evaluationSummary = payload.evaluation?.summary || {};
  const oracleAvg = optimizationSummary?.before ?? average(
    collectNumbers(resultRoot, [
      /exhaustive.*calls?/,
      /baseline.*calls?/,
      /bf.*calls?/,
      /oracle.*n_calls?/,
      /oracle.*invocations?/,
    ]),
  );
  const systemAvg = optimizationSummary?.after ?? average(
    collectNumbers(resultRoot, [
      /optimized.*calls?/,
      /pipeline.*calls?/,
      /system.*calls?/,
      /actual.*calls?/,
      /actual.*cost/,
      /llm.*calls?/,
      /n_calls?/,
    ]),
  );
  const savingsAvg = optimizationSummary?.reduction ?? average(
    collectNumbers(resultRoot, [/reduction/, /savings?/, /rho/, /cost_savings/]),
  );
  const recallAvg = evaluationSummary.sql_answer_recall ?? average(
    collectNumbers(resultRoot, [/empirical.*recall/, /final.*recall/, /^recall$/, /accuracy/]),
  );

  if (!state.runInFlight && statusPill.classList.contains("idle")) {
    setStatus(selectedDatasetCount ? "Ready" : "Setup", "idle");
  }
  overviewState.textContent = hasRun ? "Updated" : selectedDatasetCount ? "Ready" : "Setup";
  selectedCount.textContent = `${selectedDatasetCount} ${selectedDatasetCount === 1 ? "dataset" : "datasets"}`;
  overviewSelectedDatasets.textContent = selectedDatasetCount ? String(selectedDatasetCount) : "-";
  overviewSelectedQueries.textContent = selectedQueryLabel;
  if (!hasRun) {
    summaryDatasets.textContent = String(selectedDatasetCount || "-");
  }
  overviewRunDatasets.textContent = hasRun ? String(countRunDatasets(payload) || "-") : "-";
  overviewRunQueries.textContent = hasRun ? String(countRunQueries(payload) || "-") : "-";
  overviewOracleCost.textContent = formatMetric(oracleAvg);
  overviewSystemCost.textContent = formatMetric(systemAvg);
  overviewSavings.textContent = formatMetric(savingsAvg, { percent: true });
  overviewRecall.textContent = formatMetric(recallAvg, { percent: true });
  renderArchitectureFunnel({
    selectedDatasetCount,
    selectedQueryLabel,
    payload,
    hasRun,
    recallLabel: formatMetric(recallAvg, { percent: true }),
  });
  renderRelationPreview();
  renderRunButtonState();
  syncViewChrome();
}

function resetViewportScroll() {
  window.scrollTo({ top: 0, left: 0, behavior: "auto" });
  window.requestAnimationFrame(() => {
    window.scrollTo({ top: 0, left: 0, behavior: "auto" });
  });
  window.setTimeout(() => {
    window.scrollTo({ top: 0, left: 0, behavior: "auto" });
  }, 0);
}

function revealActiveWorkspace() {
  const workspace = document.querySelector(".workspace");
  if (!workspace) {
    resetViewportScroll();
    return;
  }
  workspace.scrollIntoView({ block: "start", inline: "nearest", behavior: "auto" });
  window.requestAnimationFrame(() => {
    workspace.scrollIntoView({ block: "start", inline: "nearest", behavior: "auto" });
  });
  window.setTimeout(() => {
    workspace.scrollIntoView({ block: "start", inline: "nearest", behavior: "auto" });
  }, 80);
}

function resetViewScrollTarget({ revealWorkspace = false } = {}) {
  if (revealWorkspace && window.matchMedia("(max-width: 900px)").matches) {
    revealActiveWorkspace();
    return;
  }
  resetViewportScroll();
}

function activeViewTitle(view = state.activeView) {
  if (view === "config") {
    return state.configMaterialized ? "Config loaded" : "Load a config";
  }
  if (view === "datasets") {
    return state.activeDatasetId ? "Dataset detail" : "Browse datasets";
  }
  if (view === "console") {
    if (!state.consoleRun) {
      return "Run console";
    }
    if (state.consoleRun.status === "running" || state.runInFlight) {
      return "Running pipeline";
    }
    if (state.consoleRun.status === "completed") {
      return "Run complete";
    }
    if (state.consoleRun.status === "failed") {
      return "Run failed";
    }
    return statusLabel(state.consoleRun.status);
  }
  if (view === "results") {
    return state.lastPayload?.result ? "Run complete" : "Saved results";
  }
  return "Workbench";
}

function syncViewChrome(view = state.activeView) {
  const labels = {
    config: "1. Configuration",
    datasets: "2. Dataset browser",
    console: "3. Run console",
    results: "4. Latent relational view (R_Q)",
  };
  if (viewEyebrow) {
    viewEyebrow.textContent = labels[view] || "Workbench";
  }
  runTitle.textContent = activeViewTitle(view);
}

function setActiveView(view, options = {}) {
  closeSelectionMenus();
  state.activeView = view;
  document.querySelectorAll("[data-view-button]").forEach((button) => {
    button.classList.toggle("active", button.dataset.viewButton === view);
  });
  document.querySelectorAll(".view-panel").forEach((panel) => {
    panel.classList.toggle("active", panel.id === `${view}-view`);
  });
  if (view === "datasets" && !state.activeDatasetId) {
    loadFirstSelectedDatasetDetail().catch((error) => {
      datasetDetail.innerHTML = `<div class="empty-state">${escapeHtml(error.message || String(error))}</div>`;
    });
  }
  if (view === "config") {
    scheduleAutoGrowTextareas(configWorkspace);
  }
  syncViewChrome(view);
  resetViewScrollTarget(options);
}

function setActiveConfigSection(section) {
  closeSelectionMenus();
  state.activeConfigSection = section;
  configSectionButtons.forEach((button) => {
    button.classList.toggle("active", button.dataset.configSectionButton === section);
  });
  configSectionPages.forEach((page) => {
    page.classList.toggle("active", page.dataset.configSectionPage === section);
  });
  scheduleAutoGrowTextareas(configWorkspace);
}

async function loadFirstSelectedDatasetDetail() {
  const firstDatasetId = selectedDatasetIds()[0];
  if (!firstDatasetId) {
    return;
  }
  const item = datasetItems().find((dataset) => dataset.id === firstDatasetId);
  if (item) {
    await loadDatasetDetail(item);
  }
}

function setActivePage(page) {
  closeSelectionMenus();
  state.activePage = page;
  document.querySelectorAll("[data-page-button]").forEach((button) => {
    button.classList.toggle("active", button.dataset.pageButton === page);
  });
  document.querySelectorAll(".app-page").forEach((panel) => {
    panel.classList.toggle("active", panel.id === `${page}-page`);
  });
  resetViewportScroll();
}

function applyTheme(theme) {
  const resolvedTheme = theme === "dark" ? "dark" : "light";
  document.documentElement.dataset.theme = resolvedTheme;
  localStorage.setItem(THEME_STORAGE_KEY, resolvedTheme);
  const isDark = resolvedTheme === "dark";
  themeToggle.setAttribute("aria-pressed", String(isDark));
  themeToggle.classList.toggle("is-dark", isDark);
  themeLabel.textContent = isDark ? "Light" : "Dark";
}

function renderPaperViewer() {
  const paper = paperCatalog[state.selectedPaper] || paperCatalog.paper;
  const previewHref = paper.href;
  const downloadHref = paper.downloadHref || paper.href;
  const fileLabel = downloadHref.split("/").pop() || "PDF";
  paperTabs.forEach((tab) => {
    tab.classList.toggle("active", tab.dataset.paper === state.selectedPaper);
  });
  paperViewerMount.innerHTML = `
    <article class="paper-card">
      <h3>${escapeHtml(paper.title)}</h3>
      <p>${escapeHtml(paper.description)}</p>
      <div class="paper-file-panel">
        <span>PDF</span>
        <strong>${escapeHtml(fileLabel)}</strong>
        <p>Preview opens as a document view so the demo shell stays responsive across embedded browsers.</p>
      </div>
      <div class="paper-actions">
        <a class="pill-link" href="${escapeHtml(previewHref)}">Open PDF</a>
        <a class="pill-link" href="${escapeHtml(downloadHref)}" download>Download</a>
      </div>
    </article>
  `;
}

function renderPaperExperiments() {
  if (!paperExperimentList) {
    return;
  }
  const experiments = state.paperExperiments || [];
  if (!experiments.length) {
    paperExperimentList.innerHTML = '<div class="empty-state">No web paper experiments are registered.</div>';
    return;
  }
  paperExperimentList.innerHTML = experiments
    .map((item) => {
      const readyCount = (item.evidence || []).filter((evidence) => evidence.exists).length;
      const totalCount = (item.evidence || []).length;
      const evidenceCopy = totalCount ? `${readyCount}/${totalCount} artifacts` : "No artifacts yet";
      const stepCopy = (item.steps || []).map((step) => step.label).join(" -> ");
      return `
        <article class="paper-experiment-card" data-paper-experiment-id="${escapeHtml(item.id)}">
          <header>
            <div>
              <span>${escapeHtml(item.id)}</span>
              <h3>${escapeHtml(item.title)}</h3>
            </div>
            <strong class="${item.evidence_ready ? "ready" : "pending"}">${escapeHtml(evidenceCopy)}</strong>
          </header>
          <p>${escapeHtml(item.description)}</p>
          ${item.config_path ? `<code>${escapeHtml(item.config_path)}</code>` : ""}
          <small>${escapeHtml(stepCopy || "No runnable steps")}</small>
          <button type="button" data-paper-experiment-run="${escapeHtml(item.id)}">Run</button>
        </article>
      `;
    })
    .join("");
  paperExperimentList.querySelectorAll("[data-paper-experiment-run]").forEach((button) => {
    button.addEventListener("click", () => {
      runPaperExperiment(button.dataset.paperExperimentRun).catch((error) => {
        setStatus("Error", "error");
        setError(error.message || String(error));
      });
    });
  });
}

async function loadPaperExperiments() {
  const payload = await fetchJson("/api/paper-experiments");
  state.paperExperiments = payload.experiments || [];
  state.paperExperimentOutputRoot = payload.output_root || "";
  renderPaperExperiments();
}

function paperExperimentSteps(experimentId) {
  const item = (state.paperExperiments || []).find((experimentItem) => experimentItem.id === experimentId);
  const steps = item?.steps || [];
  return steps.length
    ? steps.map((step) => ({ id: step.id, label: step.label }))
    : [{ id: experimentId, label: item?.title || experimentId }];
}

async function runPaperExperiment(experimentId) {
  if (!experimentId) {
    return;
  }
  setError("");
  setStatus("Running", "running");
  runTitle.textContent = "Running paper experiment";
  const steps = paperExperimentSteps(experimentId);
  resetConsoleRun(steps);
  setActivePage("workbench");
  setActiveView("console", { revealWorkspace: true });
  const payload = await startStreamingRun(
    {},
    {
      endpoint: `/api/paper-experiments/${encodeURIComponent(experimentId)}/runs`,
      steps,
    },
  );
  await loadPaperExperiments();
  await loadOutputResults();
  setStatus("Success", "success");
  runTitle.textContent = "Paper experiment complete";
  setActiveView("results", { revealWorkspace: true });
  renderPayload(payload);
}

async function loadDefaults() {
  const defaults = await fetchJson("/api/defaults");
  state.defaults = defaults;
  configPath.value = defaults.config_path || "";
  syncControlTitle(configPath);
  renderApiKeyStatus(defaults.api_key_status);
  const defaultExperiment = defaults.experiment || "demo";
  experiment.innerHTML = `<option value="${escapeHtml(defaultExperiment)}">${escapeHtml(defaultExperiment)}</option>`;
  experiment.value = defaultExperiment;
  syncControlTitle(experiment);
  document.querySelectorAll('input[name="stages"]').forEach((item) => {
    item.checked = (defaults.default_stages || []).includes(item.value);
  });
  await Promise.all([loadRegistryDatasets(), loadOutputResults(), loadPaperExperiments()]);
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  setError("");
  const datasetIdsForRun = runDatasetIds();
  if (!datasetIdsForRun.length) {
    if (pendingDatasetRequiresQueries() && !state.pendingQueryIds.size) {
      setError("Select at least one query.");
      return;
    }
    setError("Select at least one dataset.");
    return;
  }
  if (!state.configMaterialized && !hasPendingRunDataset()) {
    setError("Load a config or add a dataset to config first.");
    return;
  }
  setStatus("Running", "running");
  state.runInFlight = true;
  renderRunButtonState();
  resetConsoleRun();
  setActiveView("console", { revealWorkspace: true });

  const config = generatedConfigObject({ forRun: true });
  const generatedExperiment = Object.keys(config.experiments || {})[0] || experiment.value.trim();
  const payload = {
    config_path: configPath.value.trim(),
    experiment: generatedExperiment,
    config,
    stages: selectedStages(),
    datasets: datasetIdsForRun,
    query_ids: hasPendingRunDataset() ? runQueryIds() : [],
    api_key: apiKey.value.trim(),
  };

  try {
    await startStreamingRun(payload);
    setStatus("Success", "success");
    setActiveView("results", { revealWorkspace: true });
  } catch (error) {
    setStatus("Error", "error");
    setError(error.message || String(error));
  } finally {
    state.runInFlight = false;
    renderOverview();
  }
});

refreshButton.addEventListener("click", () => {
  refreshRuntimeResources();
});

paperExperimentRefresh?.addEventListener("click", async () => {
  setError("");
  paperExperimentRefresh.disabled = true;
  try {
    await loadPaperExperiments();
  } catch (error) {
    setError(error.message || String(error));
  } finally {
    paperExperimentRefresh.disabled = false;
  }
});

paperExperimentRunAll?.addEventListener("click", async () => {
  paperExperimentRunAll.disabled = true;
  try {
    await runPaperExperiment("all_paper_analogous");
  } catch (error) {
    setStatus("Error", "error");
    setError(error.message || String(error));
  } finally {
    paperExperimentRunAll.disabled = false;
  }
});

loadConfig.addEventListener("click", async () => {
  setError("");
  setStatus("Loading", "running");
  try {
    await loadConfigInfo({ resetSelection: true });
    state.configMaterialized = true;
    renderGeneratedConfig();
    setActiveView("config");
    setStatus("Ready", "idle");
    syncViewChrome();
  } catch (error) {
    setStatus("Error", "error");
    setError(error.message || String(error));
  }
});

document.querySelectorAll("[data-source]").forEach((button) => {
  button.addEventListener("click", () => {
    state.datasetSource = button.dataset.source;
    closeSelectionMenus();
    document.querySelectorAll("[data-source]").forEach((item) => {
      item.classList.toggle("active", item === button);
    });
    if (state.datasetSource !== "all") {
      state.selectedQueryIds.clear();
      state.pendingDatasetId = "";
      state.pendingDatasetLoadedId = "";
      state.pendingQueryIds.clear();
      state.pendingDatasetLoading = false;
      state.activeDatasetQueries = [];
    }
    renderDatasetPicker(datasetSelectItems());
    renderQueriesSelector();
    renderDatasets();
    renderOverview();
  });
});

document.querySelectorAll("[data-view-button]").forEach((button) => {
  button.addEventListener("click", () => setActiveView(button.dataset.viewButton, { revealWorkspace: true }));
});

configSectionButtons.forEach((button) => {
  button.addEventListener("click", () => setActiveConfigSection(button.dataset.configSectionButton));
});

configDatasetList.addEventListener("change", (event) => {
  const target = event.target;
  const card = target.closest("[data-dataset-id]");
  if (!card) {
    return;
  }
  if (target.dataset.datasetAction === "use") {
    toggleDataset(card.dataset.datasetId, target.checked);
    return;
  }
  if (target.dataset.datasetField) {
    updateDatasetDraftFromControl(target);
  }
});

configDatasetList.addEventListener("input", (event) => {
  if (event.target.dataset.datasetField) {
    updateDatasetDraftFromControl(event.target);
  }
});

configDatasetList.addEventListener("click", (event) => {
  const target = event.target;
  if (target.dataset.datasetAction !== "remove") {
    return;
  }
  const card = target.closest("[data-dataset-id]");
  if (card) {
    removeDatasetFromConfig(card.dataset.datasetId);
  }
});

document.querySelectorAll("[data-page-button]").forEach((button) => {
  button.addEventListener("click", () => setActivePage(button.dataset.pageButton));
});

themeToggle.addEventListener("click", () => {
  const nextTheme = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
  applyTheme(nextTheme);
});

querySelectTrigger.addEventListener("click", () => {
  if (querySelectTrigger.disabled) {
    return;
  }
  setQueryDropdownOpen(!state.queryDropdownOpen);
});

datasetPickerTrigger.addEventListener("click", () => {
  if (datasetPickerTrigger.disabled) {
    return;
  }
  setDatasetPickerOpen(!state.datasetPickerOpen);
});

document.addEventListener("click", (event) => {
  if (state.datasetPickerOpen && !selectionMenuContainsEvent(event, datasetPickerTrigger, datasetPickerList)) {
    setDatasetPickerOpen(false);
  }
  if (state.queryDropdownOpen && !selectionMenuContainsEvent(event, querySelectTrigger, queryList)) {
    setQueryDropdownOpen(false);
  }
});

document.addEventListener("keydown", (event) => {
  if (event.key !== "Escape") {
    return;
  }
  if (state.datasetPickerOpen) {
    setDatasetPickerOpen(false);
    datasetPickerTrigger.focus();
    return;
  }
  if (state.queryDropdownOpen) {
    setQueryDropdownOpen(false);
    querySelectTrigger.focus();
  }
});

paperTabs.forEach((tab) => {
  tab.addEventListener("click", () => {
    state.selectedPaper = tab.dataset.paper;
    renderPaperViewer();
  });
});

datasetSelect.addEventListener("change", () => {
  const datasetId = datasetSelect.value;
  if (!datasetId) {
    state.pendingDatasetId = "";
    state.pendingDatasetLoadedId = "";
    state.pendingQueryIds.clear();
    state.datasetPickerOpen = false;
    state.queryDropdownOpen = false;
    state.pendingDatasetLoading = false;
    state.activeDatasetQueries = [];
    renderDatasetPicker(datasetSelectItems());
    renderQueriesSelector();
    renderAddDatasetButton();
    renderOverview();
    return;
  }
  const item = state.registryDatasets.find((dataset) => dataset.id === datasetId);
  if (!item) {
    return;
  }
  state.pendingDatasetId = datasetId;
  state.pendingDatasetLoadedId = "";
  state.pendingQueryIds.clear();
  state.datasetPickerOpen = false;
  state.queryDropdownOpen = false;
  state.activeDatasetQueries = [];
  renderDatasetPicker(datasetSelectItems());
  renderQueriesSelector();
  renderAddDatasetButton();
  renderOverview();
  loadPendingDatasetQueries(item);
});
addDataset.addEventListener("click", () => {
  if (addDataset.disabled || !state.pendingDatasetId) {
    return;
  }
  if (!state.configDraft) {
    state.configDraft = defaultConfigDraft(state.configInfo);
    setConfigEditorFromDraft();
  }
  state.configDatasetIds.add(state.pendingDatasetId);
  const draft = ensureDatasetDraft(state.pendingDatasetId);
  if (state.pendingQueryIds.size) {
    draft.query_ids = Array.from(state.pendingQueryIds);
  } else {
    delete draft.query_ids;
  }
  state.selectedDatasetIds.add(state.pendingDatasetId);
  state.selectedQueryIds = new Set(
    selectedDatasetIds().flatMap((id) => ensureDatasetDraft(id).query_ids || []),
  );
  state.queryDropdownOpen = false;
  state.configMaterialized = true;
  renderQueriesSelector();
  renderSelectedDatasets();
  renderGeneratedConfig();
  renderOverview();
  setActiveView("config");
});
apiKey.addEventListener("input", () => {
  renderApiKeyStatus();
  renderGeneratedConfig();
});
[
  configProjectName,
  configProjectSeed,
  configExperimentId,
  configOutputDir,
  configLogDir,
  configArtifactId,
  configOutputLayout,
  configConsoleLogLevel,
  forceRerun,
  configStagesJson,
].forEach((input) => {
  const eventName = input.tagName === "SELECT" || input.type === "checkbox" ? "change" : "input";
  input.addEventListener(eventName, updateConfigDraftFromEditor);
});
[
  configOptDocFilterEnabled,
  configOptSchemaAdaptiveEnabled,
  configOptSchemaAdaptiveThreshold,
  configOptSchemaAdaptiveStreak,
  configOptSchemaAdaptiveMinDocs,
  configOptProxyEnabled,
  configOptProxyMode,
  configOptProxyModel,
  configOptProxyLearned,
  configOptProxyEmbedding,
  configOptProxyFallback,
  configOptProxyEpochs,
  configOptJoinEnabled,
].forEach((input) => {
  const eventName = input.tagName === "SELECT" || input.type === "checkbox" ? "change" : "input";
  input.addEventListener(eventName, () => {
    if (input === configOptProxyEnabled && configOptProxyEnabled.checked) {
      configOptJoinEnabled.checked = true;
    }
    updateOptimizationControlState();
    updateConfigDraftFromEditor();
  });
});
[
  ["llm", configLlmEnabled, "change", handleModelEnabledChange],
  ["llm", configLlmProvider, "change", handleModelProviderChange],
  ["llm", configLlmModel, "change", updateConfigDraftFromEditor],
  ["llm", configLlmCustomModel, "input", updateConfigDraftFromEditor],
  ["llm", configLlmApiKeyEnv, "input", updateConfigDraftFromEditor],
  ["llm", configLlmBaseUrl, "input", updateConfigDraftFromEditor],
  ["llm", configLlmStructuredBackend, "change", updateConfigDraftFromEditor],
  ["llm", configLlmMaxRetries, "input", updateConfigDraftFromEditor],
  ["llm", configLlmWaitTime, "input", updateConfigDraftFromEditor],
  ["llm", configLlmTemperature, "input", updateConfigDraftFromEditor],
  ["llm", configLlmTopP, "input", updateConfigDraftFromEditor],
  ["llm", configLlmMaxTokens, "input", updateConfigDraftFromEditor],
  ["llm", configLlmLocalModelPath, "input", updateConfigDraftFromEditor],
  ["embedding", configEmbeddingEnabled, "change", handleModelEnabledChange],
  ["embedding", configEmbeddingProvider, "change", handleModelProviderChange],
  ["embedding", configEmbeddingModel, "change", updateConfigDraftFromEditor],
  ["embedding", configEmbeddingCustomModel, "input", updateConfigDraftFromEditor],
  ["embedding", configEmbeddingApiKeyEnv, "input", updateConfigDraftFromEditor],
  ["embedding", configEmbeddingBaseUrl, "input", updateConfigDraftFromEditor],
  ["embedding", configEmbeddingBatchSize, "input", updateConfigDraftFromEditor],
  ["embedding", configEmbeddingStorageFile, "input", updateConfigDraftFromEditor],
].forEach(([kind, input, eventName, handler]) => {
  input.addEventListener(eventName, () => {
    if (input === configLlmModel || input === configEmbeddingModel) {
      updateModelCustomField(kind);
    }
    handler(kind);
  });
});
document.querySelectorAll('input[name="stages"]').forEach((input) => {
  input.addEventListener("change", () => {
    state.configMaterialized = true;
    updateOptimizationControlState();
    renderGeneratedConfig();
    renderOverview();
  });
});
experiment.addEventListener("change", () => {
  loadConfigInfo({ resetSelection: true }).catch((error) => {
    setStatus("Error", "error");
    setError(error.message || String(error));
  });
});

copyConfig.addEventListener("click", async () => {
  const copied = await copyTextToClipboard(generatedConfig.textContent || "");
  showCopyFeedback(copyConfig, copied);
});

clearConsoleLogs.addEventListener("click", () => {
  if (!state.consoleRun) {
    return;
  }
  state.consoleRun.logs = [];
  renderConsoleLogs();
});

copyConsoleLogs.addEventListener("click", async () => {
  const text = state.consoleRun
    ? state.consoleRun.logs.map((entry) => `[${entry.time}] ${entry.message}`).join("\n")
    : "";
  const copied = await copyTextToClipboard(text);
  showCopyFeedback(copyConsoleLogs, copied);
});

architectureFunnelToggle?.addEventListener("click", () => {
  setArchitectureFunnelOpen(architectureFunnelBody?.hidden);
});

document.addEventListener("input", (event) => {
  if (event.target?.matches?.("input, select, textarea")) {
    syncControlTitle(event.target);
    if (event.target.tagName === "TEXTAREA") {
      autoGrowTextarea(event.target);
    }
  }
});

document.addEventListener("change", (event) => {
  if (event.target?.matches?.("input, select, textarea")) {
    syncControlTitle(event.target);
    if (event.target.tagName === "TEXTAREA") {
      autoGrowTextarea(event.target);
    }
  }
});

window.addEventListener("resize", () => {
  if (!errorPanel.hidden) {
    revealErrorPanel();
  }
  autoGrowTextareas();
});

if ("scrollRestoration" in window.history) {
  window.history.scrollRestoration = "manual";
}
resetViewportScroll();
applyTheme(localStorage.getItem(THEME_STORAGE_KEY) || "light");
setArchitectureFunnelOpen(true);
renderPaperViewer();
renderQueriesSelector();
renderGeneratedConfig();
renderOutputResults();
renderConsoleRun();
setActiveConfigSection(state.activeConfigSection);
renderOverview();

loadDefaults().catch((error) => {
  setStatus("Error", "error");
  setError(error.message || String(error));
});
