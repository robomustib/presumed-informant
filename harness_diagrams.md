# Program flow of harness.py

The following diagrams describe the registered state of `harness.py`.

## 1. Overall flow

```mermaid
flowchart TD
    A([Start: main argv]) --> B["Parse arguments<br/>config, --timepoint, --dry-run,<br/>--pilot, --only, --env, --log-level"]
    B --> C[Configure logging]
    C --> D{Load .env<br/>load_dotenv}
    D -->|variables set| E[Check permissions<br/>check_dotenv_permissions]
    D -->|"nothing loaded + --env"| D1[Warning: no .env]
    D -->|nothing loaded, default path| F
    E --> F[load_config: build RunConfig]
    D1 --> F
    F --> G{"--only set?"}
    G -->|yes| H{"--pilot?"}
    H -->|no| H1([Error: allowed in pilot<br/>only, exit 2])
    H -->|yes| H2{unknown<br/>model keys?}
    H2 -->|yes| H3([Error: unknown keys, exit 2])
    H2 -->|no| I[Filter models]
    G -->|no| J[build_plan: list of cells]
    I --> J
    J --> K[Log: conversations, expected calls,<br/>cascade variants and hashes]
    K --> L[find_placeholders]
    L --> M{Placeholders<br/>found?}
    M -->|yes + no dry run| M1([Error: config incomplete,<br/>exit 2])
    M -->|yes + dry run| M2[Warning: placeholders]
    M -->|no| N{"--dry-run?"}
    M2 --> N
    N -->|yes| O[Print first 20 cells<br/>plus number of remaining] --> O1([exit 0])
    N -->|no| P[check_environment:<br/>API keys present?]
    P -->|missing| P1([Error: keys missing, exit 2])
    P -->|ok| Q[run_all: asyncio.run]
    Q --> R[Count status<br/>Counter c.status]
    R --> S{Aborts?}
    S -->|yes| S1[Warning: failure rate]
    S -->|no| T{Truncated/empty?}
    S1 --> T
    T -->|yes| T1[Warning: raise token budget]
    T -->|no| U([exit 0])
    T1 --> U
```

## 2. Loading the configuration (`load_config`)

```mermaid
flowchart TD
    A([load_config path]) --> B[Read YAML]
    B --> C[Build ModelSpec list]
    C --> D[Build TopicSpec list]
    D --> E{Topics without<br/>premise?}
    E -->|yes| E1[Warning: P3 not codable]
    E -->|no| F[Build SettingSpec list<br/>default: one setting named default]
    E1 --> F
    F --> G[Read cascade variants<br/>default: A]
    G --> H{Unknown<br/>variant?}
    H -->|yes| H1([ValueError])
    H -->|no| I[Return RunConfig<br/>defaults for timeout, concurrency,<br/>retries and pause]
```

## 3. Running one cell (`run_cell`)

```mermaid
flowchart TD
    A([run_cell cell, cfg, timepoint,<br/>client, sem, writer, run_label]) --> B[Initialise Conversation object<br/>run_id, metadata, params]
    B --> C[build_prompt_sequence:<br/>steps from CASCADES]
    C --> D[Acquire semaphore]
    D --> E[Loop over steps]
    E --> F[Append user prompt to history]
    F --> G[client.chat history, params]
    G --> H[Create Turn object<br/>and append to conv.turns]
    H --> I{result.ok?}
    I -->|no| J[Set status via<br/>STATUS_BY_FAILURE<br/>aborted_truncated/empty/blocked/error<br/>log error]
    J --> K[break]
    I -->|yes| L{reported_model<br/>already set?}
    L -->|no| M[Set conv.reported_model]
    L -->|yes| N[Append assistant answer<br/>to history]
    M --> N
    N --> O{pause_between_<br/>turns_s > 0?}
    O -->|yes| P[sleep]
    O -->|no| Q{more<br/>steps?}
    P --> Q
    Q -->|yes| E
    Q -->|no| R[Release semaphore]
    K --> R
    R --> S[Set finished_at]
    S --> T[writer.write conv]
    T --> U([Return Conversation])
```

## 4. Provider client with retry logic (`ProviderClient.chat`)

```mermaid
flowchart TD
    A([chat history, params]) --> B[_request: URL, payload, headers per provider<br/>model parameters merged with setting]
    B --> C[attempt = 0]
    C --> D{attempt below<br/>max_retries?}
    D -->|no| Z[Return error result<br/>ok=False, failure_kind<br/>transport, http or format]
    D -->|yes| E[attempt += 1]
    E --> F[POST via httpx]
    F -->|timeout or<br/>connection error| X1[ProviderError<br/>kind=transport, retryable]
    F --> G{HTTP status >= 400?}
    G -->|yes| H[ProviderError kind=http<br/>retryable if status in RETRY_STATUS]
    G -->|no| I[Parse r.json, _extract_text,<br/>_finish_reason]
    I -->|unexpected format| X2[ProviderError<br/>kind=format, not retryable]
    I --> L{finish_reason in<br/>TRUNCATED_REASONS?}
    L -->|yes| L1[ok=False, failure_kind=truncated]
    L -->|no| M{in BLOCKED_REASONS?}
    M -->|yes| M1[ok=False, failure_kind=blocked]
    M -->|no| N{text empty?}
    N -->|yes| N1[ok=False, failure_kind=empty]
    N -->|no| O[ok=True, metadata<br/>usage, fingerprint, reported_model]
    L1 --> P([Return result,<br/>content failures not retried])
    M1 --> P
    N1 --> P
    O --> P
    H --> Q{retryable?}
    X1 --> Q
    X2 --> Q
    Q -->|no| Z
    Q -->|yes| R[backoff = min 2^attempt + random 0 to 1, 60 s<br/>log warning]
    R --> S[sleep backoff] --> D
```

## 5. Data model (classes and relations)

```mermaid
classDiagram
    class RunConfig {
        models: list~ModelSpec~
        topics: list~TopicSpec~
        settings: list~SettingSpec~
        cascade_variants: list~str~
        repetitions: int
        output_dir: Path
        request_timeout: float
        max_concurrency_per_provider: int
        max_retries: int
        pause_between_turns_s: float
    }
    class ModelSpec {
        key: str
        provider: str
        model_id: str
        version_label: str
        api_base: str
        params: dict
    }
    class TopicSpec {
        key: str
        label: str
        initial_question: str
        counter_claim: str
        premise: str
    }
    class SettingSpec {
        key: str
        params: dict
        by_provider: dict
        for_provider(provider)
    }
    class Cell {
        model: ModelSpec
        topic: TopicSpec
        setting: SettingSpec
        variant: str
        repetition: int
    }
    class Conversation {
        run_id: str
        timepoint: str
        model_key: str
        model_id: str
        version_label: str
        reported_model: str
        provider: str
        topic_key: str
        topic_label: str
        premise: str
        cascade_variant: str
        cascade_hash: str
        setting_key: str
        params: dict
        repetition: int
        status: str
        run_label: str
        harness_version: str
        started_at: str
        finished_at: str
        turns: list~Turn~
        to_record()
    }
    class Turn {
        turn_index: int
        step: str
        prompt: str
        response_text: str
        ok: bool
        error: str
        http_status: int
        attempts: int
        elapsed_s: float
        reported_model: str
        fingerprint: str
        finish_reason: str
        usage: dict
        failure_kind: str
        timestamp: str
        raw_response: dict
    }
    class ProviderClient {
        spec: ModelSpec
        max_retries: int
        chat(history, params)
        aclose()
        _request(history, params)
        _render_messages(history)
        _extract_text(provider, raw)
        _finish_reason(provider, raw)
        _reported_model(provider, raw)
        _fingerprint(provider, raw)
        _usage(provider, raw)
    }
    class ProviderError {
        status: int
        retryable: bool
        kind: str
    }
    class JsonlWriter {
        path: Path
        write(conv)
        close()
    }

    RunConfig "1" --> "*" ModelSpec
    RunConfig "1" --> "*" TopicSpec
    RunConfig "1" --> "*" SettingSpec
    Cell --> ModelSpec
    Cell --> TopicSpec
    Cell --> SettingSpec
    Conversation ..> Cell : created from
    Conversation "1" --> "1..5" Turn
    ProviderClient --> ModelSpec
    ProviderClient ..> ProviderError : raises
    JsonlWriter ..> Conversation : writes
```

## 6. Plan generation (`build_plan`), Cartesian product

```mermaid
flowchart LR
    A[models] --> P[build_plan]
    B[topics] --> P
    C[settings] --> P
    D[cascade_variants] --> P
    E[repetitions 1..n] --> P
    P --> R[list of cells]
    R --> R1["Cell(model, topic, setting,<br/>variant, repetition)"]
```

## 7. Cascade variants (prompt sequence)

```mermaid
flowchart LR
    subgraph VA["Variant A, hash 986cd32be62ff788"]
        A1[initial] --> A2[justification] --> A3[source] --> A4["challenge<br/>I think your statement is wrong. + counter"] --> A5[revision]
    end
    subgraph VB["Variant B, hash 6e834fbe076fc802"]
        B1[initial] --> B2[justification] --> B3[source] --> B4["challenge<br/>I see this differently. + counter"] --> B5[revision]
    end
    A1 -.->|initial_question| T[TopicSpec]
    A4 -.->|counter_claim| T
    B1 -.->|initial_question| T
    B4 -.->|counter_claim| T
```

## 8. Run of one point in time (`run_all`)

```mermaid
flowchart TD
    A([run_all cfg, timepoint, run_label]) --> B[Set file name<br/>pilot_ or conversations_<br/>+ timepoint + timestamp .jsonl]
    B --> C[Open JsonlWriter]
    C --> D[One ProviderClient per model]
    D --> E[One semaphore per provider<br/>max_concurrency_per_provider]
    E --> F[build_plan: one task per cell]
    F --> G[asyncio.gather: all run_cell,<br/>at most n conversations per provider at once]
    G --> H[Write each finished conversation at once<br/>as one JSONL line, with lock]
    H --> I[Close clients, close writer]
    I --> J([Return path and conversations])
```
