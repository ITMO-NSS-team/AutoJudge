# План полного интерфейса AutoJudge

## 1. Цель интерфейса

Интерфейс должен объединить весь фактический функционал AutoJudge в один управляемый workflow:

- загрузка и нормализация трассы;
- настройка taxonomy, output schema и examples;
- генерация и проверка judge pool;
- генерация или ручная настройка DAG;
- выполнение судей и агрегатора;
- изучение verdict, evidence и node traces;
- контроль токенов, стоимости и Langfuse;
- обработка длинных трасс через summary + retrieval;
- batch-запуски и resume;
- prompt и graph optimization;
- экспорт результатов и артефактов.

UI не должен скрывать исследовательскую природу проекта: стабильные функции и экспериментальные возможности нужно явно различать.

## 2. Основные пользователи

### Evaluation researcher

Настраивает taxonomy и schema, запускает benchmarks, сравнивает результаты, модели и judge graphs.

### MAS developer

Загружает отдельную проблемную трассу, ищет виновного агента и шаг, изучает evidence и рекомендации.

### Platform operator

Контролирует batch-прогоны, ошибки, API usage, стоимость, Langfuse и PostgreSQL retrieval.

## 3. Главная навигация

```text
AutoJudge
├── Overview
├── Evaluations
│   ├── New evaluation
│   ├── Runs
│   └── Compare
├── Judge Studio
│   ├── Taxonomies
│   ├── Output schemas
│   ├── Examples
│   ├── Judge pools
│   └── Graphs
├── Batches
│   ├── New batch
│   ├── Active
│   └── History
├── Optimizer Lab
│   ├── Prompt optimizer
│   └── Graph evolution
├── Data
│   ├── Traces
│   ├── Summaries
│   └── Retrieval store
├── Observability
│   ├── Usage & cost
│   ├── Artifacts
│   └── Langfuse
└── Settings
```

На MVP часть этих разделов может быть представлена вкладками внутри одного Streamlit-приложения. Информационная архитектура должна остаться такой же, чтобы позже перейти на полноценный frontend без переработки пользовательской модели.

## 4. Глобальная оболочка

### Верхняя панель

- логотип AutoJudge;
- имя текущего workspace или benchmark;
- глобальный статус backend;
- OpenRouter connection indicator;
- Langfuse indicator;
- PostgreSQL indicator;
- текущий budget;
- меню пользователя и settings.

### Левая навигация

- основные разделы;
- быстрый переход к активным runs;
- счётчик требующих внимания ошибок;
- сворачивание до иконок.

### Глобальные статусы

| Статус | Значение |
|---|---|
| Draft | Конфигурация ещё не запускалась |
| Generating pool | PoolGenerator выполняется |
| Pool review | Требуется проверка judge pool |
| Building graph | Создаётся и валидируется DAG |
| Ready | Конфигурация валидна |
| Running | Pipeline выполняется |
| Partial | Есть результаты, но часть узлов упала |
| Completed | Pipeline и aggregator завершены |
| Failed | Результат не сформирован |
| Cancelled | Запуск остановлен пользователем |
| Stale | Входы изменены после генерации |

## 5. Overview

Главный экран отвечает на три вопроса: что выполняется, что требует внимания и сколько это стоит.

### KPI

- runs today;
- success rate pipeline;
- verdict distribution;
- failed judge nodes;
- total tokens;
- total cost;
- active batches;
- runs requiring human review.

### Основные блоки

- активные и последние runs;
- расходы по моделям;
- частые failure categories;
- проблемные benchmarks;
- состояние OpenRouter, Langfuse и PostgreSQL;
- быстрые действия **New evaluation** и **New batch**.

Важно разделять pipeline success и содержательный verdict: успешно выполненный pipeline может вернуть отрицательную оценку трассы.

## 6. New Evaluation — мастер одиночной оценки

Мастер состоит из шести явных этапов.

```text
1 Trace → 2 Evaluation Design → 3 Judge Pool → 4 Graph → 5 Run → 6 Verdict
```

### Этап 1 — Trace

#### Источники

- загрузка JSON, JSONL или CSV;
- вставка JSON/text;
- выбор ранее загруженной трассы;
- выбор benchmark example;
- ввод file path для локального режима.

#### Preview

- trace ID;
- исходный query;
- число шагов;
- роли и агенты;
- tool calls;
- объём текста и оценка токенов;
- обнаруженный формат;
- предупреждения нормализации.

#### Режим контекста

- **Full trace**;
- **Summary only**;
- **Summary + retrieval**;
- **Auto** на основании длины и budget.

При выборе retrieval UI проверяет PostgreSQL и показывает, сколько состояний сохранено и доступно по `state_id`.

### Этап 2 — Evaluation Design

#### Task

- описание evaluation objective;
- goal исходной системы;
- ground truth, если есть;
- benchmark adapter.

#### Taxonomy

- выбор сохранённой taxonomy;
- tree editor;
- Markdown/text mode;
- поиск категорий;
- version label;
- validation warnings.

#### Output schema

- выбор готовой schema;
- JSON preview/editor;
- проверка синтаксиса;
- поля verdict, evidence, attribution и confidence;
- preview ожидаемого результата.

#### Few-shot examples

- выбор набора examples;
- просмотр;
- добавление и удаление примеров;
- режим with tools / no tools.

#### Models

- PoolGenerator model и temperature;
- GraphGenerator model и temperature;
- judge node model и temperature;
- retry limits;
- предварительная оценка стоимости.

### Этап 3 — Judge Pool

Экран соответствует макету **Judge Pool Generated**.

#### Карточки судей

- имя;
- назначение;
- полные instructions;
- модель;
- заявленные tools;
- доступ к retrieval;
- предполагаемые категории taxonomy;
- метка auto-generated или edited.

#### Действия

- regenerate entire pool;
- regenerate one judge;
- edit instructions;
- duplicate judge;
- disable optional judge;
- добавить ручного судью;
- сравнить с предыдущим pool;
- сохранить pool как template.

#### Validation gate

Перед продолжением проверяются:

- наличие ровно одного `FINAL_AGGREGATOR`;
- уникальность имён;
- непустые инструкции;
- валидные models;
- доступность tools;
- соответствие taxonomy;
- output format агрегатора.

Пользователь явно подтверждает pool кнопкой **Use this pool**.

### Этап 4 — Graph

#### Режимы

- **Parallel recommended** — все judges → aggregator;
- **Generate DAG** — GraphGenerator;
- **Manual** — графический редактор;
- **Load saved graph**.

#### Graph canvas

- узлы судей;
- направленные связи;
- execution levels;
- terminal node;
- ожидаемая параллельность;
- estimated critical path;
- estimated cost by level.

#### Validation

- неизвестные узлы;
- self-links;
- циклы;
- disconnected nodes;
- несколько roots;
- несколько terminal nodes;
- недостижимый aggregator;
- обязательный порядок специальных судей.

После успешной проверки статус становится **Ready**.

### Этап 5 — Run

Рабочий экран использует три колонки.

```text
┌──────────────────────┬──────────────────────────┬─────────────────────────┐
│ Trace & Configuration│ Judge Pipeline           │ Live Result             │
│                      │                          │                         │
│ Search / filters     │ Level 0                  │ Current verdict         │
│ Step timeline        │ ├ Judge A · Complete     │ Evidence preview        │
│ Selected step        │ └ Judge B · Running      │ Aggregator status       │
│                      │ Level 1                  │                         │
│ Taxonomy             │ └ FINAL_AGGREGATOR       │ Time / calls / cost     │
│ Schema               │                          │                         │
│ Context mode         │ Node logs and outputs    │                         │
└──────────────────────┴──────────────────────────┴─────────────────────────┘
```

#### Trace column

- поиск по шагам;
- фильтр по агенту, role, tool и severity;
- виртуализированная timeline;
- раскрытие полного content;
- связи с evidence;
- состояние retrieval.

#### Judge Pipeline column

- уровень выполнения;
- queued, running, complete или failed;
- продолжительность узла;
- token usage и стоимость;
- компактный output;
- message history;
- retry отдельного узла;
- переход к родителям и потомкам.

#### Live Result column

- aggregator status;
- частичный verdict, если схема допускает;
- evidence cards;
- confidence;
- culprit agent и step;
- накопленные время, токены и стоимость.

#### Управление

- Run evaluation;
- Cancel;
- Retry failed node;
- Retry aggregator;
- Run again;
- сохранить snapshot конфигурации.

Backend сейчас прекращает pipeline при падении одного узла. UI для partial results потребует изменения backend execution policy.

### Этап 6 — Verdict

#### Summary

- тип verdict;
- pipeline status;
- confidence;
- human review status;
- число findings по severity;
- ответ `FINAL_AGGREGATOR`;
- total time, tokens, cost и LLM calls.

#### Evidence

- цитата;
- номер шага;
- агент;
- источник;
- evidence status;
- judge, который создал finding;
- переход к trace step.

#### Judge outputs

- результат каждого судьи;
- противоречия;
- вклад в итог;
- сырые messages;
- usage.

#### Graph and trace

- интерактивный DAG;
- Mermaid export;
- PipelineTrace;
- NodeTrace;
- Langfuse deep link.

#### Экспорт

- итог по output schema;
- полный run JSON;
- judge pool JSON;
- graph JSON/Mermaid/PNG;
- evidence CSV;
- trace JSON;
- ZIP всех артефактов.

## 7. Runs

Таблица всех одиночных запусков:

- run ID;
- trace ID;
- benchmark;
- created at;
- pool version;
- graph version;
- pipeline status;
- verdict;
- duration;
- tokens;
- cost;
- Langfuse status.

Функции:

- фильтры и поиск;
- сохранённые представления;
- повтор запуска;
- клонирование конфигурации;
- экспорт;
- архивирование;
- сравнение выбранных runs.

## 8. Compare

Сравнение двух или более runs:

- different models;
- full trace против summary + retrieval;
- разные taxonomies;
- разные judge pools;
- parallel graph против generated graph;
- исходные и оптимизированные prompts.

Показывать:

- verdict diff;
- findings added/removed;
- evidence overlap;
- judge pool diff;
- graph diff;
- latency, tokens и cost;
- benchmark correctness, если есть ground truth.

## 9. Judge Studio

### Taxonomies

- список и версии;
- tree editor;
- импорт/экспорт Markdown или JSON;
- clone и compare;
- ссылки на runs, где версия использовалась.

### Output schemas

- редактор;
- templates для benchmarks;
- syntax validation;
- preview результата;
- version history.

### Examples

- наборы few-shot examples;
- tags: tools/no-tools и benchmark;
- preview;
- валидация ожидаемой структуры.

### Judge pools

- generated pools;
- saved templates;
- количество судей;
- роли и модели;
- история regeneration/edit;
- test generation на выбранной trace.

### Graphs

- saved DAGs;
- graph editor;
- validation;
- compare;
- usage history.

## 10. Batch runs

### Создание batch

- dataset или каталог;
- glob/filter;
- benchmark adapter;
- taxonomy, schema и examples;
- pool strategy: per-trace, shared или per-benchmark;
- graph strategy;
- context mode;
- concurrency;
- retry policy;
- budget limit;
- output directory.

### Calibration gate

Перед массовым запуском:

1. Выполнить 1–5 calibration traces.
2. Проверить judge pools и outputs.
3. Показать стоимость одной трассы.
4. Оценить полный budget.
5. Потребовать подтверждение массового запуска.

### Active batch

- processed / total;
- running, succeeded, failed, skipped и pending;
- текущая стоимость;
- forecast стоимости;
- throughput;
- ETA только при достаточных данных;
- активные trace IDs;
- последние ошибки;
- Pause, Resume и Stop.

### Надёжность

- один result file на trace;
- атомарная запись;
- обнаружение уже готовых результатов;
- повтор только pending/failed;
- отдельный failure manifest;
- фиксирование configuration snapshot.

## 11. Data

### Traces

- загруженные трассы;
- формат и размер;
- число шагов;
- validation status;
- linked runs;
- preview и download.

### Summaries

- тип summarizer;
- модель;
- число исходных и итоговых токенов;
- summary по шагам;
- связь summary step → raw state.

### Retrieval store

- PostgreSQL connection status;
- таблицы;
- число states;
- coverage трассы;
- поиск по state ID;
- preview raw content;
- health check.

Операции создания и удаления БД/таблиц должны быть доступны только в административном режиме с отдельным подтверждением.

## 12. Optimizer Lab

Раздел помечается **Experimental** до восстановления отсутствующего Judge API.

### Prompt optimizer

- выбор run/pool;
- выбор судей;
- исходный prompt;
- optimized prompt;
- side-by-side diff;
- запуск контрольной оценки до и после;
- сравнение verdict, cost и latency;
- принять или отклонить изменения;
- сохранить новую pool version.

### Graph evolution

- initial graph;
- available agent pool;
- population size;
- generations;
- elitism;
- mutation/crossover rates;
- budget cap;
- judge/scoring configuration.

Live view:

- generation number;
- population;
- scores;
- best и average score;
- best graph;
- cumulative optimizer and pipeline cost;
- early stopping.

## 13. Observability

### Usage & cost

- стоимость по run, model, judge, benchmark и дате;
- input/output tokens;
- PoolGenerator, GraphGenerator и judge node cost отдельно;
- budget alerts;
- CSV export.

### Artifacts

- session logs;
- graph PNG/Mermaid;
- optimized prompts;
- result JSON;
- failure manifests;
- downloadable ZIP.

### Langfuse

- connection status;
- trace ID;
- link на внешний trace;
- instrumentation enabled/disabled;
- последнее flush time;
- понятное объяснение, какие функции требуют Langfuse.

## 14. Settings

### Models

- модели meta-agents;
- модели judge nodes;
- temperatures;
- max tokens;
- timeout и retries.

### Integrations

- OpenRouter;
- Langfuse;
- PostgreSQL;
- Mermaid rendering.

Секреты никогда не отображаются полностью и не сохраняются в run exports.

### Execution defaults

- graph strategy;
- context mode;
- concurrency;
- failure policy;
- default budget;
- artifact directory.

### Feature flags

- generated DAG;
- partial execution;
- prompt optimizer;
- graph evolution;
- DB administration.

## 15. Общие компоненты

| Компонент | Назначение |
|---|---|
| `StatusBadge` | Состояния run, node и integration |
| `TraceTimeline` | Виртуализированные шаги трассы |
| `JudgeCard` | Роль, инструкции, модель, tools и output |
| `GraphCanvas` | Просмотр и редактирование DAG |
| `EvidenceCard` | Цитата, источник и переход к шагу |
| `VerdictBanner` | Итог с severity и confidence |
| `UsageMeter` | Tokens, cost и budget |
| `RunProgress` | Levels и nodes в реальном времени |
| `SchemaEditor` | JSON schema и validation |
| `TaxonomyTree` | Иерархический редактор taxonomy |
| `DiffViewer` | Prompts, pools, graphs и verdicts |
| `IntegrationHealth` | OpenRouter, Langfuse и PostgreSQL |
| `ArtifactBrowser` | Логи и результаты |
| `ErrorRecovery` | Причина, сохранённые данные и следующее действие |

## 16. Связь UI с backend

| UI-функция | Backend-компонент | Состояние |
|---|---|---|
| Generate pool | `PoolGenerator.create_pool()` | Есть |
| Regenerate with feedback | `context` в `create_pool()` | Есть в examples |
| Generate graph | `GraphGenerator.create_graph()` | Есть |
| Parallel graph | `get_parallel_graph()` | Есть |
| Validate graph | `PipelineBuilder` | Есть |
| Execute pipeline | `Pipeline.ainvoke()` | Есть |
| Node progress events | Нет callback/event API | Требуется |
| Cancel run | Нет cooperative cancellation API | Требуется |
| Retry single node | Нет публичного API | Требуется |
| Partial completion | Pipeline падает на ошибке уровня | Требуется |
| Trace/messages | `Pipeline.trace` | Есть после run |
| Usage/cost | `Pipeline` и `AgentNode` properties | Есть, есть ошибки |
| Langfuse | `ainvoke_with_lf()` | Есть |
| Summary | Summarizer classes | Есть |
| Retrieval | PostgreSQL `get_content_tool` | Частично |
| Batch/resume | Benchmark scripts | Есть, не унифицировано |
| Prompt optimization | `optimize_pipeline_prompts()` | Есть |
| Graph evolution | `LLMEvoOptimizer` | Заблокировано missing Judge |
| Persistent run registry | Нет общей модели/хранилища | Требуется |

## 17. Модель данных для UI

Нужно ввести единые сущности:

- `TraceRecord`;
- `EvaluationConfig`;
- `TaxonomyVersion`;
- `OutputSchemaVersion`;
- `ExampleSetVersion`;
- `JudgePoolVersion`;
- `GraphVersion`;
- `RunRecord`;
- `NodeRunRecord`;
- `FindingRecord`;
- `EvidenceRecord`;
- `BatchRecord`;
- `ArtifactRecord`;
- `OptimizerExperiment`.

Каждый run должен хранить immutable snapshot всех входов, чтобы результат можно было воспроизвести.

## 18. Error states

Обязательные состояния:

- invalid trace;
- unsupported format;
- missing OpenRouter key;
- pool generation validation error;
- missing `FINAL_AGGREGATOR`;
- invalid DAG;
- model unavailable;
- rate limit;
- node timeout;
- node output schema mismatch;
- aggregator failure;
- Langfuse unavailable;
- PostgreSQL unavailable;
- missing retrieval state;
- budget exceeded;
- cancelled run;
- corrupted artifact;
- partial batch.

Ошибка должна показывать:

1. Что именно не выполнено.
2. Какие результаты уже сохранены.
3. Был ли списан API budget.
4. Одно основное действие восстановления.
5. Ссылку на технические детали и log.

## 19. Доступность и адаптивность

- desktop-first для сложного исследовательского workflow;
- при ширине 1024–1279 px configuration и result становятся tabs;
- на мобильном доступны просмотр runs, статусов и verdict, но не graph editor;
- управление с клавиатуры;
- видимый focus;
- status не кодируется только цветом;
- `aria-live` для прогресса;
- WCAG AA contrast;
- reduced motion;
- таблицы имеют доступный compact/card mode.

## 20. Этапы реализации

### Этап 0 — backend readiness

- исправить конструктор PoolGenerator;
- инициализировать `Pipeline._trace`;
- исправить cost model parsing;
- определить единый Judge API;
- исправить DB tool и tool registry;
- добавить run event callbacks;
- описать сериализуемые run records.

### Этап 1 — MVP одиночной оценки

- Streamlit shell;
- загрузка trace;
- taxonomy/schema editors;
- generate и review pool;
- parallel graph preview;
- запуск pipeline;
- итоговый verdict;
- node outputs, trace, usage и cost;
- JSON export.

### Этап 2 — полноценное исследование результата

- интерактивная timeline;
- evidence → trace linking;
- фильтры;
- judge output viewer;
- graph canvas;
- Langfuse link;
- run history и configuration snapshots.

### Этап 3 — длинный контекст

- summary preview;
- context mode selection;
- PostgreSQL health;
- retrieval state browser;
- coverage diagnostics.

### Этап 4 — batch workflow

- batch creation;
- calibration gate;
- per-trace atomic state;
- Pause/Resume;
- budget forecast;
- failure manifest;
- comparison and export.

### Этап 5 — Judge Studio

- versioned taxonomies;
- schemas;
- examples;
- saved pools;
- saved graphs;
- compare/diff.

### Этап 6 — Optimizer Lab

- prompt optimizer;
- before/after evaluation;
- graph evolution после восстановления Judge API;
- experiment history и budget limits.

## 21. Критерии MVP

- пользователь загружает валидную трассу без редактирования Python-кода;
- taxonomy и output schema явно видимы и проверяются;
- сгенерированный pool можно изучить до API-затрат judge pipeline;
- `FINAL_AGGREGATOR` проверяется автоматически;
- graph валидируется до запуска;
- виден статус каждого judge node;
- verdict содержит evidence и ссылки на шаги;
- pipeline success отделён от verdict;
- показываются tokens и cost;
- run можно экспортировать и воспроизвести из configuration snapshot;
- ошибки не удаляют уже полученные результаты;
- API keys отсутствуют в логах и exports.

## 22. Usability validation

Перед расширением MVP провести 5–8 модерируемых тестов с исследователями и MAS-разработчиками.

Проверить задачи:

1. Загрузить трассу и определить, почему запуск недоступен.
2. Найти и исправить отсутствующий aggregator в pool.
3. Запустить evaluation и определить текущий judge node.
4. Из verdict перейти к исходному evidence step.
5. Сравнить стоимость full trace и summary + retrieval.

Целевые показатели:

- completion rate не ниже 80%;
- пользователь находит evidence step без подсказки;
- pipeline status и verdict не путаются;
- критическая configuration error обнаруживается до запуска;
- пользователь может объяснить, какие LLM-вызовы формируют общую стоимость.

## 23. Рекомендуемый первый инкремент

Первый рабочий прототип должен реализовать один сквозной сценарий:

```text
Upload trace
→ configure taxonomy/schema
→ generate and approve judge pool
→ build parallel graph
→ run evaluation
→ inspect verdict/evidence/cost
→ export JSON
```

Для него достаточно Streamlit и существующего Python backend. Batch, persistent history, graph editing и optimizers следует добавлять после того, как одиночный запуск станет наблюдаемым и воспроизводимым.
