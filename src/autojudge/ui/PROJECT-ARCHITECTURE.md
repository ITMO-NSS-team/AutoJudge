# Архитектура AutoJudge

## 1. Кратко

AutoJudge — исследовательский Python-фреймворк для динамической генерации LLM-судей, оценивающих трассы мультиагентных систем. Вместо одного фиксированного judge prompt система создаёт специализированный пул судей под конкретную задачу, таксономию ошибок и выходную схему, связывает судей в направленный ациклический граф и передаёт их выводы финальному агрегатору.

Фактический основной поток:

```text
Trace + Task + Taxonomy + Output Schema + Examples
                         │
                         ▼
                   PoolGenerator
                создаёт AgentPool
                         │
                         ▼
      GraphGenerator или get_parallel_graph()
                   создаёт DAG
                         │
                         ▼
                  PipelineBuilder
             валидирует и собирает граф
                         │
                         ▼
                      Pipeline
      выполняет уровни DAG последовательно,
         а независимые узлы — параллельно
                         │
                         ▼
                 FINAL_AGGREGATOR
                         │
                         ▼
      Verdict + Node Traces + Usage + Cost + Graph
```

## 2. Основные входные данные

### Execution trace

Исходная история работы оцениваемой мультиагентной системы. Формат зависит от benchmark-адаптера. Трасса может содержать:

- запрос пользователя;
- шаги рассуждения;
- сообщения агентов;
- tool calls и результаты инструментов;
- состояние среды;
- итоговый ответ;
- ground truth или benchmark metadata.

### Task description

Описание того, что должна была выполнить оцениваемая система. Используется при генерации пула, построении графа и работе судей.

### Taxonomy

Иерархия допустимых категорий ошибок. PoolGenerator использует её для выбора специализаций судей.

### Output schema

Требуемый формат финального ответа `FINAL_AGGREGATOR`. Схемы для benchmarks находятся в `src/autojudge/meta_agents/prompts/output_schema_prompts/`.

### Examples

Few-shot примеры ролей судей и ожидаемого поведения. Есть варианты с доступом к БД и без него.

## 3. Генерация judge pool

`PoolGenerator` — meta-agent на базе PydanticAI. Он получает task description, taxonomy, output schema и examples, после чего возвращает структурированный список `AgentSchema`.

Каждая схема содержит:

- уникальное имя судьи;
- подробные инструкции;
- модель;
- список заявленных MCP tools;
- флаг использования tools.

Схемы преобразуются в `AgentNode` и помещаются в `AgentPool`.

Критическое соглашение: пул должен содержать узел с точным именем `FINAL_AGGREGATOR`. Benchmark-скрипты обычно проверяют его наличие и повторяют генерацию с feedback, если ответ meta-agent невалиден.

Основные файлы:

- `src/autojudge/meta_agents/pool_gen.py`;
- `src/autojudge/meta_agents/prompts/pool_prompts.py`;
- `src/autojudge/agent_pool.py`.

## 4. Построение judge graph

### Динамический граф

`GraphGenerator` передаёт список доступных судей отдельной LLM и получает adjacency list:

```json
{
  "EVIDENCE_JUDGE": ["FINAL_AGGREGATOR"],
  "TOOL_JUDGE": ["FINAL_AGGREGATOR"],
  "FINAL_AGGREGATOR": []
}
```

Ответ проверяется на:

- наличие только известных агентов;
- отсутствие self-links;
- корректный тип дочерних списков;
- отсутствие циклов.

### Параллельный граф

`get_parallel_graph()` соединяет каждого специализированного судью непосредственно с `FINAL_AGGREGATOR`.

```text
Judge A ─┐
Judge B ─┼──▶ FINAL_AGGREGATOR
Judge C ─┘
```

Этот вариант чаще используется в benchmark-скриптах: он снижает риск невалидного LLM-сгенерированного DAG и позволяет выполнять независимых судей параллельно.

Основные файлы:

- `src/autojudge/meta_agents/graph_gen.py`;
- `src/autojudge/pipeline/pipeline_builder.py`.

## 5. Сборка pipeline

`PipelineBuilder`:

1. Находит все узлы, упомянутые в графе.
2. Создаёт глубокие копии соответствующих `AgentNode`.
3. Связывает родителей и потомков.
4. Проверяет отсутствие циклов через DFS.
5. Выполняет топологическую сортировку.
6. Создаёт `Pipeline` с вычисленным порядком выполнения.

Глубокие копии защищают от конфликтов, когда из одного пула параллельно строятся несколько графов.

## 6. Выполнение узлов

`Pipeline` группирует узлы по dependency level:

- entry nodes имеют level 0;
- level узла равен максимальному level его родителей плюс один;
- все узлы одного level запускаются через `asyncio.gather()`;
- следующий level начинается после завершения предыдущего.

### Контекст entry node

Entry node получает исходный запрос без дополнительной упаковки.

### Контекст промежуточного узла

`NodeSession` формирует JSON со следующими данными:

- session ID;
- original query;
- текущий узел и его роль;
- результаты и инструкции родителей;
- сведения о дочерних узлах;
- признак terminal node;
- текст текущей задачи.

### Выполнение LLM

`AgentNode.build_agent()` создаёт PydanticAI Agent через OpenRouter. По умолчанию используется модель из `AGENT_NODE_MODEL` и температура из `AGENT_NODE_TEMPERATURE`.

После ответа сохраняются:

- output;
- полная message history;
- token usage;
- node metadata;
- время выполнения результата в NodeSession.

Если один из узлов уровня завершается ошибкой, pipeline формирует общую ошибку уровня и не продолжает следующие уровни.

Основные файлы:

- `src/autojudge/pipeline/node.py`;
- `src/autojudge/pipeline/node_session.py`;
- `src/autojudge/pipeline/pipeline.py`;
- `src/autojudge/pipeline/types.py`.

## 7. Финальная агрегация

`FINAL_AGGREGATOR` является единственным terminal node в рекомендуемой архитектуре. Он получает результаты специализированных судей и должен:

- разрешить противоречия;
- выбрать подтверждённые findings;
- сформировать итоговый verdict;
- соблюдать переданную output schema;
- вернуть пользовательский или машинно-читаемый результат.

Pipeline считает ответ последнего узла в топологическом порядке финальным результатом.

## 8. Trace, usage и стоимость

`PipelineTrace` содержит:

- session ID;
- исходный запрос;
- список `NodeTrace`;
- финальный output.

`NodeTrace` хранит модель, message history и usage отдельного судьи.

Стоимость рассчитывается через `genai-prices` по входным и выходным токенам каждого узла. `Pipeline.cost` суммирует стоимость всех судей.

Pipeline также может сформировать Mermaid-граф и запросить PNG у внешнего `mermaid.ink`.

## 9. Langfuse

Если заданы `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY` и `LANGFUSE_HOST`, включается глобальная PydanticAI instrumentation.

В Langfuse передаются:

- pool metadata;
- session ID;
- число узлов и execution levels;
- Mermaid и dict graph;
- token usage;
- стоимость;
- запрос и итоговый ответ.

Без Langfuse pipeline выполняется, но `ainvoke_with_lf()` возвращает `trace_id=None`. Некоторые экспериментальные ветки требуют trace ID и без Langfuse не работают.

Основной файл: `src/autojudge/utils/langfuse_utils.py`.

## 10. Полная трасса и summary + retrieval

### Full trace

Вся подготовленная трасса передаётся судьям в запросе. Режим проще, но ограничен контекстным окном и стоимостью.

### Summary + retrieval

Для длинных трасс проект содержит три варианта суммаризации:

- `TraceSummarizer` — общая структурированная сводка;
- `StepByStepSummarizer` — сводка по каждому шагу;
- `StepsBatchSummarizer` — пакетная суммаризация с предыдущим контекстом.

Полные состояния можно сохранить в PostgreSQL. Судьи получают сокращённый контекст и при необходимости вызывают `get_content_tool(state_id, table_name)`.

Основные каталоги:

- `src/autojudge/meta_agents/*summarizer.py`;
- `src/autojudge/db/`;
- `src/autojudge/db/create_db/`.

## 11. Benchmark workflow

Примеры в `examples/` являются основной исследовательской обвязкой. Они:

1. Загружают benchmark dataset.
2. Преобразуют трассу в judge input.
3. Создают PoolGenerator с конкретными taxonomy и schema.
4. Генерируют pool с retry и feedback.
5. Строят динамический или параллельный граф.
6. Запускают pipeline.
7. Разбирают JSON агрегатора.
8. Сохраняют один результат на трассу.
9. Пропускают уже обработанные элементы.
10. Отдельно фиксируют неудачные трассы.

Поддерживаются Who&When, TRAIL, AEGIS, AgentErrorBench, AgentRewardBench/WebArena и Pumpkin.

## 12. Оптимизаторы

### Prompt optimization

`HyPEPromptOptimizer` параллельно переписывает инструкции всех узлов, сохраняя язык, output constraints и роль terminal/intermediate agent. Результаты сохраняются в JSON-артефакт.

### Evolutionary graph optimization

`LLMEvoOptimizer` экспериментирует с DAG:

- создаёт начальную population;
- мутирует графы;
- скрещивает успешные варианты;
- выполняет каждый кандидат;
- оценивает результат внешним Judge;
- применяет tournament selection и elitism;
- останавливается при идеальном score.

Эта подсистема присутствует в исходниках, но сейчас зависит от отсутствующего `autojudge.judge` и не импортируется без доработки.

## 13. Публичные интерфейсы

### Высокоуровневый wrapper

Класс `autojudge` предоставляет `run()` и `arun()`:

```python
from autojudge import autojudge

result = autojudge().run("Evaluate this trace")
```

Wrapper создаёт PoolGenerator и GraphGenerator, запускает pipeline и возвращает answer, pool и Mermaid graph.

Фактическое ограничение: wrapper не принимает taxonomy и output schema, поэтому полноценные benchmark-запуски используют низкоуровневые классы напрямую.

### Низкоуровневый API

```python
pool_gen = PoolGenerator(
    taxonomy=taxonomy,
    output_schema=output_schema,
    examples=examples,
)
pool = await pool_gen.create_pool(judge_input)
graph = get_parallel_graph(pool)
pipeline = PipelineBuilder().create_from_pool(pool, graph).build()
result = await pipeline.ainvoke(judge_input)
```

Это основной рабочий путь в текущих experiments.

## 14. Конфигурация

Основные переменные окружения:

| Переменная | Назначение |
|---|---|
| `OPENROUTER_API_KEY` | Обязательный ключ для meta-agents и judge nodes |
| `DEFAULT_META_MODEL` | Базовая модель meta-agents |
| `POOL_GEN_MODEL` | Модель PoolGenerator |
| `POOL_GEN_TEMPERATURE` | Температура PoolGenerator |
| `GRAPH_GEN_MODEL` | Модель GraphGenerator |
| `AGENT_NODE_MODEL` | Модель judge nodes |
| `AGENT_NODE_TEMPERATURE` | Температура judge nodes |
| `LANGFUSE_*` | Трассировка и внешний trace ID |
| `DB_*` | Подключение к PostgreSQL |

Шаблон находится в `.env.template`.

## 15. Логи и артефакты

Loguru создаёт каталог:

```text
autojudge_logs/session_YYYY-MM-DD_HH-MM-SS/
```

Туда могут сохраняться:

- `autojudge.log`;
- Mermaid PNG;
- результаты prompt optimization;
- другие JSON и текстовые артефакты.

Benchmark-результаты обычно записываются непосредственно в подкаталоги `examples/`.

## 16. Фактическое состояние

Проверено локально:

- 57 Python-файлов в `src/autojudge`;
- около 4270 строк исходного Python-кода;
- `python -m compileall -q src/autojudge` проходит;
- из выбранных офлайн cost-тестов проходят 5 из 7;
- из трёх выбранных офлайн trace-тестов проходит 1;
- интеграционные тесты с LLM не запускались, чтобы не расходовать API.

Текущие технические ограничения:

- `autojudge()` падает без явно заданного `POOL_GEN_TEMPERATURE` из-за прямого чтения `os.environ`;
- `arun_with_judge` и evolutionary optimizer зависят от отсутствующего `autojudge.judge`;
- расчёт стоимости расходится с тестами для model references;
- `Pipeline.trace` до запуска вызывает `AttributeError` вместо `None`;
- `NodeTrace.model` обязателен, но один тест ожидает значение по умолчанию;
- DB retrieval tool жёстко привязан к `agent_error`;
- заявленные `mcp_tools` не подключаются динамически: AgentNode фактически знает только `get_content_tool`;
- README описывает каталоги, которых нет в фактическом дереве;
- полноценной UI-реализации пока нет: каталог `src/autojudge/ui` содержит проектные материалы.

## 17. Вывод

Рабочее ядро проекта состоит из четырёх последовательных слоёв:

1. `PoolGenerator` создаёт роли судей.
2. `GraphGenerator` или `get_parallel_graph()` задаёт их связи.
3. `PipelineBuilder` проверяет и собирает DAG.
4. `Pipeline` выполняет судей, собирает trace и отдаёт ответ агрегатора.

Вокруг ядра находятся benchmark adapters, суммаризация длинных трасс, PostgreSQL retrieval, Langfuse, стоимость и экспериментальные оптимизаторы. Проект ближе к воспроизводимому research prototype, чем к готовому пользовательскому продукту; новый UI должен сделать эти разрозненные возможности единым управляемым workflow.
