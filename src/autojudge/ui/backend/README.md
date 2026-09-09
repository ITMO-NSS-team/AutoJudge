# Local backend

## Настраиваемое подключение AI

Settings → AI connection: LLM_BASE_URL — полный базовый URL OpenAI-совместимого
Chat Completions API (включая /v1), LLM_API_KEY — ключ этого сервиса.
HTTPS обязателен, кроме HTTP на localhost/127.0.0.1/::1. Локальный сервер может работать без ключа.
AGENT_NODE_MODEL задаёт model ID сервиса; явная модель мастера имеет приоритет.
LLM_API_KEY имеет приоритет над старым ключом. OPENROUTER_API_KEY используется только
при стандартном https://openrouter.ai/api/v1 и отсутствии LLM_API_KEY.
При смене сервиса замените или отключите LLM_API_KEY. Endpoint фиксируется в snapshot запуска.
Сохранение настроек не вызывает AI. Нативные несовместимые API не поддерживаются.

Запуск из корня AutoJudge:

```powershell
python -m pip install -r src/autojudge/ui/backend/requirements.txt
python -m uvicorn server:app --app-dir src/autojudge/ui/backend --host 127.0.0.1 --port 8000
```

Во втором терминале, из `src/autojudge/ui/web`: `npm run dev`.
Vite перенаправляет `/api` в локальный backend. Swagger: http://127.0.0.1:8000/docs.

## Контракт

- GET/PUT `/api/settings/env`: 23 разрешённых параметра из `.env.template` и кода.
  PUT принимает `values` (изменённые значения) и `reset` (имена для удаления override).
  Приоритет: сохранённый override → process env → корневой `.env` → default.
  Пустой secret override отключает секрет, reset возвращает env/default.
  `.env` не перезаписывается; секреты из него возвращаются только как configured boolean.
  Temperatures: 0–2; DB_PORT: 1–65535; Langfuse URL без credentials/query.
  AI runner применяет OPENROUTER_API_KEY, AGENT_NODE_TEMPERATURE и AGENT_NODE_MODEL.
  Явная модель из мастера имеет приоритет; openrouter/auto использует AGENT_NODE_MODEL.

- GET/PUT/DELETE `/api/credentials/openrouter`: статус, сохранение/замена и удаление ключа.
  PUT принимает JSON `{ "key": "..." }`; секрет никогда не возвращается.
  Изменения допускаются только с Origin локального UI (:5173) или API (:8000).
  Ключ хранится отдельно от workspace в SQLite, зашифрованный Windows DPAPI
  для пользователя, под которым работает backend. Без Windows сохранение возвращает 503.
  Наличие ключа не означает проверку у провайдера и не разрешает платные вызовы.

- GET `/api/health`: статус, ai_available и необходимость явного подтверждения.
- GET/POST `/api/runs`: список и создание offline или ai run.
- GET `/api/runs/{id}`: snapshot и outputs.
- GET `/api/runs/{id}/events`: SSE, при переподключении отдаёт актуальный snapshot.
- POST `/api/runs/{id}/cancel`: отмена с сохранением результатов.
- PATCH `/api/runs/{id}`: archived boolean.
- POST `/api/graph/validate`: валидация графа и уровни DAG.
- GET/PUT `/api/workspace`: конфигурация, версии, трассы и настройки.

SQLite хранится в `.autojudge/workspace.sqlite3` в корне проекта и исключён из Git.
Использовать один uvicorn worker. API предназначен для localhost, без авторизации.

## Реальный AI pipeline

По умолчанию execution=offline: без запросов к ИИ, calls/tokens/cost=0.
Для настоящей оценки POST /api/runs требует execution=ai, confirm_paid=true и
Origin локального приложения. Ключ извлекается только на сервере.
ai_runner.py строит AgentNode и PipelineBuilder из утверждённых UI pool/edges
и выполняет настоящий Pipeline.ainvoke с наблюдением за каждым узлом.
Модель получает цель, taxonomy, examples и trace; aggregator возвращает JSON,
проверяемый по output schema. Невалидный JSON/schema означает Failed с сохранением outputs.

Первый режим: Full trace, максимум 8 узлов и 100 KB JSON трассы, без tools и Langfuse.
Один запрос на узел, без SDK/agent retries, до 1024 выходных токенов, timeout 180 секунд.
Только один активный AI run на локальный сервер. Автогенерация pool/graph не выполняется.
Budget target НЕ является жёстким денежным лимитом. Ограничивайте расходы лимитом ключа
на стороне OpenRouter. Стоимость возвращается null, а не фиктивный ноль.
После ошибки/отмены usage может быть неполным; уже отправленные запросы могут тарифицироваться.
Метаданные health.paid_calls_enabled=false означают отсутствие автоматического разрешения;
ai_available=true и ai_requires_explicit_confirmation=true описывают ручной запуск.

В UI: Settings → ключ/модель → New evaluation → Full trace → Design → Pool → Graph →
Run → AI → подтверждение → Run AI evaluation. Платный прогон при разработке не выполнялся.
Batch и Optimizer Lab пока остаются клиентскими демонстрационными сценариями.

## Проверки

Из этого каталога: `python -m unittest test_server test_ai_runner -v`.
Проверяются credentials, env, lifecycle, подтверждение AI, реальный Pipeline с FunctionModel,
OpenRouter SDK с MockTransport, отмена, скрытие ошибок. Сеть блокируется в модельных тестах;
используются вымышленные ключи и временная БД.
