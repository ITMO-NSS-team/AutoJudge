# AutoJudge UI prototype

Обновление AI integration: на шаге Run доступен настоящий AutoJudge Pipeline через
OpenRouter с явным подтверждением. Поддерживается Full trace, утверждённые pool/edges,
SSE, node outputs, final JSON, tokens и экспорт. Ключ берётся на сервере из Settings/env.
Автогенерация pool, tools и retrieval ещё не подключены. Budget target не является
денежным ограничителем; лимит расходов задавайте для ключа OpenRouter.
Точный контракт и запуск описаны в `../backend/README.md`.

React-интерфейс AutoJudge с десятью разделами по FULL-INTERFACE-PLAN.md.
Точка входа — `src/Workspace.tsx`; ранний трёхколоночный прототип сохранён в `src/App.tsx`.

## Реализованные сценарии

- Overview: локальные счётчики и последние запуски.
- New evaluation: шестишаговый мастер, импорт JSON/JSONL, нормализация и проверка ID,
  редакторы taxonomy/schema/examples, редактирование pool и рёбер DAG, проверка циклов
  и достижимости terminal aggregator, отменяемый демонстрационный запуск.
- Runs / Compare: поиск, фильтрация, архивирование и восстановление, клонирование,
  сравнение конфигураций, просмотр node outputs и переход к evidence step, JSON export.
- Judge Studio: редакторы, локальные версии конфигураций, загрузка копии и экспорт.
- Batches: демонстрационный calibration gate, progress, pause/resume/stop и экспорт состояния.
- Data: библиотека сохранённых трасс, повторное использование и скачивание.
- Optimizer Lab: ручной редактор кандидата prompt, accept/reject и экспорт.
- Observability / Settings: артефакты запусков, состояние интеграций, execution defaults.

Состояние хранится в localStorage данного браузера. Это не серверная база данных.
Перезагрузка во время одиночной симуляции не возобновляет её.

## Запуск

```powershell
npm install
npm run dev
```

Локальный адрес по умолчанию: `http://127.0.0.1:5173/`.

## Проверка production-сборки

```powershell
npm run build
```

## Границы прототипа

- trace можно загрузить собственный; pool редактируется вручную;
- offline проверяет структуру, AI вызывает настоящую модель после подтверждения;
- Python API и SQLite подключены;
- разделы доступны через навигацию и URL hash;
- CSV, полноценный графический canvas, автоматическое summary/retrieval, реальная
  калибровка batch, оптимизация и Langfuse требуют дальнейшей реализации;
- schema проверяется на JSON-синтаксис и корневой type object, не полным JSON Schema validator;
- defaults сохраняются, но ещё не применяются к реальному runner;
- симуляции не формируют реальные confidence, usage, cost и quality metrics.

Следующий технический этап — добавить FastAPI-контракт для создания run и поток событий
выполнения, после чего заменить демонстрационное состояние реальными данными AutoJudge.
