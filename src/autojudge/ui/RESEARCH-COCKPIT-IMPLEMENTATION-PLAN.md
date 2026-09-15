# План реализации Research Cockpit для AutoJudge

## 1. Решение

Research Cockpit принят как целевое визуальное направление AutoJudge.

Референс: [research-cockpit-concept.png](research-cockpit-concept.png).

Цель — сделать интерфейс ярче, быстрее для чтения и удобнее для анализа judge pipeline, сохранив исследовательскую точность и существующий сквозной workflow:

```text
Trace → Design → Judge pool → Graph → Run → Verdict
```

Макет является визуальным ориентиром, а не точной спецификацией данных. UI не должен показывать вымышленные findings, confidence, cost, health или статусы.

## 2. Принципы

1. Цвет обозначает состояние или тип информации, а не используется как декор.
2. Текущая модель, endpoint и режим выполнения видны до платного запуска.
3. Pipeline status и содержательный verdict визуально и терминологически разделены.
4. Judge instructions доступны полностью, но не занимают карточку по умолчанию.
5. Evidence всегда ведёт к фактическому trace step, если такая связь присутствует в результате.
6. Demo, offline и неподключённые функции явно маркируются.
7. Старые run snapshots сохраняют фактическую модель и конфигурацию запуска.

## 3. Визуальная система

### 3.1. Цветовые токены

| Назначение | Цвет | Использование |
|---|---|---|
| Primary | Indigo `#4F46E5` | активный шаг, primary action, выбранный узел |
| Secondary | Violet `#7C3AED` | aggregator, AI-related metadata |
| Information | Blue `#2563EB` | evidence и trace references |
| Success | Emerald `#059669` | completed, healthy, valid |
| Warning | Amber `#D97706` | needs review, partial, stale |
| Error | Coral `#E5484D` | failed, invalid, destructive warning |
| Text | Slate `#172033` | основной текст |
| Muted | Slate `#667085` | вторичные подписи |
| Canvas | Cool gray `#F6F8FC` | фон рабочей области |
| Surface | White `#FFFFFF` | карточки и панели |

Для фоновых статусов используются светлые оттенки тех же семантических цветов. Контраст текста и интерактивных состояний должен соответствовать WCAG AA.

### 3.2. Форма и глубина

- базовая сетка 8 px;
- радиус карточек 12 px, небольших controls — 8 px;
- тонкая нейтральная граница вместо тяжёлых теней;
- тень только у плавающих панелей, выбранного узла и модальных окон;
- минимальная интерактивная область 44 × 44 px;
- transition 150–200 ms, без обязательной непрерывной анимации.

### 3.3. Типографика

- один современный sans-serif стек без обязательной загрузки внешнего шрифта;
- крупный заголовок страницы 30–32 px;
- заголовок панели 20–22 px;
- основной текст 14–16 px;
- metadata 12–13 px;
- технические значения и JSON остаются в monospace.

## 4. Глобальная оболочка

### Левая панель

- сохранить существующие разделы и маршруты;
- объединить родственные пункты визуальными группами, не меняя URL/hash navigation;
- добавить блок текущей модели и статуса backend;
- quick actions оставить только для реально работающих действий;
- на узком экране сворачивать панель до иконок или drawer.

### Верхняя панель

- breadcrumb текущего workspace и страницы;
- поиск добавлять только после появления рабочего поиска;
- справа показывать текущую judge model и основное действие текущего экрана;
- connection status брать из backend, а не из статического списка сервисов.

## 5. Компоненты

### 5.1. SemanticBadge

Единый компонент для `Draft`, `Ready`, `Running`, `Completed`, `Partial`, `Failed`, `Cancelled`, `Stale`, `Offline`, `Demo` и `Not connected`.

Acceptance criteria:

- цвет не является единственным носителем смысла;
- badge содержит текст и при необходимости иконку;
- одинаковый статус одинаково выглядит на всех экранах.

### 5.2. JudgeCard

Свернутое состояние:

- имя и иконка специализации;
- краткое назначение, полученное из первой содержательной части instructions или отдельного summary;
- модель;
- статус узла;
- фактическое число findings, если оно доступно;
- действия `View output` и `Instructions`.

Раскрытое состояние или drawer:

- полные instructions;
- raw output;
- usage и duration, если backend их вернул;
- evidence references;
- редактирование разрешено только в режиме конфигурации до запуска.

`FINAL_AGGREGATOR` отображается отдельным крупным узлом после специалистов и не дублируется обычной карточкой в том же представлении.

### 5.3. PipelineGraph

Первый инкремент реализуется на собственном SVG без новой крупной зависимости:

- раскладка узлов по execution levels;
- кривые связи со стрелками;
- ручное перемещение узлов pointer drag и клавишами стрелок с сохранением layout локально;
- быстрые шаблоны `Parallel → final` и `Sequential chain → final`;
- отдельные стили queued, running, completed, failed и selected;
- подсветка родителей и потомков выбранного узла;
- адаптивное размещение без горизонтального overflow;
- reduced-motion режим.

React Flow рассматривается только при необходимости полноценного visual edge editing; перемещение узлов реализуется без новой зависимости.

### 5.4. VerdictPanel

- pipeline status располагается отдельно от verdict;
- крупный итог берётся из output schema и ответа aggregator;
- attribution показывает agent и step только при наличии этих полей;
- confidence показывается только при наличии фактического значения;
- evidence cards открывают соответствующий trace step;
- при неизвестной структуре доступен безопасный JSON viewer.

### 5.5. WizardProgress

- шесть существующих шагов сохраняются;
- completed, current, available и blocked имеют разные состояния;
- blocked step объясняет причину;
- изменение Design после генерации помечает pool и graph как `Stale`.

## 6. Экраны

### Overview

- текущая judge model;
- backend health;
- число runs и активных запусков;
- последние run snapshots с фактическими моделями;
- блок `Requires attention` только на основании Failed, Partial, Stale или review state;
- cost и usage не показываются как ноль, если данные неизвестны.

### New evaluation

- мастер получает новую визуальную оболочку без изменения существующего API-контракта;
- Run показывает модель и endpoint, а AI execution выбран по умолчанию;
- Judge pool использует JudgeCard и отдельный aggregator;
- Graph использует PipelineGraph;
- Run отображает live status на тех же узлах;
- Verdict использует VerdictPanel и evidence navigation.

### Runs и Compare

- таблица сохраняет плотный исследовательский формат;
- статусы переводятся на SemanticBadge;
- модель является snapshot конкретного запуска;
- Compare получает визуальное выделение различий, но не генерирует отсутствующие метрики.

### Settings

- группы остаются функциональными, без декоративных service cards;
- секреты показывают фактический OS storage backend;
- неиспользуемые настройки не возвращаются в UI;
- опасные и платные настройки получают amber callout, но не красный error style.

## 7. Архитектура frontend

Планируемое разбиение текущего `Workspace.tsx`:

```text
src/
├── components/
│   ├── SemanticBadge.tsx
│   ├── JudgeCard.tsx
│   ├── PipelineGraph.tsx
│   ├── VerdictPanel.tsx
│   └── WizardProgress.tsx
├── pages/
│   ├── OverviewPage.tsx
│   ├── EvaluationWizard.tsx
│   ├── RunsPage.tsx
│   └── SettingsPage.tsx
├── styles/
│   ├── tokens.css
│   └── cockpit.css
└── Workspace.tsx
```

Разбиение выполняется постепенно. Нельзя одновременно переписывать всю state-модель и визуальную систему: сначала выделяются чистые presentational components, затем страницы.

## 8. Этапы реализации

### Этап 1 — Foundation

- добавить CSS tokens;
- обновить фон, типографику, sidebar, header, buttons и form controls;
- создать SemanticBadge;
- адаптировать Overview;
- сохранить существующую логику и API.

Готово, когда Overview и глобальная оболочка соответствуют выбранному направлению на desktop и mobile, а все текущие действия продолжают работать.

### Этап 2 — Judge experience

- создать JudgeCard;
- свернуть длинные instructions в drawer/details;
- отделить specialist judges от FINAL_AGGREGATOR;
- добавить честные статусы и модель;
- применить компонент в New evaluation и Judge Studio.

Готово, когда pool из 2–8 узлов читается без длинного вертикального полотна и полностью редактируется до запуска.

### Этап 3 — Pipeline graph

- заменить текстовый DAG на SVG PipelineGraph;
- реализовать уровни, стрелки и semantic states;
- поддержать ручное размещение узлов и последовательные judge chains;
- связать выбор узла с output/details;
- проверить большие имена и 2–8 узлов.

Готово, когда граф корректно отображает фактические edges и `FINAL_AGGREGATOR`, не создаёт циклы визуально и не выходит за viewport.

### Этап 4 — Run и Verdict

- применить состояния выполнения к JudgeCard и PipelineGraph;
- создать VerdictPanel;
- добавить evidence-to-trace navigation;
- различать неизвестные, частичные и завершённые результаты.

Готово, когда пользователь может определить статус pipeline, итог, виновного агента и evidence step без просмотра raw JSON, если backend предоставил эти данные.

### Этап 5 — Polish и accessibility

- responsive states для 390, 768, 1280 и 1536 px;
- keyboard navigation и focus visibility;
- WCAG AA contrast review;
- reduced motion;
- empty, loading, error и long-content states;
- визуальная регрессия ключевых экранов.

## 9. Проверки

Минимальный gate для каждого этапа:

```powershell
cd src/autojudge/ui/web
npm run build
```

Дополнительно:

- backend unit/integration tests для неизменности API-контрактов;
- browser smoke test: Overview → New evaluation → Judge pool → Graph → Run → Verdict;
- проверка 390 px без горизонтального overflow;
- keyboard-only проход мастера;
- проверка long judge names, long instructions и malformed output;
- проверка offline, AI, Failed, Cancelled и Interrupted;
- отсутствие секретов в DOM, exports, localStorage и логах.

## 10. Не входит в визуальный редизайн

Следующие возможности нельзя считать реализованными только за счёт нового UI:

- настоящий cost calculation, если backend возвращает `null`;
- Langfuse и PostgreSQL health без реальной проверки;
- partial pipeline execution;
- retry отдельного judge node;
- автоматическая генерация произвольного DAG;
- drag-and-drop graph editing;
- batch processing и optimizer execution;
- confidence и findings, если их нет в output schema.

Для них нужны отдельные backend-инкременты.

## 11. Порядок выполнения

Рекомендуемый порядок: Этап 1 → Этап 2 → browser review gate → Этап 3 → browser review gate → Этап 4 → Этап 5.

Первый review gate проводится после оболочки и карточек судей: он позволит скорректировать визуальный язык до более дорогой реализации интерактивного графа.
