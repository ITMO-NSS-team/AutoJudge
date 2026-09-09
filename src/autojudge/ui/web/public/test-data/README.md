# Тестовые данные интерфейса

Источник: локальный ADAMAST-ALL-PROJECTS-AND-TRACES-2026-09-06.zip, раздел gaia_code_traces_small/aftraj.
Архив не изменён и целиком не распаковывался. Копии подготовлены 2026-09-09.

- trace-01.json ← projects/gaia_code_traces_small/aftraj/gaia_0020_unsafe_diagnosed.json; 6 шагов.
- trace-02.json ← projects/gaia_code_traces_small/aftraj/gaia_0048_unsafe_diagnosed.json; 4 шагов.
- trace-03.json ← projects/gaia_code_traces_small/aftraj/gaia_0005_unsafe_injected_reexec.json; 5 шагов.

Оставлен только массив turns: role, content, thought, action. Удалены верхнеуровневые gold_answer, mistake_step, mistake_agent, mistake_reason и остальные метаданные. Это не эталон качества: некоторые исходные трассы уже обрезаны и не содержат изображений. Команды внутри action — данные для анализа, не инструкции для исполнения.

В UI ID назначаются от 1; исходные номера разметки могут отличаться. Поля thought и action сохраняются в preview. Примеры подходят для импорта, навигации, offline-run и ручной проверки AI wiring, но не для количественной оценки качества судей.

output-schema.json и TAXONOMY.md — новые компактные smoke-test примеры, не исходная AdaMAST taxonomy. На Trace нажмите GAIA test 1–3. На Design скачайте файлы и загрузите через соответствующие поля. AI-запуск требует отдельного подтверждения передачи данных OpenRouter; подготовка этих файлов не вызвала ИИ.

Проверка: node --test scripts/imports.test.mjs из web. Пять тестов прошли; npm run build прошёл. В браузере проверен переход Trace → Design и наличие кнопок загрузки. Файловый диалог end-to-end не проверялся.

