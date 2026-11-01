# Воспроизведение TrustFlow

Все команды выполняются из корня репозитория. Проверено на Windows / Python 3.12. Для применения сохранённых моделей исходный дамп не нужен.

## Окружение и быстрая проверка

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts/context_payment_layer.py --input models/context_layer/example_input.json
.\.venv\Scripts\python.exe scripts/verify_repository.py
```

Другой исследовательский режим:

```powershell
.\.venv\Scripts\python.exe scripts/context_payment_layer.py --mode high_recall --input models/context_layer/example_input.json
```

Вход: 27 признаков по [схеме](../models/context_layer/feature_manifest.json), сумма и `mandatory_hold`. Пример уже содержит готовые признаки одной синтетической операции. Для собственных событий сначала нужна корректная история, а не произвольное заполнение признаков. Выход — рекомендация PASS/INTERVENE. `mandatory_hold=true` всегда возвращает вмешательство.

## Подготовка локальных данных

Нужен переданный исходный архив `vtb_antifraud_complete.zip`; положите его в `.local/`. Не добавляйте архив или отзывы в Git. Для чтения custom-format дампа требуется PostgreSQL `pg_restore`, сервер БД не нужен.

```powershell
.\.venv\Scripts\python.exe scripts/prepare_local_data.py --dataset all
```

По умолчанию используется PostgreSQL 17 в `C:/Program Files/PostgreSQL/17/bin/pg_restore.exe`; альтернативный путь передаётся через `--pg-restore`. Скрипт проверяет SHA-256 дампа, извлекает COPY для FDH и PaySim в `tmp/`, SQL не исполняет. Можно передать только `--dataset fdh` или `--dataset paysim`.

## FDH: обучение и проверка

```powershell
.\.venv\Scripts\python.exe scripts/fdh_context_experiment.py
.\.venv\Scripts\python.exe scripts/fdh_low_friction.py
.\.venv\Scripts\python.exe scripts/fdh_fairness_audit.py
.\.venv\Scripts\python.exe scripts/verify_fdh_context.py
.\.venv\Scripts\python.exe scripts/report_context_layer_final.py
```

Обучение перезаписывает результаты в `models/context_layer/`; перед ним сохраните текущую версию в Git, если нужно сравнение. Протоколы уже входят в репозиторий. Большие промежуточные прогнозы `*.npz` и кеши `tmp/` исключены из Git и создаются заново. Веса и небольшие JSON/CSV сохраняются. Итоговый отчёт — [experiments.md](experiments.md), график — [model_comparison.png](../assets/model_comparison.png).

Основной high-recall протокол был зафиксирован до первого теста дней 157–182. Low-friction и дополнительные сильные сравнения — последующие проверки на том же тесте. Повторный запуск не превращает уже исследованный период в новый независимый тест.

## PaySim: контрольные эксперименты

```powershell
.\.venv\Scripts\python.exe scripts/decision_layer_v2.py
.\.venv\Scripts\python.exe scripts/verify_decision_layer_v2.py
.\.venv\Scripts\python.exe scripts/decision_layer_v3.py
```

V2 сохраняет модели и результаты в `models/paysim_v2/`; V3 — в `research/paysim/v3/`. V3 использует замороженный history-baseline V2. В первоначальном аудите были выявлены проблемы позиционного соединения, тестовых порогов и учёта DELAY; его [агрегированные результаты](../research/paysim/audit/) сохранены. Прежний `hello_world.py` и зависящие от него устаревшие скрипты удалены. Его аудит не выдаётся за повторяемый без исходного файла; актуальные V2/V3 воспроизводятся отдельно.

## ЦБ, отзывы и локальный NLP

```powershell
.\.venv\Scripts\python.exe scripts/analyze_dump.py
.\.venv\Scripts\python.exe scripts/verify_cbr.py
.\.venv\Scripts\python.exe scripts/curate_sbp.py
.\.venv\Scripts\python.exe scripts/make_figures.py
```

Агрегаты записываются в `research/analysis/`, изображения — в `assets/`. Тексты и разметка с фрагментами — только в `.local/analysis/`. `curate_sbp.py` проверяет доказательные фрагменты относительно исходных текстов; это разметка ИИ, не независимые человеческие метки.

Для локального RuBERT:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-nlp.txt
.\.venv\Scripts\python.exe scripts/fetch_sources.py --model
.\.venv\Scripts\python.exe scripts/review_semantics.py
```

`fetch_sources.py` скачивает публичные страницы ЦБ и веса модели; локальные данные не отправляет. Сохранённые HTML уже входят в репозиторий, поэтому повторное скачивание для проверки ЦБ не требуется. RuBERT закреплён на revision `e8ed3b0c8bbf4fb6984c3de043bf7d2f4e5969ae`, проверяет хеши и выполняет inference offline из `.local/models/rubert-tiny2/`. Его большие веса не входят в Git. Ранжирование, embeddings и шаблон человеческой разметки остаются локально; публичный итог запуска — [bert_run.json](../research/analysis/bert_run.json).

## Экономика

Node.js, без дополнительных пакетов:

```powershell
node research/finance/calculate.cjs
```

Команда проверяет согласованность формул и обновляет [scenarios.json](../research/finance/scenarios.json). Сценарные параметры редактируются в [financial-model.js](../research/finance/financial-model.js), трактовка и ограничения — в [finance.md](finance.md).

## Что удалено при подготовке репозитория

Временные SQL-выгрузки, матрицы признаков, журналы прогнозов, численный precheck с исправленной ошибкой, старые PDF, предыдущие дубли отчётов, сборочные файлы Figma, кеши Python, лог установки и старые архивы результатов. Численные итоги, протоколы и оговорки неудачных сравнений сохранены. Исходный архив и локальные NLP-данные сохранены в `.local/`; виртуальное окружение исключено из Git. PDF презентации не изменялся и не копировался в репозиторий.
