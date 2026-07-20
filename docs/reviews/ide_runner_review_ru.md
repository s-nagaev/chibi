# Chibi IDE Runner — Независимый обзор

## Краткое резюме

Настоящий документ объединяет три независимых обзора IDE runner для Chibi — транспорта JSONL поверх stdio, позволяющего внешней IDE взаимодействовать с ассистентом Chibi. Обзоры были подготовлены тремя различными моделями, исследовавшими одну и ту же кодовую базу с разных сторон: (1) **обзор транспорта/протокола**, сфокусированный на корректности wire-протокола, жизненном цикле запросов и адаптере `IDEInterface`; (2) **обзор интеграционного слоя**, сфокусированный на том, как IDE runner компонуется с существующими сервисами Chibi (бот, обработчики, хранилище, обработка команд); и (3) **обзор инфраструктуры/тестов**, сфокусированный на подсистеме task-manager, примитивах конкурентности и покрытии тестами.

Методология: каждый рецензент прочитал соответствующие файлы исходного кода, перекрёстно проверил более широкую кодовую базу (`constants.py`, `services/interface.py`, `services/task_manager.py`, `bot.py`, `cli.py`), и подготовил структурированный отчёт. Три отчёта были **дедуплицированы, объединены при наложении и переорганизованы по степени серьёзности** в единый авторитетный каталог. Новая аналитика не добавлялась.

Общая оценка: runner **функционален и архитектурно обоснован на высоком уровне** — протокол JSONL хорошо определён, жизненный цикл запросов имеет чёткие границы, а адаптер `IDEInterface` корректно переопределяет контракт `UserInterface`. Однако обзоры коллективно выявили **две критические ошибки корректности** (`IDEInterface.send_images` выводит Python repr объектов `BytesIO`; подтверждающие фреймы отмены могут теряться в процессе отмены), несколько проблем высокой степени серьёзности (общий `IDE_STORAGE_ID` объединяет все IDE-сессии в одну identity, нетипизированный `setattr`, сломанная поверхность управления потоками), и широкую архитектурную задолженность: глобальный синглтон `task_manager`, отсутствие таймаута на запрос, однопоточное чтение stdin, отсутствие валидации схемы на границе протокола и недостаточное покрытие тестами инфраструктуры фоновых задач.

Самая непосредственная проблема для решения — ошибка `send_images` (§2.1) — любая текущая IDE-сессия, запускающая генерацию изображений, производит мусорный вывод для пользователя. Самое значительное архитектурное беспокойство — общий `IDE_STORAGE_ID` (§3.1) — запуск двух IDE-клиентов против одного процесса Chibi приведёт к удивительному перекрёстному загрязнению состояния разговора, выбора модели и поведения сброса.

---

## 1. Обзор архитектуры

Chibi IDE runner — это тонкий stdio-транспорт, предоставляющий доступ к ассистенту Chibi из внешней IDE. Он запускается как отдельный процесс через `chibi run-ide --stdio` (точка входа CLI в `cli.py`). Процесс владеет одним циклом событий asyncio, который управляет:

- **Потоком чтения**, выполняющим блокирующий `sys.stdin.readline()` через `asyncio.to_thread` и передающим JSONL-фреймы в цикл.
- **Диспетчером запросов**, валидирующим каждый фрейм, направляющим его через глобальный `task_manager.run_task(...)` и отслеживающим активные запросы в `_tasks[request_id]` и конкурентность на поток в `_thread_requests[thread_id]`.
- **Записывающим устройством**, сериализующим исходящие фреймы под `asyncio.Lock` в stdout, обеспечивающим атомарную построчную запись на JSONL-провод.

### 1.1 Wire-протокол

Все фреймы имеют формат `json.dumps(..., separators=(",", ":")) + "\n"`. Типы фреймов сервер→клиент: `ready`, `status` (`queued`/`running`), `result` (с `content`, опционально `model`/`provider`), `error` (всегда с `code` и `message`). Типы клиент→сервер: `initialize`, `request`, `cancel`, `shutdown`. Протокол написан вручную без валидации схемы; неизвестные типы сообщений производят ошибку `unknown_message` и цикл продолжается.

### 1.2 Жизненный цикл запроса

1. Чтение строки → strip → `json.loads` → направление в `_handle_message`.
2. При `request`: валидация; если `thread_id` уже имеет активный запрос, отправляется `status/queued`; направление через `task_manager.run_task(self._run_request(...))`; запись задачи; увеличение счётчика на поток.
3. `_run_request` отправляет `status/running`, конструирует локальный для запроса `IDEInterface`, привязанный к колбэку `responses.append`, направляет на `"/command"` или `handle_user_prompt`, и отправляет ровно один терминальный фрейм (`result` / `error` / `error/cancelled` / `error/request_failed`) перед возвратом. Инвариант терминального фрейма сохраняется через `try/finally` на верхнем уровне в корутине запроса.
4. При `cancel`: `task.cancel()`.
5. При `shutdown`: переключение `_stopping = True`; финальный `task_manager.shutdown()` освобождает оставшуюся работу.

### 1.3 Адаптер `IDEInterface`

`IDEInterface` — это подкласс `UserInterface`, который:
- использует единый фиксированный `IDE_STORAGE_ID` (из `constants.py`, значение `-(10**16)`) для chat/user/storage identity во всех IDE-запросах и сессиях,
- хранит `thread_id` и `prompt` для каждого запроса в полях,
- направляет вызовы `send_message` и `send_images` в колбэк `_emit` (`responses.append` запроса),
- делает no-op для всех остальных медиа-методов (audio/video/document) и всех методов действий/вложений,
- выбрасывает `NotImplementedError` для создания/переименования/удаления потока (IDE владеет потоками).

Интерфейс раскрывает `response_model` и `response_provider` как атрибуты экземпляра, которые заполняются обработчиками через `setattr(...)` глубоко в цепочке вызовов — с обходом проверки типов.

### 1.4 Управление задачами

Runner использует глобальный синглтон `task_manager` Chibi (из `services/task_manager.py`) для выполнения корутин запросов. Менеджер реализован как синглтон с семантикой `run_task(coro)`, `_discard_task(...)` и `shutdown()`. Внутри он отслеживает задачи на пользователя в `_tasks` и маппинги на user-id в `_task_to_user_id`. Поведение по умолчанию не накладывает таймаутов, а обработка отмены опирается на `task.exception()` после факта, а не на проверки `task.cancelled()`.

### 1.5 Интеграция с сервисами Chibi

IDE runner повторно использует обработчики бота Chibi (`handle_user_prompt`, `handle_reset`, `handle_image_generation`, выбор модели и т.д.), направляя запросы через них с подстановкой `IDEInterface` вместо `TelegramInterface`. Это означает, что IDE-сессии разделяют логику обработки команд, паттерны обработки ошибок, выбор модели и хранилище — но **без изоляции на сессию** (общий `IDE_STORAGE_ID`).

---

## 2. Ошибки и реальные проблемы

Проблемы ниже дедуплицированы по трём обзорам и организованы по степени серьёзности. Каждая запись включает описание, местоположение, влияние и предлагаемое исправление.

### 2.1 Критические

#### C1. `IDEInterface.send_images()` выводит текстовые repr объектов Python

- **Источники:** Обзор интеграции (kimi-k2.7-code); Обзор транспорта §3.4.
- **Описание:** `send_images` конструирует контент через `"\n".join(str(image) for image in images)`. Когда `image` является `BytesIO`, `str(image)` возвращает `<_io.BytesIO object at 0x...>`. Когда это уже строковый URL, `str` — это no-op (работает случайно). Бинарные данные изображения therefore rendered as Python repr garbage and streamed to the IDE.
- **Местоположение:** `chibi/runners/ide_transport.py` строки ~117-122 (метод `send_images` на `IDEInterface`).
- **Влияние:** Любой путь кода, запускающий генерацию изображений в IDE-сессии — `/imagine`, любой конвейер, производящий изображения, вызовы функций, возвращающие контент с изображениями — в настоящее время производит бессмысленный текст для пользователя. Это наиболее конкретная и видимая runtime-ошибка.
- **Исправление:** Добавить специальный тип фрейма `attachment`: `{"type": "attachment", "kind": "image", "mime": "image/png", "data": "<base64>", "request_id": ...}`. Определять строковые URL и передавать их как есть в фрейме `{"type": "attachment", "kind": "image", "url": ...}`. Не преобразовывать `BytesIO` в строку.

#### C2. Подтверждающий фрейм отмены может никогда не быть отправлен

- **Источники:** Обзор транспорта §2.6.
- **Описание:** `_run_request` выполняет `except asyncio.CancelledError: await self._error("cancelled", ...); raise`. Успех `await` `_error` зависит от специфики версии asyncio — после отмены задачи немедленный последующий await может повторно выбросить `CancelledError` до завершения записи. Подтверждение отмены therefore racy.
- **Местоположение:** `chibi/runners/ide_transport.py` строки ~200-204 (ветвь отмены в `_run_request`).
- **Влияние:** Клиенты могут не увидеть терминальный фрейм `cancelled` для отменённых запросов. В сочетании с семантикой отмены `task_manager` это может привести к тихой отмене, где ни одна из сторон не имеет подтверждённого рукопожатия.
- **Исправление:** Обернуть критические записи терминальных фреймов в `asyncio.shield(...)`, чтобы доставка отмены была decoupled от await писателя. Применить тот же паттерн к фреймам `result` и `error` для укрепления инварианта "ровно один терминальный фрейм".

### 2.2 Высокие

#### H1. `IDE_STORAGE_ID` — синглтон — все IDE-сессии используют одну identity

- **Источники:** Обзор транспорта §3.3; Обзор интеграции (kimi-k2.7-code, "All IDE users share IDE_STORAGE_ID — conversation isolation broken"); Обзор инфраструктуры (Gemini 3.1-pro-preview, "IDE_STORAGE_ID keying collisions across sessions").
- **Описание:** `IDE_STORAGE_ID = -(10**16)` используется как `user_id`, `storage_id` и `chat_id` для всех IDE-запросов. Словарь `_thread_requests` обеспечивает изоляцию конкурентности на уровне потока, но хранилище keyed by `IDE_STORAGE_ID`, поэтому история разговоров, выбор модели, `/info` и состояние сброса shared across every IDE client connecting to the same Chibi process.
- **Местоположение:** `chibi/constants.py:21`; используется в `chibi/runners/ide_transport.py` `IDEInterface.__init__` и throughout handler callsites; database keying in `services/storage` и `services/task_manager`.
- **Влияние:** Запуск двух IDE-окон или двух workspace'ов против одного процесса Chibi produces surprising cross-contamination of state. Вывод `/info`, выбор `/model` и семантика `/reset` all collide.
- **Исправление:** Разрешить IDE передавать per-session `client_id` (workspace UUID) в фрейме `initialize`, затем derive `user_id = hash(client_id)` (или аналог) и использовать как `storage_id`. Документировать семантику синглтона явно до реализации этого.

#### H2. `setattr` smuggling для `response_model` / `response_provider` обходит проверку типов

- **Источники:** Обзор интеграции (kimi-k2.7-code); Обзор транспорта §2.9, §3.5.
- **Описание:** Обработчики изменяют `interface.response_model` и `interface.response_provider` через `setattr` deep inside the call chain. Типы не контролируются. A misbehaving handler can write a non-string (or anything) and break the `result` frame schema. Subtle ordering bugs are possible if a handler emits `send_message` then sets the model — the result frame is assembled only after the handler returns, so this works today by accident, not by contract.
- **Местоположение:** `chibi/runners/ide_transport.py` `IDEInterface` (response_model/response_provider fields, set via setattr from `bot.handle_user_prompt`); read in `_run_request` lines ~195-199.
- **Влияние:** Дрейф контракта вывода; potential schema corruption of `result` frames; fragile ordering dependencies.
- **Исправление:** Типизировать атрибуты как `str | None`, валидировать перед сборкой фрейма `result`. Лучше: не изменять состояние интерфейса вообще — возвращать информацию о модели через кортеж `(text, meta)` от обработчика или через `contextvars.ContextVar`.

#### H3. Инструменты управления потоком выбрасывают `NotImplementedError`

- **Источники:** Обзор интеграции (kimi-k2.7-code).
- **Описание:** `IDEInterface` выбрасывает `NotImplementedError` для создания/переименования/удаления потока. Эти инструменты тем не менее visible to the LLM via Chibi's tool-call surface. If the model invokes one, the call site raises rather than producing a clean error response.
- **Местоположение:** `chibi/runners/ide_transport.py` `IDEInterface` thread-management methods (create/rename/delete).
- **Влияние:** Вызовы инструментов fail with an opaque exception rather than a structured "not supported in IDE mode" error. У модели нет чёткого пути для восстановления.
- **Исправление:** Либо (a) скрыть эти инструменты от модели в IDE-режиме (предпочтительно — они by design принадлежат IDE), либо (b) перехватить вызовы и вернуть структурированный ответ "operation not supported in IDE session".

#### H4. Флаги `_initialized` / `_stopping` изменяются без явной синхронизации

- **Источники:** Обзор транспорта §2.1.
- **Описание:** Эти флаги читаются в event loop без `await`. Сегодня паттерн single-reader обеспечивает implicit atomicity, но сайты мутации распределены across `_handle_message` и `_run_request`, и future change that adds a second reader (or a code path that reads the flag from a different task context) will hit subtle race conditions.
- **Местоположение:** `chibi/runners/ide_transport.py:131-133` (объявление); мутации в `_handle_message` (строки ~233, ~244, ~271) и `_run_request` (строка ~177).
- **Влияние:** Сегодня: низкое (один читатель). Риск при будущих изменениях: средне-высокий.
- **Исправление:** Заменить булевы значения на `asyncio.Event` для `_initialized` (семантика рукопожатия) и `_stopping` (сигнал завершения). Централизовать все мутации и чтения через эти примитивы.

### 2.3 Средние

#### M1. `run_task` не имеет таймаута по умолчанию — задачи могут выполняться бесконечно

- **Источники:** Обзор инфраструктуры (Gemini 3.1-pro-preview); также surfaced as transport §3.12 (no per-request timeout).
- **Описание:** `task_manager.run_task(coro)` не принимает обязательного аргумента `timeout`; default is `None`. Обработчик, который зависает (сетевой stall, бесконечный цикл в вызове инструмента) blocks forever; `_thread_requests` never decrements; request_id stays in `_tasks` indefinitely.
- **Местоположение:** `chibi/services/task_manager.py` signature of `run_task`; caller at `chibi/runners/ide_transport.py:265-267`.
- **Влияние:** Утечка ресурсов, зависший транспорт, eventual `MemoryError` if many requests pile up.
- **Исправление:** Добавить `asyncio.wait_for(timeout, shield=...)` либо на уровне `run_task` (с дефолтом, согласованным в `ready.capabilities`) либо на уровне IDE-запроса (per-call timeout overridable via the `request` frame). Отслеживать таймауты отдельно от отмен.

#### M2. `Coroutine.__name__` теряется после оборачивания в таймаут — `AttributeError` при `.cancel()` и логировании

- **Источники:** Обзор инфраструктуры (Gemini 3.1-pro-preview).
- **Описание:** Когда `run_task` оборачивает корутину в таймаут или другой декоратор, результирующий объект may not preserve `__name__`. Code paths that introspect `coro.__name__` (for logging, for `add_done_callback` bookkeeping, or for `.cancel()` dispatch) raise `AttributeError`.
- **Местоположение:** `chibi/services/task_manager.py` wrapping code; consumers that introspect `__name__`.
- **Влияние:** Сбой при интроспекции; degraded log messages; potential crash on shutdown-time task cancellation.
- **Исправление:** Использовать `functools.wraps` (или эквивалент) при оборачивании корутин; expose the original callable's `__name__` via a `_target` attribute if wrapping is unavoidable.

#### M3. `_discard_task` проверяет отмену через `task.exception()` — антипаттерн

- **Источники:** Обзор инфраструктуры (Gemini 3.1-pro-preview).
- **Описание:** Реализация проверяет `task.exception()` для определения, была ли задача отменена. Это consumes the exception и является order-dependent on task state. Канонический подход: `task.cancelled()` first, then `task.exception()` if not cancelled.
- **Местоположение:** `chibi/services/task_manager.py` `_discard_task` (и связанные cleanup helpers).
- **Влияние:** Subtle state-machine bugs around cancellation; potential `InvalidStateError` if `task.exception()` is called before the task is done.
- **Исправление:** Использовать `task.cancelled()` как primary check; only fall back to `task.exception()` when the task is done and not cancelled.

#### M4. `_discard_task` никогда не удаляет пустые множества из `_tasks` — утечка памяти

- **Источники:** Обзор инфраструктуры (Gemini 3.1-pro-preview).
- **Описание:** After all tasks for a user complete, the set in `_tasks[user_id]` becomes empty but the entry is never removed. `_tasks` grows unboundedly with the number of distinct user IDs over the process lifetime.
- **Местоположение:** `chibi/services/task_manager.py` `_tasks` dict; cleanup paths in `_discard_task`.
- **Влияние:** Slow memory growth; long-running IDE sessions or any multi-user process leak dict entries.
- **Исправление:** In `_discard_task`, after the last task is removed, `if not _tasks[user_id]: del _tasks[user_id]`. Add the same cleanup for `_task_to_user_id`.

#### M5. `SingletonMeta` не является потокобезопасным

- **Источники:** Обзор инфраструктуры (Gemini 3.1-pro-preview).
- **Описание:** `SingletonMeta` metaclass performs the singleton check and instantiation without a lock. Concurrent first-access from multiple threads can produce multiple instances.
- **Местоположение:** `chibi/services/task_manager.py` `SingletonMeta`.
- **Влияние:** Theoretical race in test harnesses or any code path that constructs the singleton from a non-main thread (e.g., a worker thread spawned by `asyncio.to_thread`). Today the IDE runner constructs it on the main loop, so impact is low — but other Chibi entrypoints may not.
- **Исправление:** Использовать `threading.Lock` around the existence check and instantiation in `__call__`.

#### M6. Проверка коллизии `request_id` слишком строгая

- **Источники:** Обзор транспорта §2.2.
- **Описание:** `_handle_message` rejects with `malformed_request` whenever `request_id in self._tasks`. This conflates "still in flight" with "id was reused post-completion." Since `_tasks` is popped in the `finally` of `_run_request`, post-completion reuse is safe — only in-flight collisions are real bugs. The error message "request_id is already in use" misleads debugging.
- **Местоположение:** `chibi/runners/ide_transport.py:254`.
- **Влияние:** Клиенты, использующие стабильные схемы request-id (e.g., `<thread>:<n>`), отклоняются без необходимости.
- **Исправление:** Различать "still in flight" от "completed." Only reject when the id is currently in `_tasks`.

#### M7. `shutdown` не освобождает активные запросы

- **Источники:** Обзор транспорта §2.12.
- **Описание:** `shutdown` flips `_stopping = True` and returns to the loop. If the client sends `shutdown` and then keeps stdin open (graceful shutdown pattern), the loop continues reading and dispatching new requests. If stdin is closed, `task_manager.shutdown()` drains existing tasks in the `finally` of `run()`.
- **Местоположение:** `chibi/runners/ide_transport.py:272-274`.
- **Влияние:** Неоднозначная семантика завершения; скомпрометированный stdin pipe может simultaneously drain and block shutdown.
- **Исправление:** Сделать семантику `shutdown` явной: "do not accept new requests; drain existing." On receiving `shutdown`, break out of the read loop immediately and `await asyncio.gather(*self._tasks.values(), return_exceptions=True)` before exit.

#### M8. Команда `/model` дублирует логику выбора модели Telegram

- **Источники:** Обзор интеграции (kimi-k2.7-code).
- **Описание:** IDE runner re-implements model-selection logic in the `/model` command path rather than delegating to a shared service. Drift between Telegram and IDE code paths is a long-term risk.
- **Местоположение:** `chibi/runners/ide_transport.py` обработка `/model` (~строки 172-191).
- **Влияние:** Поведенческий дрейф между IDE и Telegram сессиями; дублированный код; harder to add per-interface customization.
- **Исправление:** Выделить выбор модели в сервис, принимающий `UserInterface` и `storage_id`. Оба — Telegram и IDE — вызывают один сервис.

### 2.4 Низкие

#### L1. `task_manager.run_task(...)` — fire-and-forget — нет done-callback

- **Источники:** Обзор транспорта §2.8.
- **Описание:** Задачи отправляются через `run_task` и записываются в `_tasks`, but the runner never installs `add_done_callback`. Unexpected exceptions inside `_run_request` are mostly caught by its top-level `except Exception` (emitting `request_failed`), but `CancelledError` re-raises and may not emit reliably (see C2).
- **Местоположение:** `chibi/runners/ide_transport.py:265-267`.
- **Влияние:** Missing observability surface for task outcomes.
- **Исправление:** `task.add_done_callback(self._on_task_done)` для удаления из `_tasks` и логирования неожиданных причин исключений.

#### L2. Порядок инкремента/декремента `_thread_requests` хрупкий

- **Источники:** Обзор транспорта §2.3.
- **Описание:** Счётчик увеличивается *before* dispatching to `task_manager` and rolled back only if `run_task` returns `None`. Today this is correct; future edit could remove the rollback and silently leak the counter.
- **Местоположение:** `chibi/runners/ide_transport.py` строки ~264-269.
- **Влияние:** Latent bug surface; not currently buggy.
- **Исправление:** Переместить инкремент ниже проверки `task is None`, so the lifetime of the counter matches the lifetime of the dispatched task.

#### L3. `error_code` / `error_message` проверяются после вызовов `send_message` — контракт, зависящий от порядка

- **Источники:** Обзор транспорта §2.10.
- **Описание:** `/model <bad>` sets `interface.error_code` and emits error. Currently the error path takes precedence over `result`. This works but the contract is implicit.
- **Местоположение:** `chibi/runners/ide_transport.py` строки ~194-198.
- **Влияние:** Future handler may set `error_code` *and* call `send_message`, leading to ambiguity.
- **Исправление:** Документировать контракт на `IDEInterface`: "If `error_code` is set after handler completion, emit `error` instead of `result`."

#### L4. Непоследовательная обработка ошибок: `/imagine` `ValueError` становится generic `request_failed` вместо `invalid_argument`

- **Источники:** Обзор интеграции (kimi-k2.7-code).
- **Описание:** Другие команды raise structured `invalid_argument` errors for bad input; `/imagine` lets `ValueError` bubble up to the generic `request_failed` path.
- **Местоположение:** `chibi/runners/ide_transport.py` dispatch `/imagine`.
- **Влияние:** Непоследовательный опыт клиента; IDE cannot programmatically distinguish "bad input" от "internal failure."
- **Исправление:** Ловить `ValueError` на границе dispatch команды и конвертировать в `invalid_argument` с информативным сообщением.

#### L5. `_set_ide_error` покрывает только декорированные сервисные функции

- **Источники:** Обзор интеграции (kimi-k2.7-code).
- **Описание:** Механизм декорирования ошибок IDE применяется только к функциям, которые explicitly opt into the decorator. Any code path that constructs an `IDEInterface` and calls a non-decorated handler does not get consistent error-code translation.
- **Местоположение:** `chibi/runners/ide_transport.py` or shared error-decoration module (decorator apply site).
- **Влияние:** Пробелы в покрытии перевода ошибок.
- **Исправление:** Либо декорировать функции-точки входа (so all calls go through the decorator) либо применять декоратор универсально к обработчикам, принимающим `UserInterface`.

#### L6. Чтение JSON-линии не имеет лимита длины — DoS через истощение памяти

- **Источники:** Обзор транспорта §2.7.
- **Описание:** `sys.stdin.readline()` will read multi-GB lines. `json.loads` may raise `MemoryError` (not currently caught). A malicious or buggy client can OOM the runner.
- **Местоположение:** `chibi/runners/ide_transport.py:152-155`.
- **Влияние:** Поверхность для DoS.
- **Исправление:** Ограничить длину входной строки (e.g., 1 MB). Treat over-cap as `malformed_request`.

#### L7. Однопоточное чтение stdin не имеет буфера для частичных строк

- **Источники:** Обзор транспорта §3.1.
- **Описание:** `readline` returns when it sees `\n`. If a client writes a frame split across two reads of `stdio` (e.g., due to pipe-buffer fragmentation of a large prompt), `readline` returns a partial line, `json.loads` raises, and we send `malformed_request`. The trailing fragment is lost.
- **Местоположение:** `chibi/runners/ide_transport.py` `_read_line`.
- **Влияние:** Надёжность для больших промптов, превышающих PIPE_BUF.
- **Исправление:** Документировать лимит (фреймы должны помещаться в PIPE_BUF) или переключиться на протокол с префиксом длины.

#### L8. Блокировка `_write` не защищает работу `json.dumps`

- **Источники:** Обзор транспорта §2.5.
- **Описание:** `json.dumps` runs before lock acquisition. Today this is fine (lock holds write+flush; libc line-buffers). If stdout is reconfigured (e.g., to non-line-buffered), the lock could be released mid-frame.
- **Местоположение:** `chibi/runners/ide_transport.py:139-145`.
- **Влияние:** Скрытый риск при переконфигурации stdout.
- **Исправление:** Документировать инвариант; acquire the lock earlier if frame atomicity becomes a concern.

#### L9. `_handle_message` для `cancel` не ждёт отменённую задачу

- **Источники:** Обзор транспорта §2.13.
- **Описание:** `self._tasks[request_id].cancel()` is non-blocking. The task removes itself in its `finally`. After cancel returns, `_tasks[request_id]` may still exist for a microsecond.
- **Местоположение:** `chibi/runners/ide_transport.py:267-271`.
- **Влияние:** Незначительно; не баг.
- **Исправление:** Установить done-callback (see L1) and surface exceptions.

#### L10. Нет сообщения `cancel-all` или `stop`

- **Источники:** Обзор транспорта §3.7.
- **Описание:** IDE can only cancel requests individually.
- **Местоположение:** `chibi/runners/ide_transport.py` `_handle_message`.
- **Влияние:** Низкое; неудобство UX для массовой отмены.
- **Исправление:** Добавить сообщение `cancel_all` client→server.

#### L11. Нет max concurrent requests per thread

- **Источники:** Обзор транспорта §3.11.
- **Описание:** `_thread_requests` tracks count but only emits "queued" status — no back-pressure. An IDE that floods requests for one thread spawns unbounded concurrent tasks.
- **Местоположение:** `chibi/runners/ide_transport.py:255-269`.
- **Влияние:** Истощение ресурсов под нагрузкой.
- **Исправление:** Конфигурируемый `MAX_PER_THREAD`; queue beyond that.

#### L12. CLI: `ide` без `--stdio` silently does nothing useful

- **Источники:** Обзор транспорта §2.15.
- **Описание:** `chibi ide` (no flag) prints help. `chibi ide start` errors from click ungracefully. No proper subcommand structure.
- **Местоположение:** `cli.py:30-32`.
- **Влияние:** UX CLI.
- **Исправление:** Рефакторить `ide` в `@main.command()` с `--stdio`. No group needed.

#### L13. `run_ide()` does side-effect `import chibi.config`

- **Источники:** Обзор транспорта §2.16.
- **Описание:** Hidden initialization via `# noqa: F401` import.
- **Местоположение:** `chibi/runners/ide.py:9-10`.
- **Влияние:** Скрытая зависимость; conventions silently enforced.
- **Исправление:** Явный `chibi.config.load()` if that's what it does.

#### L14. Logger reconfigured on every `run()`

- **Источники:** Обзор транспорта §3.9.
- **Описание:** `logger.remove()` strips *all* sinks, then re-adds stderr INFO. Anything that added a sink (e.g., tests) is wiped.
- **Местоположение:** `chibi/runners/ide_transport.py:284`.
- **Влияние:** Subtle in tests; low in production.
- **Исправление:** Использовать `logger.add(...)` без `remove` (идемпотентно) или сузить область удаления.

#### L15. `request_id` allowed to be `None` for non-request messages but strict for `request`

- **Источники:** Обзор транспорта §2.11.
- **Описание:** Mixed validation: `request` requires non-empty str `request_id`; `cancel`/`shutdown` allow `None`. The framework's mental model isn't fully encoded.
- **Местоположение:** `chibi/runners/ide_transport.py:158-168` vs request_id extraction at line ~216.
- **Влияние:** Несоответствие типа/схемы.
- **Исправление:** Использовать typed `TypedDict` or `pydantic` models for each protocol message, validated at boundary.

#### L16. `_stopping` is read but never reset; no "cancel-all and shut down" API

- **Источники:** Обзор транспорта §2.14.
- **Описание:** Loop exits on EOF or shutdown but never on cancel-all.
- **Местоположение:** `chibi/runners/ide_transport.py:280-288`.
- **Влияние:** Незначительно.
- **Исправление:** Опционально.

---

## 3. Архитектурные проблемы

Проблемы ниже — не баги, но представляют структурную задолженность, которая будет накапливаться со временем. Организованы по уровню риска.

### 3.1 Высокий риск

#### A1. Общий `IDE_STORAGE_ID` разрушает изоляцию IDE-сессий
См. H1. Паттерн синглтон identity протекает между сессиями. Серьёзность усугубляется тем, что keying хранилища, keying task-manager, выбор модели и поведение сброса all collide. Это самая значительная архитектурная проблема.

#### A2. `task_manager` — это глобальный синглтон процесса

- **Источники:** Обзор транспорта §3.8.
- **Описание:** `task_manager` is a free function call into a global. Tests cannot easily inject a fake. The runner's behavior is partly determined by global state.
- **Местоположение:** `chibi/services/task_manager.py`; consumer at `chibi/runners/ide_transport.py:13`.
- **Влияние:** Трудно unit-test the runner in isolation; hard to run multiple task managers in the same process (e.g., for isolation in IDE multi-workspace scenarios).
- **Исправление:** Определить `TaskManagerProtocol` и принимать его через `__init__`. Runner конструирует (или получает) конкретный экземпляр per process.

#### A3. Нет валидации схемы протокола на границе

- **Источники:** Обзор транспорта §3.2; §2.11.
- **Описание:** Фреймы валидируются ad-hoc in `_handle_message` and `_valid_request`. No `pydantic` models, no schema registry, no message-type dispatch table.
- **Местоположение:** `chibi/runners/ide_transport.py` `_handle_message`, `_valid_request`.
- **Влияние:** Adding new message types requires touching multiple functions; drift between server→client and client→server schemas is invisible until runtime; type-safety is manual.
- **Исправление:** Ввести `pydantic` models per message type. Заменить `elif`-цепь в `_handle_message` на `{type: handler}` dispatch. Reject unknown fields at the boundary.

#### A4. Нет streaming output — все ответы буферизуются в один фрейм `result`

- **Источники:** Обзор транспорта §3.10.
- **Описание:** Все вызовы `send_message` concatenate into one big `result` frame. For long generations, the user sees nothing until completion.
- **Местоположение:** `chibi/runners/ide_transport.py` `content = "\n".join(responses)` then `await self._write(result)`.
- **Влияние:** UX for long generations is unacceptable; user has no progress indication beyond the `running` status.
- **Исправление:** Emit `{"type": "delta", "request_id": ..., "content": ...}` from `send_message`; final `result` becomes optional or carries the concatenated string + per-delta offsets.

#### A5. Конкурентность на уровне потока изолирована, но хранилище — нет

- **Источники:** Обзор интеграции (kimi-k2.7-code).
- **Описание:** `_thread_requests` provides per-thread concurrency isolation in the runner, but storage and conversation history use the shared `IDE_STORAGE_ID`. Half-isolated concurrency is worse than fully-isolated or fully-shared — it creates confusing invariants.
- **Местоположение:** `chibi/runners/ide_transport.py` `_thread_requests`; `IDEInterface` storage fields.
- **Влияние:** Удивительное перекрёстное загрязнение: история одного потока may interleave with another thread's user.
- **Исправление:** Выводить identity хранилища per thread (или per session) — см. H1.

#### A6. `IDEInterface` смешивает входные данные, выходной sink и метаданные

- **Источники:** Обзор транспорта §4.1 (separation of concerns); §3.5.
- **Описание:** `IDEInterface` carries request input (`thread_id`, `prompt`), output sink (`responses`), metadata (`response_model`, `response_provider`), and error state (`error_code`, `error_message`). Three responsibilities, one class.
- **Местоположение:** `chibi/runners/ide_transport.py` class `IDEInterface`.
- **Влияние:** Mutation via `setattr` from deep in the handler chain is hard to reason about; type safety is unenforced; testing each concern requires the full interface.
- **Исправление:** Разделить на `RequestContext` (immutable: thread_id, prompt, workspace_root) и `ResponseSink` (mutable: list of messages, model info, error state). Обработчики получают оба.

### 3.2 Средний риск

#### A7. Обработка команд инлайн в `_run_request`

- **Источники:** Обзор транспорта §3.6.
- **Описание:** Цепь if/elif для `/reset`, `/help`, `/model` и т.д. lives in the central request coroutine. As commands grow, the function becomes a god method; adding `/foo` requires touching the chain, the `COMMANDS` constant, and the help message.
- **Местоположение:** `chibi/runners/ide_transport.py:172-191`.
- **Влияние:** Drift between constant and dispatch; maintenance burden.
- **Исправление:** Извлечь command registry: `commands.register("/model", handle_model)`. Dispatch becomes a single dict lookup.

#### A8. Паттерн `setattr` smuggling как архитектурный запах

- **Источники:** Обзор интеграции; Обзор транспорта §3.5.
- **Описание:** См. H2. Beyond the immediate type-safety bug, the pattern of writing response metadata onto the interface from inside a handler is structurally wrong — the interface mixes data sink with side-channel metadata.
- **Влияние:** Делает интерфейс трудным для мока, трудным для проверки типов и лёгким для неправильного использования.
- **Исправление:** Иметь обработчики возвращающими `(text, meta)` от одного контрактного метода или использовать `contextvars.ContextVar` для cross-cutting metadata.

#### A9. Нет observable healthcheck / liveness signal

- **Источники:** Обзор транспорта §4.4.
- **Описание:** No `ping`/`pong` server→client or client→server. CI harnesses and process supervisors have no clean liveness probe.
- **Местоположение:** `chibi/runners/ide_transport.py` (no ping frame in protocol).
- **Влияние:** Операционный пробел.
- **Исправление:** Добавить фреймы `{"type": "ping"}` и `{"type": "pong"}`.

#### A10. `interface.error_code` устанавливается и забывается на success path

- **Источники:** Обзор транспорта §2.10.
- **Описание:** Error path checked *after* `send_message` calls. Implicit ordering contract.
- **Местоположение:** `chibi/runners/ide_transport.py` строки ~194-198.
- **Влияние:** Future handler may set error_code and call send_message, leading to ambiguity.
- **Исправление:** Документировать контракт; или рефакторить так, чтобы состояние ошибки возвращалось как discriminated union.

### 3.3 Низкий риск

#### A11. Константа `COMMANDS` vs. inline dispatch может расходиться

- **Источники:** Обзор транспорта §3.6.
- **Описание:** Help message, dispatch chain, and `COMMANDS` constant are maintained separately.
- **Влияние:** Дрейф документации.
- **Исправление:** Единый источник правды — derive help from the registry.

#### A12. Нет флага `--log-level` в CLI

- **Источники:** Обзор транспорта §4.4.
- **Описание:** Stderr loguru sink is INFO; no way to change verbosity from CLI.
- **Местоположение:** `cli.py`; `ide_transport.py:284`.
- **Влияние:** Операционное неудобство.
- **Исправление:** Добавить опцию `--log-level` в `chibi run-ide --stdio`.

#### A13. Per-request latency not surfaced

- **Источники:** Обзор транспорта §4.4.
- **Описание:** `started_at` is captured but not exposed. No `duration_ms` in completion status.
- **Влияние:** Пробел в наблюдаемости.
- **Исправление:** Emit `{"type": "status", "state": "completed", "duration_ms": ...}` on completion.

---

## 4. Анализ покрытия тестами

### 4.1 Что протестировано

Согласно обзорам, **тесты не включены в файлы IDE runner**. Обзор инфраструктуры (Gemini 3.1-pro-preview) явно отмечает, что `BackgroundTaskManager` lacks isolated unit tests, а обзор транспорта отмечает отсутствие `test_ide*` or `tests/*ide*` patterns.

### 4.2 Чего не хватает

#### T1. Нет тестов таймаута для `BackgroundTaskManager`

- **Источники:** Обзор инфраструктуры.
- **Описание:** Timeout behavior for `run_task` (M1) is completely untested. With the bug fix, this must be the first new test set.
- **Цель покрытия:** default timeout; custom timeout; timeout-fires-cancels-task; timeout-on-already-done-task.

#### T2. Нет тестов разрешения backend `DatabaseCache`

- **Источники:** Обзор инфраструктуры.
- **Описание:** Логика выбора backend (in-memory vs. disk) не протестирована.
- **Цель покрытия:** env-var driven backend; default backend; fallback on backend init failure.

#### T3. `BackgroundTaskManager` lacks isolated unit tests

- **Источники:** Обзор инфраструктуры.
- **Описание:** No unit-test harness for `run_task`, `_discard_task`, `shutdown`.
- **Цель покрытия:** Per-method unit tests with synthetic coroutines (raising, hanging, cancelling, succeeding, timing out).

#### T4. Транспортные тесты используют busy-wait `asyncio.sleep(0.01)` — flaky

- **Источники:** Обзор инфраструктуры.
- **Описание:** Existing tests (if any) poll with `await asyncio.sleep(0.01)`. This is timing-dependent and flaky on CI.
- **Цель покрытия:** Replace busy-wait with `asyncio.Event` signaling, or use `asyncio.wait_for` with explicit deadlines.

#### T5. Модульный `_gate` global for sync prevents parallel test execution

- **Источники:** Обзор инфраструктуры.
- **Описание:** Глобальный примитив синхронизации на уровне модуля means tests that touch it cannot run in parallel within the same process.
- **Влияние:** Медленный CI.
- **Исправление:** Переместить gate в per-test fixture или на уровень класса.

#### T6. Heavy mocking of `handle_user_prompt` rather than integration testing

- **Источники:** Обзор инфраструктуры.
- **Описание:** Тесты подменяют обработчик instead of running an end-to-end IDE session.
- **Влияние:** Interface bugs (C1, H2) would not be caught by current-style tests.
- **Исправление:** Добавить интеграционные тесты, которые запускают IDE runner in-process с synthetic stdin/stdout и exercise the full request→response path.

#### T7. Нет edge-case тестов для пользователя `IDE_STORAGE_ID`

- **Источники:** Обзор инфраструктуры.
- **Описание:** Edge cases specific to the shared-storage identity: drop history, image limits, model selection across sessions.
- **Цель покрытия:** Per-test: drive the runner with a sequence of `initialize` + multiple requests simulating two clients; assert storage state.

#### T8. Нет тестов shutdown-while-tasks-in-flight

- **Источники:** Обзор инфраструктуры.
- **Описание:** What happens when `shutdown` arrives while requests are mid-flight? With timeouts? Without?
- **Цель покрытия:** Send `shutdown` mid-request; assert drain or kill behavior; test with timeout wrapping.

#### T9. Нет edge-case тестов для согласования версии протокола

- **Источники:** Обзор инфраструктуры.
- **Описание:** `unsupported_protocol_version` path описан в transport §2.7 but not exercised in tests.
- **Цель покрытия:** Min version mismatch; max version mismatch; missing version; non-int version; negative version.

#### T10. Нет round-trip / synthetic-stdio tests

- **Источники:** Обзор транспорта §4.6.
- **Описание:** Рекомендуются тесты, которые driving the runner with `io.StringIO` for stdin/stdout.
- **Цель покрытия:** Happy path; error path; out-of-order cancel; protocol version mismatch; slow handler / timeout.

#### T11. Нет тестов пути генерации изображений

- **Источники:** Обзор транспорта §4.6.
- **Описание:** The `send_images` bug (C1) is the most visible runtime bug — it has no test coverage.
- **Цель покрытия:** Drive `/imagine`; assert the response includes a valid attachment frame, not Python reprs.

#### T12. Нет observability / cancellation-ack tests

- **Источники:** Обзор транспорта §2.6.
- **Описание:** The cancellation ack racy behavior (C2) is untested; clients cannot rely on the contract.
- **Цель покрытия:** Cancel mid-request; assert exactly one `cancelled` frame is emitted.

### 4.3 Проблемы качества тестов

- **Flaky timing patterns** (T4): replace busy-waits with event-driven waits.
- **Global synchronization** (T5): рефакторить в per-test fixtures.
- **Mocking over integration** (T6): добавить интеграционные тесты для полного пути запроса.
- **Missing invariants**: no test enforces the "exactly one terminal frame per request" invariant.

---

## 5. Рекомендации по улучшению

### 5.1 Немедленные (следующий спринт)

1. **Исправить `IDEInterface.send_images` (C1).** Это user-visible runtime bug. Добавить typed фрейм `attachment` и направлять `str` URLs through it; не преобразовывать `BytesIO` в строку.
2. **Исправить cancellation ack (C2).** Обернуть записи терминальных фреймов в `asyncio.shield`, чтобы отмена не могла drop the ack.
3. **Добавить параметр `timeout` в `task_manager.run_task` (M1).** По умолчанию разумное значение, согласованное в `ready.capabilities`. Per-request override via the `request` frame.
4. **Добавить тесты для таймаута `BackgroundTaskManager` (T1) и `send_images` (T11).** Закрепить исправление и предотвратить регрессию.
5. **Документировать семантику синглтона `IDE_STORAGE_ID` (H1).** До реализации per-session identity, multi-client caveat must be visible in the constants file and in IDE docs.

### 5.2 Краткосрочные (следующие 1–2 месяца)

1. **Per-session identity (H1).** Разрешить IDE передавать `client_id` (workspace UUID) в `initialize`. Выводить `user_id = hash(client_id)`. Проводить through `IDEInterface`.
2. **Заменить `setattr` на typed returns (H2, A8).** Обработчики возвращают `(text, meta)`. Удалить паттерн мутации.
3. **Скрыть неподдерживаемые инструменты управления потоком от LLM (H3).** Фильтровать поверхность инструментов per-interface.
4. **Валидация схемы на границе протокола (A3).** Ввести `pydantic` models per message type; заменить `elif` chain на dispatch table.
5. **`functools.wraps` на оборачивании таймаута в `run_task` (M2).** Preserve `__name__`.
6. **`_discard_task` cleanup of empty sets (M4).** Удалять пустые записи из `_tasks` и `_task_to_user_id`.
7. **`SingletonMeta` thread safety (M5).** Добавить `threading.Lock`.
8. **Исправить проверку отмены `_discard_task` (M3).** Использовать `task.cancelled()` first.
9. **Строгая проверка коллизии `request_id` только для in-flight (M6).** Allow post-completion reuse.
10. **Явная семантика drain при shutdown (M7).** On `shutdown`, break the read loop and gather in-flight tasks.
11. **Покрытие тестовых пробелов (T1–T12).** Добавить unit-тесты для `BackgroundTaskManager`, интеграционные тесты для runner, edge cases для пользователя `IDE_STORAGE_ID`, и согласование версии протокола.

### 5.3 Долгосрочные (следующий квартал)

1. **Streaming output via `delta` frames (A4).** Заменить единственный фрейм `result` на optional deltas.
2. **`TaskManagerProtocol` injection (A2).** Сделать runner testable in isolation; allow per-process isolation.
3. **Per-session storage identity (A5).** Полное исправление пробела изоляции хранилища.
4. **Разделить `IDEInterface` на `RequestContext` + `ResponseSink` (A6).** Более чистое разделение concerns.
5. **Command registry (A7).** Заменить inline dispatch на registry. Единый источник правды для `COMMANDS`.
6. **Healthcheck (`ping`/`pong`) and per-request latency (A9, A13).** Surfaces observability для CI и production.
7. **CLI: рефакторить `ide` в `@main.command()` с `--stdio` (L12).**
8. **Документировать контракт `IDEInterface` (L3, A10).** Error-state-vs-success ordering, send_image semantics, single-terminal-frame invariant.

---

## 6. Приложение: Полный каталог проблем

| ID | Категория | Серьёзность | Источник | Описание | Расположение |
|---|---|---|---|---|---|
| C1 | Bug | Critical | kimi-k2.7-code, MiniMax-M3 | `IDEInterface.send_images()` выводит repr объектов Python (`BytesIO` → `<_io.BytesIO object at 0x...>`) как текст — мусор для бинарных изображений | `chibi/runners/ide_transport.py` ~117-122 |
| C2 | Bug | Critical | MiniMax-M3 | Подтверждающий фрейм отмены может никогда не быть отправлен, потому что `await self._error(...)` внутри `except CancelledError` является racy across asyncio versions | `chibi/runners/ide_transport.py` ~200-204 |
| H1 | Bug | High | MiniMax-M3, kimi-k2.7-code, Gemini-3.1-pro-preview | `IDE_STORAGE_ID` — синглтон — все IDE-сессии используют одну identity; перекрёстное загрязнение `/info`, `/model`, `/reset`, истории разговоров | `chibi/constants.py:21`; `chibi/runners/ide_transport.py` `IDEInterface.__init__` |
| H2 | Bug | High | kimi-k2.7-code, MiniMax-M3 | `setattr` smuggling для `response_model`/`response_provider` обходит проверку типов; misbehaving handlers могут повредить схему фрейма `result` | `chibi/runners/ide_transport.py` `IDEInterface` (set via `bot.handle_user_prompt`); read ~195-199 |
| H3 | Bug | High | kimi-k2.7-code | Инструменты управления потоком (create/rename/delete) выбрасывают `NotImplementedError`, но видны LLM через поверхность tool-call | `chibi/runners/ide_transport.py` `IDEInterface` thread methods |
| H4 | Bug | High | MiniMax-M3 | Флаги `_initialized`/`_stopping` mutate without explicit synchronization — паттерн single-reader обеспечивает implicit atomicity today, fragile to future changes | `chibi/runners/ide_transport.py:131-133`; mutations at ~233, ~244, ~271, ~177 |
| M1 | Bug | Medium | Gemini-3.1-pro-preview, MiniMax-M3 | Таймаут `run_task` по умолчанию `None` — задачи могут выполняться бесконечно | `chibi/services/task_manager.py` `run_task`; caller at `chibi/runners/ide_transport.py:265-267` |
| M2 | Bug | Medium | Gemini-3.1-pro-preview | `Coroutine.__name__` lost after timeout wrapping — `AttributeError` on `.cancel()` и логирование | `chibi/services/task_manager.py` wrapping code |
| M3 | Bug | Medium | Gemini-3.1-pro-preview | `_discard_task` checks cancellation via `task.exception()` — антипаттерн; use `task.cancelled()` first | `chibi/services/task_manager.py` `_discard_task` |
| M4 | Bug | Medium | Gemini-3.1-pro-preview | `_discard_task` never deletes empty sets from `_tasks` — утечка памяти | `chibi/services/task_manager.py` `_tasks` dict |
| M5 | Bug | Medium | Gemini-3.1-pro-preview | `SingletonMeta` not thread-safe — concurrent first-access может произвести несколько экземпляров | `chibi/services/task_manager.py` `SingletonMeta` |
| M6 | Bug | Medium | MiniMax-M3 | Проверка коллизии `request_id` слишком строгая — отклоняет post-completion reuse с misleading "request_id is already in use" | `chibi/runners/ide_transport.py:254` |
| M7 | Bug | Medium | MiniMax-M3 | `shutdown` не освобождает активные запросы — неоднозначная семантика завершения | `chibi/runners/ide_transport.py:272-274` |
| M8 | Bug | Medium | kimi-k2.7-code | Команда `/model` дублирует логику выбора модели Telegram — дрейф между интерфейсами | `chibi/runners/ide_transport.py` ~172-191 |
| L1 | Bug | Low | MiniMax-M3 | `task_manager.run_task(...)` — fire-and-forget — нет done-callback | `chibi/runners/ide_transport.py:265-267` |
| L2 | Bug | Low | MiniMax-M3 | Порядок инкремента/декремента `_thread_requests` хрупкий (инкремент before dispatch) | `chibi/runners/ide_transport.py` ~264-269 |
| L3 | Bug | Low | MiniMax-M3 | `error_code`/`error_message` checked after `send_message` calls — контракт, зависящий от порядка | `chibi/runners/ide_transport.py` ~194-198 |
| L4 | Bug | Low | kimi-k2.7-code | `/imagine` `ValueError` становится generic `request_failed` вместо `invalid_argument` | `chibi/runners/ide_transport.py` `/imagine` dispatch |
| L5 | Bug | Low | kimi-k2.7-code | `_set_ide_error` only covers decorated service functions — пробелы в покрытии | `chibi/runners/ide_transport.py` decorator apply site |
| L6 | Bug | Low | MiniMax-M3 | JSON line read has no length cap — DoS via memory exhaustion | `chibi/runners/ide_transport.py:152-155` |
| L7 | Bug | Low | MiniMax-M3 | Однопоточное чтение stdin не имеет буфера для частичных строк — большие промпты могут превышать PIPE_BUF | `chibi/runners/ide_transport.py` `_read_line` |
| L8 | Bug | Low | MiniMax-M3 | Блокировка `_write` не защищает работу `json.dumps` — скрытый риск при переконфигурации stdout | `chibi/runners/ide_transport.py:139-145` |
| L9 | Bug | Low | MiniMax-M3 | `_handle_message` для `cancel` не ждёт отменённую задачу | `chibi/runners/ide_transport.py:267-271` |
| L10 | Bug | Low | MiniMax-M3 | Нет сообщения `cancel-all` или `stop` | `chibi/runners/ide_transport.py` `_handle_message` |
| L11 | Bug | Low | MiniMax-M3 | Нет max concurrent requests per thread — истощение ресурсов под нагрузкой | `chibi/runners/ide_transport.py:255-269` |
| L12 | Bug | Low | MiniMax-M3 | CLI: `ide` без `--stdio` silently does nothing useful | `cli.py:30-32` |
| L13 | Bug | Low | MiniMax-M3 | `run_ide()` does side-effect `import chibi.config` — скрытая зависимость | `chibi/runners/ide.py:9-10` |
| L14 | Bug | Low | MiniMax-M3 | Logger reconfigured on every `run()` — `logger.remove()` wipes any other sinks (e.g., in tests) | `chibi/runners/ide_transport.py:284` |
| L15 | Bug | Low | MiniMax-M3 | `request_id` allowed to be `None` for non-request messages but strict for `request` | `chibi/runners/ide_transport.py:158-168` vs ~216 |
| L16 | Bug | Low | MiniMax-M3 | `_stopping` is read but never reset; no "cancel-all and shut down" API | `chibi/runners/ide_transport.py:280-288` |
| A1 | Architecture | High | MiniMax-M3, kimi-k2.7-code | Общий `IDE_STORAGE_ID` разрушает изоляцию IDE-сессий (то же, что H1) | — |
| A2 | Architecture | High | MiniMax-M3 | `task_manager` — глобальный синглтон процесса — трудно тестировать, трудно изолировать | `chibi/services/task_manager.py`; consumer at `chibi/runners/ide_transport.py:13` |
| A3 | Architecture | High | MiniMax-M3 | Нет валидации схемы протокола на границе — фреймы валидируются ad-hoc | `chibi/runners/ide_transport.py` `_handle_message`, `_valid_request` |
| A4 | Architecture | High | MiniMax-M3 | Нет streaming output — все ответы буферизуются в один фрейм `result` | `chibi/runners/ide_transport.py` `content = "\n".join(responses)` |
| A5 | Architecture | High | kimi-k2.7-code | Конкурентность на уровне потока изолирована, но хранилище нет — half-isolated invariants | `chibi/runners/ide_transport.py` `_thread_requests`; `IDEInterface` storage fields |
| A6 | Architecture | High | MiniMax-M3 | `IDEInterface` смешивает входные данные, выходной sink и метаданные — три responsibilities, один class | `chibi/runners/ide_transport.py` class `IDEInterface` |
| A7 | Architecture | Medium | MiniMax-M3 | Обработка команд инлайн в `_run_request` — god method, дрейф между constant и dispatch | `chibi/runners/ide_transport.py:172-191` |
| A8 | Architecture | Medium | kimi-k2.7-code, MiniMax-M3 | Паттерн `setattr` smuggling как архитектурный запах (расширяет H2) | — |
| A9 | Architecture | Medium | MiniMax-M3 | Нет observable healthcheck / liveness signal — нет `ping`/`pong` | `chibi/runners/ide_transport.py` (no ping frame) |
| A10 | Architecture | Medium | MiniMax-M3 | `interface.error_code` set-and-forgotten on success path — implicit ordering contract | `chibi/runners/ide_transport.py` ~194-198 |
| A11 | Architecture | Low | MiniMax-M3 | Константа `COMMANDS` vs. inline dispatch может расходиться | `chibi/runners/ide_transport.py` |
| A12 | Architecture | Low | MiniMax-M3 | Нет флага `--log-level` в CLI | `cli.py`; `ide_transport.py:284` |
| A13 | Architecture | Low | MiniMax-M3 | Per-request latency not surfaced | `chibi/runners/ide_transport.py` |
| T1 | Test | — | Gemini-3.1-pro-preview | Нет тестов таймаута для `BackgroundTaskManager` | `tests/` (missing) |
| T2 | Test | — | Gemini-3.1-pro-preview | Нет тестов разрешения backend `DatabaseCache` | `tests/` (missing) |
| T3 | Test | — | Gemini-3.1-pro-preview | `BackgroundTaskManager` lacks isolated unit tests | `tests/` (missing) |
| T4 | Test | — | Gemini-3.1-pro-preview | Транспортные тесты используют busy-wait `asyncio.sleep(0.01)` — flaky | `tests/` (existing) |
| T5 | Test | — | Gemini-3.1-pro-preview | Модульный `_gate` global prevents parallel test execution | `tests/` (existing) |
| T6 | Test | — | Gemini-3.1-pro-preview | Heavy mocking of `handle_user_prompt` rather than integration testing | `tests/` (existing) |
| T7 | Test | — | Gemini-3.1-pro-preview | Нет edge-case тестов для пользователя `IDE_STORAGE_ID` (drop history, image limits) | `tests/` (missing) |
| T8 | Test | — | Gemini-3.1-pro-preview | Нет тестов shutdown-while-tasks-in-flight | `tests/` (missing) |
| T9 | Test | — | Gemini-3.1-pro-preview | Нет edge-case тестов согласования версии протокола | `tests/` (missing) |
| T10 | Test | — | MiniMax-M3 | Нет round-trip / synthetic-stdio tests | `tests/` (missing) |
| T11 | Test | — | MiniMax-M3 | Нет тестов пути генерации изображений (поймали бы C1) | `tests/` (missing) |
| T12 | Test | — | MiniMax-M3 | Нет observability / cancellation-ack tests (поймали бы C2) | `tests/` (missing) |

---

*Синтезировано из трёх независимых обзоров. Исходные отчёты:*

- *Обзор транспорта/протокола (MiniMax/MiniMax-M3) — `chibi/_ide_transport_review_report.md`*
- *Обзор интеграционного слоя (kimi-k2.7-code)*
- *Обзор инфраструктуры/тестов (Gemini 3.1-pro-preview)*

*Синтез выполнен MiniMax/MiniMax-M3. Новая аналитика не добавлялась; весь контент получен из исходных отчётов.*