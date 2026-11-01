# Права и атрибуция полной локальной копии

Полная база содержит все семь источников. Дамп предназначен для локальной проверки; его техническая полнота не подтверждает право передачи всего содержимого коллегам. Файлы не отправлялись. Пароли, клиентские исходные идентификаторы отзывов и оригиналы переписки в пакет не включены.

| Компонент | Происхождение и условия | Передача |
| --- | --- | --- |
| ЦБ | Банк России, официальные обзоры https://www.cbr.ru/statistics/ib/ ; сохранить ссылку, периоды, единицы и авторство | Исследовательское использование с атрибуцией |
| PaySim | E. A. Lopez-Rojas, https://www.kaggle.com/datasets/ealaxi/paysim1 , версия 2; CC BY-SA 4.0 https://creativecommons.org/licenses/by-sa/4.0/ | Соблюдать атрибуцию, ShareAlike и уведомления о преобразованиях |
| Credit Card | ULB Machine Learning Group / Worldline, https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud , версия 3; ODbL 1.0 https://opendatacommons.org/licenses/odbl/1-0/ и DbCL 1.0 https://opendatacommons.org/licenses/dbcl/1-0/ | Соблюдать условия базы, содержимого, производных баз и уведомлений; это не CC0 |
| FDH | Независимая реализация и собственная генерация по методологии Yann-Aël Le Borgne, Wissam Siblini, Bertrand Lebichot, Gianluca Bontempi, ULB, 2022: https://github.com/Fraud-Detection-Handbook/fraud-detection-handbook | Исходные данные авторов не скачивались. Собственная лицензия результатов пока not_declared; ранее записанный gate not_approved_pending_dataset_license не изменён |
| IEEE-CIS | Vesta / IEEE CIS, https://www.kaggle.com/competitions/ieee-fraud-detection/rules ; официальные файлы лично получены пользователем | B.7.A разрешает некоммерческое исследование, B.7.B ограничивает доступ участниками, принявшими правила; B.6.B и отдельное правило команд ограничивают обмен. Статус конкретных получателей не подтверждён. Локальный Kaggle-доступ не разрешает передачу всей команде |
| Яндекс | Geo Reviews Dataset 2023, © 2023 YANDEX LLC, https://github.com/yandex/geo-reviews-dataset-2023 ; MIT; фактическая Parquet-копия https://huggingface.co/datasets/d0rj/geo-reviews-dataset-2023 | Сохранить текст MIT и copyright. Идентичность официальному бинарному файлу не доказана |
| Т-Банк | T-Bank customer reviews, DANO 2025; автор Т-Банк; первичная публикация https://dano.hse.ru/data2025 ; фактическое зеркало https://github.com/yahyyyk/T-Bank_Review_Analysis | Пользователь сообщил положительный ответ info@danocontest.ru от 30.10.2026 на запрос о некоммерческом исследовании, PostgreSQL/NLP и частной передаче проектной команде. Условия — ссылка на первоисточник и автор Т-Банк. Оригинал письма не предоставлен; локально не проверен, открытая лицензия не заявляется |

PaySim и FDH — экспериментальная синтетика, ЦБ — официальная агрегированная статистика; эти типы данных нельзя смешивать. Преобразования PaySim/Credit Card: типизация, snake_case, lineage, сохранение исходных значений и меток; изменения документированы в словаре. Для всей составной базы единая лицензия не объявляется.

Повторная проверка FDH: README первоисточника разграничивает GPLv3 для notebook-кода и CC BY-SA 4.0 для текста/иллюстраций. Эти лицензии не становятся автоматически лицензией результатов собственного независимого симулятора. GNU FAQ разъясняет отдельность вывода программы: https://www.gnu.org/licenses/gpl-faq.en.html#GPLOutput . Не найдено стороннего гранта, который требовалось бы присвоить собственным числовым симуляциям; формальные условия передачи собственных результатов ещё не зафиксированы, исторические metadata сохранены без выдуманной лицензии. README upstream simulated-data-raw не содержит лицензии: https://github.com/Fraud-Detection-Handbook/simulated-data-raw . Его записи в проект не импортировались.

Полный дамп держать локально до подтверждения IEEE-CIS для конкретных получателей, проверки оригинала переписки DANO и фиксации условий собственного FDH-набора. Это не ограничивает завершённую техническую проверку локального бэкапа.

## MIT — Geo Reviews Dataset 2023

Copyright (c) 2023 YANDEX LLC

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
