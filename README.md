## Anti-fraud system
Anti-fraud system - система мониторинга и защиты от мошенничества, разработанная для коммерческих банковских учреждений, предоставляющих услуги онлайн-платежей. 

Система представляет собой комплекс моделей машинного обучения и искусственного интеллекта, целью которых является идентификация и предотвращение мошеннических транзакций в реальном времени, тем самым значительно повышая безопасность финансовых операций клиентов и снижая финансовые риски для банка.

### Цели и функциональные требеования для проектирования и реализации антифрод-системы:

1. __Разработка и внедрение модуля машинного обучения__, способного в реальном времени анализировать транзакции на предмет их мошеннического характера, с целью их автоматического отклонения в случае подозрения.
2. __Эффективность системы__. Система должна обеспечивать уровень обнаружения мошеннических операций не хуже, чем у конкурентов (не более двух мошеннических операций на каждые сто транзакций), с целью предотвращения потерь, не превышающих 500 тыс. руб. в месяц.
3. __Производительность и масштабируемость__. Система должна быть способна обрабатывать колеблющийся объем транзакций — от 50 транзакций в секунду в обычные дни до 400 транзакций в секунду в периоды перед праздниками.
4. __Интерпретируемость решений__. Важно, чтобы модель машинного обучения была интерпретируемой, то есть аналитик мог понять почему модель приняла решение о транзакции по значениям выбранных признаков.
5. __Минимизация ложных срабатываний__. Доля невиновно отклоненных транзакций (ложноположительные результаты) не должна превышать 5%, чтобы избежать потери клиентской лояльности и оттока клиентов.
6. __Использование внешних вычислительных ресурсов__. Поскольку компания не готова размещать модуль на собственных серверах, необходимо обеспечить работу системы на внешних платформах, таких как облачные сервисы.
7. __Обеспечение конфиденциальности данных__. При обработке транзакций и обучении модели необходимо гарантировать защиту конфиденциальных данных клиентов, предотвращая любые риски утечки информации.
8. __Бюджет и сроки проекта__. Разработка должна уложиться в бюджет не более 10 млн. руб. (без учета зарплаты специалистов) и предоставить первые результаты в течение трех месяцев для оценки перспективности дальнейшего развития проекта. В случае положительного решения, окончательная реализация проекта должна быть завершена в течение полугода.

### Метрики машинного обучения
Для анти-фрод системы критически важно не иметь пропусков мошеннических транзакий, поэтому метрика Recall должна быть близка к единице. Заказчик делает поправку 5 процентов на количество ложноположительных сработок - то есть ситуаций, когда легитимную транзакцию система посчитает вредоносной. Это не так критично, как пропуск мошеннических транзакий - но тоже важно, чтобы с такой нагрузкой на потоке справлялся отдел информационной безопасности (операторы SOC). Поэтому второй целевой метрикой является FPR (False Positive Rate) не больше 0.05. Так как мы имеем дело с сильным дисбалансом классов (мошеннических транзакций может быть не больше 2 из 100) смотреть на метрику precision будет не очень информативно: ведь она будет мала. При выборе моделей можно также посмотреть метрику F1, которая является гармоническим средним между Precision и Recall.

### Особенности проекта по MISSION Canvas

1. Миссия (Mission): Разработка и внедрение модуля машинного обучения для выявления и предотвращения мошеннических транзакций в реальном времени, чтобы снизить финансовые потери и повысить доверие клиентов к банковским услугам.
2. Интересы (Interests):
    Компания: Снижение количества мошеннических операций, минимизация финансовых потерь, укрепление позиций на рынке, улучшение качества обслуживания клиентов.
    Клиенты: Обеспечение безопасности их транзакций, снижение риска потери денежных средств.
    Разработчики и аналитики данных: Реализация технологически сложного и социально значимого проекта, профессиональный рост и развитие.
3. Сообщества (Social): Вовлечение сообщества профессионалов в области безопасности данных и машинного обучения для обмена опытом, знаниями и лучшими практиками.
4. Сообщения (Messages): Коммуникация успехов и достижений проекта, его значимости для общества и отдельных пользователей, преимуществ перед конкурентами.
5. Инструменты (Instruments):
    Технологии машинного обучения: Алгоритмы классификации, обучение с учителем, аномалий детекция.
    Большие данные и аналитика: Обработка больших объемов данных в реальном времени.
    Облачные вычисления: Использование облачных платформ для размещения и масштабирования системы.

### Определение задач для реализации проекта по принципу S.M.A.R.T.:

1. Анализ требований и сбор данных:
    Конкретная: Собрать и проанализировать требования к функциям и производительности антифрод-системы, а также существующие данные о транзакциях.
    Измеримая: Определить количество и типы необходимых данных для обучения модели.
    Достижимая: Убедиться, что данные доступны и могут быть предоставлены в соответствии с требованиями конфиденциальности.
    Релевантная: Обеспечение достаточного количества качественных данных является ключевым для разработки эффективной модели.
    Ограниченная во времени: 1 месяц на сбор данных и анализ требований.

3. Разработка и тестирование прототипа модели:
  Конкретная: Создать прототип модели машинного обучения для идентификации мошеннических транзакций.
  Измеримая: Достичь точности определения мошеннических операций c метриками Recall не менее 0.95 и FPR не более 0.05.
  Достижимая: Использовать современные алгоритмы машинного обучения и накопленные данные для достижения цели.
  Релевантная: Прямо влияет на способность системы эффективно предотвращать мошенничество.
  Ограниченная во времени: 2 месяца на разработку и 1 месяц на тестирование на реальном потоке данных.

Интеграция системы в существующую IT-инфраструктуру:
  Конкретная: Внедрить антифрод-систему в IT-инфраструктуру банка без существенных нарушений в работе существующих систем.
  Измеримая: Обеспечить стабильную работу системы 24/7 с обработкой до 400 транзакций в секунду в пиковые периоды.
  Достижимая: Разработать масштабируемую архитектуру и использовать облачные технологии для гибкости и масштабирования.
  Релевантная: Ключевой этап для реализации возможности системы работать в реальных условиях.
  Ограниченная во времени: 1 месяц на интеграцию и тестирование.

Обеспечение соответствия требованиям безопасности и конфиденциальности:
  Конкретная: Разработать меры по обеспечению безопасности хранения и обработки данных в антифрод-системе.
  Измеримая: Соответствие требованиям местного законодательства.
  Достижимая: Применение лучших практик шифрования данных и контроля доступа.
  Релевантная: Критически важно для защиты конфиденциальной информации клиентов.
  Ограниченная во времени: 1 месяц на разработку и внедрение мер безопасности.

Обучение и адаптация персонала:
  Конкретная: Разработать и провести программу обучения для сотрудников банка, работающих с антифрод-системой, включая операционный персонал, аналитиков данных и службу безопасности.
  Измеримая: Обучить не менее 90% целевой аудитории использованию системы, а также методам распознавания и реагирования на потенциальные мошеннические операции в течение первого квартала после внедрения системы.
  Достижимая: Создать комплексные учебные материалы, включая руководства, видеоуроки и вебинары, с учетом различных уровней квалификации и специализаций сотрудников.
  Релевантная: Ключевой элемент для эффективного использования антифрод-системы и повышения общего уровня безопасности финансовых операций в банке.
  Ограниченная во времени: 2 месяца на разработку учебных материалов и проведение обучающих сессий с последующей оценкой эффективности обучения.

Эти задачи создают основу для детального плана реализации проекта по созданию антифрод-системы, обеспечивая чёткую последовательность действий и временные рамки для каждого этапа работы.

