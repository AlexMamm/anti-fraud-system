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

1. __Анализ требований и сбор данных__:
- Конкретная: Собрать и проанализировать требования к функциям и производительности антифрод-системы, а также существующие данные о транзакциях.
- Измеримая: Определить количество и типы необходимых данных для обучения модели.
- Достижимая: Убедиться, что данные доступны и могут быть предоставлены в соответствии с требованиями конфиденциальности.
- Релевантная: Обеспечение достаточного количества качественных данных является ключевым для разработки эффективной модели.
- Ограниченная во времени: 1 месяц на сбор данных и анализ требований.

2. __Разработка и тестирование прототипа модели__:
- Конкретная: Создать прототип модели машинного обучения для идентификации мошеннических транзакций.
- Измеримая: Достичь точности определения мошеннических операций c метриками Recall не менее 0.95 и FPR не более 0.05.
- Достижимая: Использовать современные алгоритмы машинного обучения и накопленные данные для достижения цели.
- Релевантная: Прямо влияет на способность системы эффективно предотвращать мошенничество.
- Ограниченная во времени: 2 месяца на разработку и 1 месяц на тестирование на реальном потоке данных.

3. __Интеграция системы в существующую IT-инфраструктуру__:
- Конкретная: Внедрить антифрод-систему в IT-инфраструктуру банка без существенных нарушений в работе существующих систем.
- Измеримая: Обеспечить стабильную работу системы 24/7 с обработкой до 400 транзакций в секунду в пиковые периоды.
- Достижимая: Разработать масштабируемую архитектуру и использовать облачные технологии для гибкости и масштабирования.
- Релевантная: Ключевой этап для реализации возможности системы работать в реальных условиях.
- Ограниченная во времени: 1 месяц на интеграцию и тестирование.

4. __Обеспечение соответствия требованиям безопасности и конфиденциальности__:
- Конкретная: Разработать меры по обеспечению безопасности хранения и обработки данных в антифрод-системе.
- Измеримая: Соответствие требованиям местного законодательства.
- Достижимая: Применение лучших практик шифрования данных и контроля доступа.
- Релевантная: Критически важно для защиты конфиденциальной информации клиентов.
- Ограниченная во времени: 1 месяц на разработку и внедрение мер безопасности.

5. __Обучение и адаптация персонала__:
- Конкретная: Разработать и провести программу обучения для сотрудников банка, работающих с антифрод-системой, включая операционный персонал, аналитиков данных и службу безопасности.
- Измеримая: Обучить не менее 90% целевой аудитории использованию системы, а также методам распознавания и реагирования на потенциальные мошеннические операции в течение первого квартала после внедрения системы.
- Достижимая: Создать комплексные учебные материалы, включая руководства, видеоуроки и вебинары, с учетом различных уровней квалификации и специализаций сотрудников.
- Релевантная: Ключевой элемент для эффективного использования антифрод-системы и повышения общего уровня безопасности финансовых операций в банке.
- Ограниченная во времени: 2 месяца на разработку учебных материалов и проведение обучающих сессий с последующей оценкой эффективности обучения.

Эти задачи создают основу для детального плана реализации проекта по созданию антифрод-системы, обеспечивая чёткую последовательность действий и временные рамки для каждого этапа работы.

### Настройка облачной инфраструктуры
- Создано объектное хранилище s3 и скопированы данные для обучения: s3://amamylov-mlops
![image](https://github.com/AlexMamm/anti-fraud-system/assets/107932243/f8e194b4-4ed9-4a85-8d91-09e5444f3f6b)
![image](https://github.com/AlexMamm/anti-fraud-system/assets/107932243/c4561491-dcad-4cea-9070-dd75d3c86cd9)

- Создан spark-кластер согласно конфигурации, переданной от Заказчика. 
![image](https://github.com/AlexMamm/anti-fraud-system/assets/107932243/fc56286e-c8d2-44f1-adb1-3149e22a9b84)
![image](https://github.com/AlexMamm/anti-fraud-system/assets/107932243/ab638e0a-12b6-467c-95dc-a7a5742edb4d)
- Настроено ssh-соединение с мастернодой. Скопированы данные для обучения в распределенную систему HDFS.
![image](https://github.com/AlexMamm/anti-fraud-system/assets/107932243/71040994-1992-4fab-b105-c62af4853100)
![image](https://github.com/AlexMamm/anti-fraud-system/assets/107932243/53383e71-e7a5-4acd-9b97-542c38c1905c)

### Препроцессинг датасетов в Apache Spark
- Произведен разведочный анализ данных, и на его основании выполнен препроцессинг датасетов. Ознакомиться с результатами можно в файле preprocessing.ipynb
- Итоговые parquet-файлы сохранены в собственное хранилище s3
![Screenshot from 2024-05-14 14-52-58](https://github.com/AlexMamm/anti-fraud-system/assets/107932243/c05660f1-1521-4d66-b142-8ad07ecd3b30)

### Выполнение препроцессинга по расписанию в Airflow
- Поднят и настроен Apache Airflow в облачной инфраструктуре
- Реализован DAG для развертывания DataProc кластера и выполнения pyspark-cкрипта с препроцессингом на новых данных в Object Storage S3
![Screenshot from 2024-06-04 06-24-04](https://github.com/AlexMamm/anti-fraud-system/assets/107932243/6841c691-c393-49df-b979-aed9fe3796b7)

### Обучение модели Fraud Detection, логирование метрик и артефактов в MlFlow
- Подготовлена инфраструктура для взаимодействия компонент:
    1. Развернут PostgreSQL для взаимодействия с MlFlow
![Screenshot from 2024-06-24 23-11-33](https://github.com/AlexMamm/anti-fraud-system/assets/107932243/1e460cce-300f-477f-ad54-01ccd211ba44)

    2. Развернута машина и настроен MlFlow (настроен сервисный аккаунт, создано виртуальное окружение, запуск MlFlow завернут в сервис, добавлены доступы на S3)
![Screenshot from 2024-06-24 23-10-48](https://github.com/AlexMamm/anti-fraud-system/assets/107932243/b9f3c925-57b7-494f-b8d1-47a9b8d6b84d)

    3. Поднят и настроен Airflow (установка mlflow через properties в pyspark таске)
 
- Реализован класс для препроцессинга датасетов, полученных из S3, и обучения модели с сохранением артефактов в MlFlow.
![Screenshot from 2024-06-26 23-51-11](https://github.com/AlexMamm/anti-fraud-system/assets/107932243/12db597b-02f9-44f9-bf8a-e47f9119335c)
![Screenshot from 2024-06-26 23-51-36](https://github.com/AlexMamm/anti-fraud-system/assets/107932243/e5878c0e-a7b1-487e-982d-e1a21db30eff)
![Screenshot from 2024-06-26 23-54-25](https://github.com/AlexMamm/anti-fraud-system/assets/107932243/9d8f24ea-93f3-460f-9b63-7beddd225abd)

- Настроено регулярное переобучение модели на новых данных в AirFlow

### Автоматическая валидация моделей
- Реализован новый фича инжениринг для модели случайного леса. Из даты были извлечены дни недели и часы, в которые была произведена транзакция. Посчитаны средние суммы транзакий по id клиента и id терминала, посчитаны отклонения текущей транзакии от этих средних.
![image](https://github.com/AlexMamm/anti-fraud-system/assets/107932243/b99d28ca-8a80-434c-a34c-604e55392223)
- Таким образом мы получаем две модели на разных признаках и стоит вопрос как автоматически выбрать модель, которая перформит лучшее качество.
![image](https://github.com/AlexMamm/anti-fraud-system/assets/107932243/1dc2a1b0-226f-4415-907b-df6881229c8d)
- Для этой цели был реализован a/b тест. Строим случайные бутстрап выборки по тестовым выборкам моделей для более точной оценки. Вычисляются метрики precision, recall, f1, auc и на выходе получаем структуру данных словарь со значениями метрик старой и новой модели. Также вычисляем границы доверительного интервала. Параметр альфа - вероятность ошибки первого рода и размер бутстрап выборок можно задавать через параметры в реализованный метод класса.
![untitled(1)(1)](https://github.com/AlexMamm/anti-fraud-system/assets/107932243/243b08a6-a812-47df-9b8e-5a91f5c2f363)
- На a/b тесте проверяем гипотезу, есть ли статистически разница между выборочным средним по каждой из выборок по метрикам, или нет. Если получаем p-value, меньшее альфа, то статистическое различие есть и отклоняем нулевую гипотезу, иначе - принимаем. Результат проведенного теста складываем как артефакты в Mlflow. Там можем убедиться есть ли разница между моделями. Как видим, новый фича инжениринг сильно увеличил метрики и можно смело говорить, что данная модель лучше.
![untitled(2)](https://github.com/AlexMamm/anti-fraud-system/assets/107932243/e477c2c9-d46e-4131-9b04-8b534ce6d1e0)
- Настроено регулярное переобучение моделей и проведение a/b теста в Airflow

### Асинхронный инференс модели на потоке
- Запущена система Apache Kafka на отдельной виртуальной машине Yandex cloud. Настроена группа безопаности для доступа по портам.
![Screenshot from 2024-07-09 11-09-55](https://github.com/AlexMamm/anti-fraud-system/assets/107932243/e2685a91-5f0a-45ae-9b75-1f010dd48bb7)
- Настроена общая схема взаимодействия сервисов для инференса в асинхронном режиме. Написан скрипт, который реализует функции Producer - отправляет сообщение в Kafka, развернут на локальной машине. Написан Pyspark скрипт, который поднимается по расписанию в Airflow. Выполняет функцию Consumer и забирает сообщение из Kafka, получает модель из S3 по run id лучшей модели в Mlflow, делает предикт. Скрипт настроен на работу по таймеру, который передается параметром, и после завершения сохраняет результат в S3. Шаги с получением данных, препроцессингом и тренировкой модели были реализованы ранее. Таким образом получаем схему, в которой по расписанию происходит тренировка и валидация моделей с сохранением метрик в Mlflow и артефактов в S3. Далее по расписанию поднимается скрипт, который забирает сообщения и выполняет инференс модели, складывая результат в S3.
![image](https://github.com/AlexMamm/anti-fraud-system/assets/107932243/56a799b7-bddd-47c7-8037-d875909f2074)
- Пример работы Producer c отправкой сообщений 
![Screenshot from 2024-07-10 01-35-02](https://github.com/AlexMamm/anti-fraud-system/assets/107932243/95b39694-3a06-46fd-803e-c46b1e6257ba)
- Пример работы Consumer с получением сообщений, инференсом модели, сохранением результата в S3
![Screenshot from 2024-07-10 01-40-32](https://github.com/AlexMamm/anti-fraud-system/assets/107932243/9229920c-9ec2-46b3-9735-746b51720609)
![Screenshot from 2024-07-10 01-37-24](https://github.com/AlexMamm/anti-fraud-system/assets/107932243/72b7f76f-5e87-4cc2-a2fb-4c44cb69715b)
- В подсети реализованной схемы был поднят один брокер сообщений и один Consumer. За 10 минут было отправлено 6690 сообщений и обработано 4235 сообщений. Скороть работы модели на потоке составляет 7 транзакций в секунду. Если интенсивность отправки сообщений выше 7 транзакций в секунду, то начинает накапливаться очередь.

### REST-сервис в k8s
- Написан веб-сервис для предсказания банковских транзакций. Реализованы эндпоинта для инференса, и хэлсчеки для проверки в кластере Kubernetes. Написан Dockerfile, получен образ и успешно запуллен в DockerHub
![Screenshot from 2024-07-30 11-37-31](https://github.com/user-attachments/assets/ab1ca854-14ed-41c9-b077-2cf5d4d46f2f)
- Создал кластер из трех рабочих и мастер нодой. Написан deployment манифест для развертывания приложения. Как видно на скриншоте, сервис в кластере работает и отвечает на запросы.
![Screenshot from 2024-07-29 22-34-32](https://github.com/user-attachments/assets/fcc7b59f-943c-4b50-b775-d4141cffcc5c)
![Screenshot from 2024-07-30 12-16-37](https://github.com/user-attachments/assets/7d5c8059-5564-4c1f-b2bf-e424ad525058)
- Поднянт nginx-ingress контроллер, настроены службы ClusterIP, Ingress для проксирования трафика с машины в кластер.
![Screenshot from 2024-07-30 15-49-27](https://github.com/user-attachments/assets/9d102e49-c28e-44e5-b30d-3fb75a520418)
![Screenshot from 2024-07-30 13-04-25](https://github.com/user-attachments/assets/a295c1b0-e53b-4be6-abed-ef5d5d3cce2b)
- Сервис развернут и протестирован на данных с транзакциями
![Screenshot from 2024-07-30 21-16-17](https://github.com/user-attachments/assets/4c655dd0-0e58-4060-8218-253daaca6296)
![Screenshot from 2024-07-30 21-27-56](https://github.com/user-attachments/assets/51d6e23b-bad2-49c4-aea7-9dad565908da)
![Screenshot from 2024-07-30 21-21-48](https://github.com/user-attachments/assets/751c6024-8e23-4c0e-8b88-1c7f7e77da7c)

### Алертинг
- Добавлен эндпоинт с вычислением метрик количества легитимных и вредоносных транзакций. Пересобран образ и запушен в хаб.
- Настроен кластер, поднят сервисы prometheus и grafana через helm, добавлены необходимые ресурсы для взаимодействия компонент в кластере. Настроен ServiceMonitor для опроса сервиса по эндпоинту с метриками.
![Screenshot from 2024-08-06 22-01-48](https://github.com/user-attachments/assets/4ec3494f-8c36-4a07-8b6e-8427d7385bc7)
- Настроена grafana для отрисовки дашборда и фиксации алерта.
- ![Screenshot from 2024-08-06 22-02-09](https://github.com/user-attachments/assets/1f3f0567-5610-4d8f-bdf3-4c0ef5ce0711)
- Настроен телеграм бот и канал для получения алертов.
![Screenshot from 2024-08-06 22-02-39](https://github.com/user-attachments/assets/6c6556ad-e689-4579-978b-740754dbc902)





  


 

