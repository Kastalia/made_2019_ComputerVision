**Решение соревнования:** https://www.kaggle.com/c/made-thousand-facial-landmarks

**Использование:**
1. Конвертирование данных: python convertData.py --data "path"(опц. флаг --part="0.1" - конвертация части выборки). Сохраняются сконвертированные данные в директорию "path".
2. Запуск ноутбука, в котором необходимо указать путь к сконвертированным данным и файлу "test_points.csv".

**Решение:**
1. Конвертация Landmarks в numpy массив для ускорения этапа чтения данных.
2. Обучение ResNeXt50. batch_size=384, lr=1e-3, Adam, ReduceLROnPlateau(factor=0.5,patience=2) на 15 эпохах, с сохранением всех моделей, которые понизили val loss, и информацией о train\val loss.
3. Выбираем непереобученную модель и снова отправляем ResNeXt50 batch_size=384, восстанавливаем lr=1e-3, Adam,  ReduceLROnPlateau(factor=0.1,patience=3) на 30 эпох, с сохранением всех моделей, которые понизили val loss, и информацией о train\val loss.
4. Выбираем непереобученную модель с лучшим loss.

![Image alt](https://github.com/Kastalia/made_2019_ComputerVision/raw/master/01_FacePoints/LB.png)


