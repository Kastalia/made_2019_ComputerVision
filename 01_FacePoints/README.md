**Решение соревнования:** https://www.kaggle.com/c/made-thousand-facial-landmarks

**Использование:**
1. Конвертирование данных: python convertData.py --data "path"(опц. флаг --part="0.1" - конвертация части выборки). Сохраняются сконвертированные данные в директорию "path".
2. Запуск ноутбука, в котором необходимо указать путь к сконвертированным данным pathData и файлу "test_points.csv" testPoints. Сохраняются удачные модели с обоих этапов обучения и при обнаружении переобучения, следует поставить более раннюю модель в соответствующем поле pathModel. 

**Решение:**
1. Конвертация Landmarks в numpy массив для ускорения этапа чтения данных.
2. Обучение ResNeXt50. batch_size=384, lr=1e-3, Adam, ReduceLROnPlateau(factor=0.4,patience=2) на 15 эпохах, с сохранением всех моделей, которые понизили val loss, и информацией о train\val loss.
3. Выбираем непереобученную модель, восстанавливаем значение lr=1e-3, меняем  ReduceLROnPlateau(factor=0.1,patience=3) продолжаем обучение на 30 эпох, с сохранением всех моделей, которые понизили val loss, и информацией о train\val loss.
4. Выбираем непереобученную модель с лучшим loss.

![Image alt](https://github.com/Kastalia/made_2019_ComputerVision/raw/master/01_FacePoints/LB.png)


