**Решение соревнования:** https://www.kaggle.com/c/car-plates-ocr-made

**Решение:**
1. MaskRCNN с предобученной частью fasterrcnn_resnet50_fpn и с головами FastRCNNPredictor, MaskRCNNPredictor
2. Выравнивание автономера с помощью cv2 "выправление перспективы"
3. GRU для распознования

**Не зашло:**
1. UMAP; smp.UMAP и smp.FPN с backbone resnext55. Возможно не получилось нормально отладить.
2. DiceLoss был неудобен 



