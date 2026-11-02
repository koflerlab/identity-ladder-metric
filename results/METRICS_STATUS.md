# Текущие метрики и результаты

Обновлено (UTC): 2026-11-02T07:22:22.367472+00:00. Генерируется из сохранённых агрегатов; исходные результаты не перезаписываются.

## Что сейчас проверяется

| Блок | Метрики | Данные | Текущий статус |
|---|---|---|---|
| BraTS 2D | Все 9 выходов: RMSE, MSE, MAE, MSLE, PSNR, PSNR_eps, PSNR_01, PSNR_01_eps, SSIM | GRAPE, UPOL, UTIRIS; центральная область 64×64 из 224×224 | Посчитано; описательные результаты |
| Признаки 2D | Пиксели P, DINOv2 D, контроль яркости N; RETFound R и сосудистые Dice/компоненты/ветвления для GRAPE | Отдельный протокол и представление | Посчитано отдельно; см. ниже |
| QTIM MRI | 9 BraTS-метрик; сегментационные проверки отдельным этапом | SyN + P90 | Текущих проверенных чисел нет; прежние affine-результаты не используем |
| CARE, итерация 2 | SSIM, PSNR; NSD крупных/малых органов; clDice кишечника/сосудов | 36 тестовых КТ; результаты авторов | протокол итерации 2 зафиксирован; шесть метрик ожидают пересчёта |
| H19 на CARE | Dice, NSD 2 мм, clDice на mean/shrinkage/wrong-detail | Изображения через фиксированный сегментатор | Ещё не выполнено |

c = доля согласованных по условиям посторонних, получивших худшую оценку, с половиной веса совпадений; 0.5 — типичный посторонний. Это калибровка, а не ещё одна метрика.

## Все девять BraTS-выходов: собственное изображение / population mean

Здесь указаны средние c. GRAPE: другая дата, та же камера, <1 года; UPOL/UTIRIS: повтор в той же сессии. Mean в исходном агрегате объединён по same-eye строкам; это НЕ парное сравнение с отдельной строкой GRAPE.

| Метрика | Лучше в исходной шкале | GRAPE своё / mean | UPOL своё / mean | UTIRIS своё / mean | QTIM |
|---|---|---|---|---|---|
| RMSE | ↓ | 0.650 / 0.654 | 0.814 / 0.918 | 0.689 / 0.884 | не готово |
| MSE | ↓ | 0.650 / 0.654 | 0.814 / 0.918 | 0.689 / 0.884 | не готово |
| MAE | ↓ | 0.626 / 0.631 | 0.888 / 0.831 | 0.724 / 0.775 | не готово |
| MSLE | ↓ | 0.652 / 0.652 | 0.836 / 0.901 | 0.689 / 0.879 | не готово |
| PSNR | ↑ | 0.641 / 0.630 | 0.810 / 0.917 | 0.677 / 0.869 | не готово |
| PSNR_eps | ↑ | 0.650 / 0.654 | 0.814 / 0.918 | 0.689 / 0.884 | не готово |
| PSNR_01 | ↑ | 0.650 / 0.654 | 0.814 / 0.918 | 0.689 / 0.884 | не готово |
| PSNR_01_eps | ↑ | 0.650 / 0.654 | 0.814 / 0.918 | 0.689 / 0.884 | не готово |
| SSIM | ↑ | 0.790 / 0.943 | 0.808 / 0.914 | 0.684 / 0.811 | не готово |

RMSE/MSE и варианты PSNR могут давать одинаковые ранги в одном случае: девять выходов не означают девять независимых свидетельств.
Старый текст «единственное исключение — MAE на UPOL» неполон: plain PSNR на GRAPE также выше у своего изображения (0.641), чем у объединённого mean (0.630). Это описательное сравнение разных пулов, без парного теста и без вывода о значимости.

## Выборки и все ступени из таблицы на скриншоте

| Набор | Ступень | Пар | Людей | c MSE | c SSIM |
|---|---|---|---|---|---|
| GRAPE | other visit, same camera, 1 year or more | 71 | 37 | 0.673 | 0.757 |
| GRAPE | population mean | 318 | 106 | 0.654 | 0.943 |
| GRAPE | other visit, other camera | 152 | 83 | 0.572 | 0.665 |
| GRAPE | mirrored other side (fellow eye flipped) | 119 | 119 | 0.764 | 0.695 |
| GRAPE | other visit, same camera, under 1 year | 95 | 53 | 0.650 | 0.790 |
| UPOL | same-sitting repeat | 128 | 64 | 0.814 | 0.808 |
| UPOL | population mean | 128 | 64 | 0.918 | 0.914 |
| UPOL | mirrored other side (fellow eye flipped) | 64 | 64 | 0.727 | 0.743 |
| UTIRIS | same-sitting repeat | 156 | 78 | 0.689 | 0.684 |
| UTIRIS | population mean | 156 | 78 | 0.884 | 0.811 |
| UTIRIS | mirrored other side (fellow eye flipped) | 78 | 78 | 0.602 | 0.611 |

На скриншоте нет: семи остальных выходов пакета, исходных значений, 95% интервалов c, размеров выборок, строки посторонних, shrinkage-пути, других представлений, семейных сравнений и готовых QTIM-значений. Пустая клетка не равна нулю.

## CARE: каждая метрика отдельно

Воспроизводится агрегация опубликованных оценок, а не генерация КТ или запуск сегментатора. Сравнение — медиана и два квартиля с таблицей 2, допуск округления 0.050001.

| Метрика | Совпало | Не совпало | Нет покрытия | Нет конечной оценки | Статус |
|---|---|---|---|---|---|
| SSIM ↑ | — | — | — | — | ожидается итерация 2 |
| PSNR ↑ | — | — | — | — | ожидается итерация 2 |
| NSD крупных органов ↑ | — | — | — | — | ожидается итерация 2 |
| NSD малых органов ↑ | — | — | — | — | ожидается итерация 2 |
| clDice кишечника ↑ | — | — | — | — | ожидается итерация 2 |
| clDice сосудов ↑ | — | — | — | — | ожидается итерация 2 |

[CARE: текущие численные значения и сравнение с публикацией](care_iteration2_report.md). История: [13 случаев](care_reproduction_report.md), [36 случаев: прежняя остановка по покрытию](care_reproduction_all_phases_posthoc_report.md). Это не результат проверки H19.

## Другие представления: отдельный 2D-прогон

В каждой ячейке c своего изображения / c среднего для одной и той же ступени. Это другой оператор и область, чем BraTS-таблица выше; числа не склеиваются.

| Представление | GRAPE | UPOL | UTIRIS |
|---|---|---|---|
| P | primary | 0.817 / 0.879 | 0.860 / 0.904 | 0.770 / 0.827 |
| D | primary | 0.972 / 0.010 | 0.994 / 0.300 | 0.935 / 0.009 |
| N | primary | 0.608 / 0.618 | 0.926 / 0.540 | 0.832 / 0.251 |
| R | primary | 0.900 / нет mean | не применялось | не применялось |
| V_dice | primary | 0.866 / 0.464 | не применялось | не применялось |
| V_components | primary | 0.674 / 0.010 | не применялось | не применялось |
| V_branches | primary | 0.649 / 0.369 | не применялось | не применялось |

Все ступени, lambda=0/0.25/0.5/0.75/1, интервалы и чувствительности сохранены в [ladder2d.json](ladder2d.json). Отсутствующая lambda не достраивается.

## Полные исходные шкалы и интервалы: девять BraTS-выходов

c: среднее и 95% bootstrap-интервал с группировкой по людям; исходная шкала: медиана [Q1, Q3]. Источник: [сохранённый агрегат](ladder2d_brats_baseline.json).

| Набор | Ступень | Метрика | Исходная медиана [Q1, Q3] | c [95% интервал] | Пар / людей |
|---|---|---|---|---|---|
| GRAPE | other visit, same camera, 1 year or more | RMSE | 0.0824 [0.05482, 0.1542] | 0.673 [0.595, 0.751] | 71 / 37 |
| GRAPE | other visit, same camera, 1 year or more | MSE | 0.006789 [0.003005, 0.0238] | 0.673 [0.595, 0.751] | 71 / 37 |
| GRAPE | other visit, same camera, 1 year or more | MAE | 0.07569 [0.04689, 0.1424] | 0.645 [0.568, 0.723] | 71 / 37 |
| GRAPE | other visit, same camera, 1 year or more | MSLE | 0.00277 [0.001439, 0.0101] | 0.666 [0.587, 0.746] | 71 / 37 |
| GRAPE | other visit, same camera, 1 year or more | PSNR | 14.97 [10.45, 18.6] | 0.665 [0.588, 0.741] | 71 / 37 |
| GRAPE | other visit, same camera, 1 year or more | PSNR_eps | 14.94 [10.06, 18.51] | 0.673 [0.595, 0.751] | 71 / 37 |
| GRAPE | other visit, same camera, 1 year or more | PSNR_01 | 21.68 [16.24, 25.22] | 0.673 [0.595, 0.751] | 71 / 37 |
| GRAPE | other visit, same camera, 1 year or more | PSNR_01_eps | 21.68 [16.24, 25.22] | 0.673 [0.595, 0.751] | 71 / 37 |
| GRAPE | other visit, same camera, 1 year or more | SSIM | 0.7129 [0.6381, 0.7821] | 0.757 [0.678, 0.826] | 71 / 37 |
| GRAPE | population mean | RMSE | 0.122 [0.07776, 0.2098] | 0.654 [0.626, 0.682] | 318 / 106 |
| GRAPE | population mean | MSE | 0.01489 [0.006046, 0.044] | 0.654 [0.626, 0.682] | 318 / 106 |
| GRAPE | population mean | MAE | 0.1109 [0.06463, 0.1969] | 0.631 [0.603, 0.659] | 318 / 106 |
| GRAPE | population mean | MSLE | 0.006764 [0.002724, 0.01813] | 0.652 [0.624, 0.680] | 318 / 106 |
| GRAPE | population mean | PSNR | 11.81 [8.252, 15.33] | 0.630 [0.600, 0.659] | 318 / 106 |
| GRAPE | population mean | PSNR_eps | 11.35 [7.141, 15.28] | 0.654 [0.626, 0.682] | 318 / 106 |
| GRAPE | population mean | PSNR_01 | 18.27 [13.57, 22.19] | 0.654 [0.626, 0.682] | 318 / 106 |
| GRAPE | population mean | PSNR_01_eps | 18.27 [13.57, 22.19] | 0.654 [0.626, 0.682] | 318 / 106 |
| GRAPE | population mean | SSIM | 0.7187 [0.6399, 0.7837] | 0.943 [0.932, 0.954] | 318 / 106 |
| GRAPE | other visit, other camera | RMSE | 0.1946 [0.1203, 0.2766] | 0.572 [0.512, 0.631] | 152 / 83 |
| GRAPE | other visit, other camera | MSE | 0.03789 [0.01447, 0.07652] | 0.572 [0.512, 0.631] | 152 / 83 |
| GRAPE | other visit, other camera | MAE | 0.1766 [0.09174, 0.2675] | 0.561 [0.501, 0.621] | 152 / 83 |
| GRAPE | other visit, other camera | MSLE | 0.01604 [0.00583, 0.03013] | 0.574 [0.515, 0.634] | 152 / 83 |
| GRAPE | other visit, other camera | PSNR | 8.95 [6.226, 13.28] | 0.560 [0.504, 0.619] | 152 / 83 |
| GRAPE | other visit, other camera | PSNR_eps | 7.715 [4.733, 12.13] | 0.572 [0.512, 0.631] | 152 / 83 |
| GRAPE | other visit, other camera | PSNR_01 | 14.22 [11.16, 18.4] | 0.572 [0.512, 0.631] | 152 / 83 |
| GRAPE | other visit, other camera | PSNR_01_eps | 14.22 [11.16, 18.4] | 0.572 [0.512, 0.631] | 152 / 83 |
| GRAPE | other visit, other camera | SSIM | 0.6167 [0.5185, 0.6984] | 0.665 [0.608, 0.723] | 152 / 83 |
| GRAPE | mirrored other side (fellow eye flipped) | RMSE | 0.08944 [0.05654, 0.1398] | 0.764 [0.714, 0.812] | 119 / 119 |
| GRAPE | mirrored other side (fellow eye flipped) | MSE | 0.008 [0.003197, 0.01955] | 0.764 [0.714, 0.812] | 119 / 119 |
| GRAPE | mirrored other side (fellow eye flipped) | MAE | 0.07198 [0.04524, 0.1239] | 0.760 [0.710, 0.807] | 119 / 119 |
| GRAPE | mirrored other side (fellow eye flipped) | MSLE | 0.003357 [0.001479, 0.008867] | 0.769 [0.719, 0.815] | 119 / 119 |
| GRAPE | mirrored other side (fellow eye flipped) | PSNR | 15.77 [11.81, 18.86] | 0.765 [0.714, 0.814] | 119 / 119 |
| GRAPE | mirrored other side (fellow eye flipped) | PSNR_eps | 15.76 [11.73, 18.85] | 0.764 [0.714, 0.812] | 119 / 119 |
| GRAPE | mirrored other side (fellow eye flipped) | PSNR_01 | 20.97 [17.09, 24.95] | 0.764 [0.714, 0.812] | 119 / 119 |
| GRAPE | mirrored other side (fellow eye flipped) | PSNR_01_eps | 20.97 [17.09, 24.95] | 0.764 [0.714, 0.812] | 119 / 119 |
| GRAPE | mirrored other side (fellow eye flipped) | SSIM | 0.6437 [0.5757, 0.7227] | 0.695 [0.650, 0.740] | 119 / 119 |
| GRAPE | other visit, same camera, under 1 year | RMSE | 0.1163 [0.05765, 0.1711] | 0.650 [0.579, 0.722] | 95 / 53 |
| GRAPE | other visit, same camera, under 1 year | MSE | 0.01354 [0.003323, 0.02927] | 0.650 [0.579, 0.722] | 95 / 53 |
| GRAPE | other visit, same camera, under 1 year | MAE | 0.1024 [0.04745, 0.1576] | 0.626 [0.557, 0.696] | 95 / 53 |
| GRAPE | other visit, same camera, under 1 year | MSLE | 0.005354 [0.001653, 0.01252] | 0.652 [0.582, 0.722] | 95 / 53 |
| GRAPE | other visit, same camera, under 1 year | PSNR | 13.44 [9.859, 17.63] | 0.641 [0.569, 0.712] | 95 / 53 |
| GRAPE | other visit, same camera, under 1 year | PSNR_eps | 13.18 [9.188, 17.42] | 0.650 [0.579, 0.722] | 95 / 53 |
| GRAPE | other visit, same camera, under 1 year | PSNR_01 | 18.69 [15.34, 24.78] | 0.650 [0.579, 0.722] | 95 / 53 |
| GRAPE | other visit, same camera, under 1 year | PSNR_01_eps | 18.69 [15.34, 24.78] | 0.650 [0.579, 0.722] | 95 / 53 |
| GRAPE | other visit, same camera, under 1 year | SSIM | 0.7296 [0.63, 0.7791] | 0.790 [0.729, 0.846] | 95 / 53 |
| GRAPE | stranger (matched) | RMSE | 0.1519 [0.09637, 0.2461] | 0.500 (по построению) | 19178 reference fills; не независимые люди |
| GRAPE | stranger (matched) | MSE | 0.02307 [0.009287, 0.06057] | 0.500 (по построению) | 19178 reference fills; не независимые люди |
| GRAPE | stranger (matched) | MAE | 0.1337 [0.07949, 0.232] | 0.500 (по построению) | 19178 reference fills; не независимые люди |
| GRAPE | stranger (matched) | MSLE | 0.01016 [0.004014, 0.02715] | 0.500 (по построению) | 19178 reference fills; не независимые люди |
| GRAPE | stranger (matched) | PSNR | 10.65 [7.239, 14.37] | 0.500 (по построению) | 19178 reference fills; не независимые люди |
| GRAPE | stranger (matched) | PSNR_eps | 9.746 [5.537, 13.9] | 0.500 (по построению) | 19178 reference fills; не независимые люди |
| GRAPE | stranger (matched) | PSNR_01 | 16.37 [12.18, 20.32] | 0.500 (по построению) | 19178 reference fills; не независимые люди |
| GRAPE | stranger (matched) | PSNR_01_eps | 16.37 [12.18, 20.32] | 0.500 (по построению) | 19178 reference fills; не независимые люди |
| GRAPE | stranger (matched) | SSIM | 0.6106 [0.5253, 0.6775] | 0.500 (по построению) | 19178 reference fills; не независимые люди |
| UPOL | same-sitting repeat | RMSE | 0.1394 [0.1141, 0.1616] | 0.814 [0.770, 0.854] | 128 / 64 |
| UPOL | same-sitting repeat | MSE | 0.01944 [0.01301, 0.02611] | 0.814 [0.770, 0.854] | 128 / 64 |
| UPOL | same-sitting repeat | MAE | 0.07032 [0.05527, 0.09201] | 0.888 [0.853, 0.920] | 128 / 64 |
| UPOL | same-sitting repeat | MSLE | 0.007848 [0.005319, 0.01119] | 0.836 [0.794, 0.875] | 128 / 64 |
| UPOL | same-sitting repeat | PSNR | 15.76 [14.43, 17.26] | 0.810 [0.765, 0.852] | 128 / 64 |
| UPOL | same-sitting repeat | PSNR_eps | 15.76 [14.43, 17.26] | 0.814 [0.770, 0.854] | 128 / 64 |
| UPOL | same-sitting repeat | PSNR_01 | 17.11 [15.83, 18.86] | 0.814 [0.770, 0.854] | 128 / 64 |
| UPOL | same-sitting repeat | PSNR_01_eps | 17.11 [15.83, 18.86] | 0.814 [0.770, 0.854] | 128 / 64 |
| UPOL | same-sitting repeat | SSIM | 0.6461 [0.557, 0.7131] | 0.808 [0.760, 0.854] | 128 / 64 |
| UPOL | population mean | RMSE | 0.1243 [0.1068, 0.1501] | 0.918 [0.902, 0.932] | 128 / 64 |
| UPOL | population mean | MSE | 0.01545 [0.0114, 0.02254] | 0.918 [0.902, 0.932] | 128 / 64 |
| UPOL | population mean | MAE | 0.08671 [0.06956, 0.1108] | 0.831 [0.797, 0.862] | 128 / 64 |
| UPOL | population mean | MSLE | 0.006947 [0.005048, 0.01059] | 0.901 [0.881, 0.919] | 128 / 64 |
| UPOL | population mean | PSNR | 16.56 [15.27, 17.92] | 0.917 [0.902, 0.932] | 128 / 64 |
| UPOL | population mean | PSNR_eps | 16.55 [15.27, 17.92] | 0.918 [0.902, 0.932] | 128 / 64 |
| UPOL | population mean | PSNR_01 | 18.11 [16.47, 19.43] | 0.918 [0.902, 0.932] | 128 / 64 |
| UPOL | population mean | PSNR_01_eps | 18.11 [16.47, 19.43] | 0.918 [0.902, 0.932] | 128 / 64 |
| UPOL | population mean | SSIM | 0.6605 [0.6038, 0.7168] | 0.914 [0.898, 0.928] | 128 / 64 |
| UPOL | mirrored other side (fellow eye flipped) | RMSE | 0.1578 [0.1437, 0.1745] | 0.727 [0.669, 0.784] | 64 / 64 |
| UPOL | mirrored other side (fellow eye flipped) | MSE | 0.0249 [0.02065, 0.03044] | 0.727 [0.669, 0.784] | 64 / 64 |
| UPOL | mirrored other side (fellow eye flipped) | MAE | 0.09241 [0.07636, 0.1026] | 0.826 [0.781, 0.870] | 64 / 64 |
| UPOL | mirrored other side (fellow eye flipped) | MSLE | 0.01066 [0.008527, 0.01309] | 0.758 [0.702, 0.811] | 64 / 64 |
| UPOL | mirrored other side (fellow eye flipped) | PSNR | 14.61 [13.63, 15.84] | 0.722 [0.663, 0.780] | 64 / 64 |
| UPOL | mirrored other side (fellow eye flipped) | PSNR_eps | 14.6 [13.63, 15.84] | 0.727 [0.669, 0.784] | 64 / 64 |
| UPOL | mirrored other side (fellow eye flipped) | PSNR_01 | 16.04 [15.17, 16.85] | 0.727 [0.669, 0.784] | 64 / 64 |
| UPOL | mirrored other side (fellow eye flipped) | PSNR_01_eps | 16.04 [15.17, 16.85] | 0.727 [0.669, 0.784] | 64 / 64 |
| UPOL | mirrored other side (fellow eye flipped) | SSIM | 0.6008 [0.54, 0.6478] | 0.743 [0.685, 0.800] | 64 / 64 |
| UPOL | stranger (matched) | RMSE | 0.1799 [0.1554, 0.207] | 0.500 (по построению) | 6016 reference fills; не независимые люди |
| UPOL | stranger (matched) | MSE | 0.03238 [0.02415, 0.04284] | 0.500 (по построению) | 6016 reference fills; не независимые люди |
| UPOL | stranger (matched) | MAE | 0.12 [0.09782, 0.1473] | 0.500 (по построению) | 6016 reference fills; не независимые люди |
| UPOL | stranger (matched) | MSLE | 0.01476 [0.01067, 0.02016] | 0.500 (по построению) | 6016 reference fills; не независимые люди |
| UPOL | stranger (matched) | PSNR | 13.56 [12.47, 14.79] | 0.500 (по построению) | 6016 reference fills; не независимые люди |
| UPOL | stranger (matched) | PSNR_eps | 13.51 [12.38, 14.75] | 0.500 (по построению) | 6016 reference fills; не независимые люди |
| UPOL | stranger (matched) | PSNR_01 | 14.9 [13.68, 16.17] | 0.500 (по построению) | 6016 reference fills; не независимые люди |
| UPOL | stranger (matched) | PSNR_01_eps | 14.9 [13.68, 16.17] | 0.500 (по построению) | 6016 reference fills; не независимые люди |
| UPOL | stranger (matched) | SSIM | 0.5287 [0.4665, 0.5913] | 0.500 (по построению) | 6016 reference fills; не независимые люди |
| UTIRIS | same-sitting repeat | RMSE | 0.2446 [0.2039, 0.2848] | 0.689 [0.644, 0.729] | 156 / 78 |
| UTIRIS | same-sitting repeat | MSE | 0.05982 [0.04159, 0.08112] | 0.689 [0.644, 0.729] | 156 / 78 |
| UTIRIS | same-sitting repeat | MAE | 0.1735 [0.1424, 0.22] | 0.724 [0.684, 0.762] | 156 / 78 |
| UTIRIS | same-sitting repeat | MSLE | 0.03174 [0.02222, 0.04491] | 0.689 [0.646, 0.730] | 156 / 78 |
| UTIRIS | same-sitting repeat | PSNR | 10.87 [9.474, 12.14] | 0.677 [0.634, 0.715] | 156 / 78 |
| UTIRIS | same-sitting repeat | PSNR_eps | 10.81 [9.216, 12.14] | 0.689 [0.644, 0.729] | 156 / 78 |
| UTIRIS | same-sitting repeat | PSNR_01 | 12.23 [10.91, 13.81] | 0.689 [0.644, 0.729] | 156 / 78 |
| UTIRIS | same-sitting repeat | PSNR_01_eps | 12.23 [10.91, 13.81] | 0.689 [0.644, 0.729] | 156 / 78 |
| UTIRIS | same-sitting repeat | SSIM | 0.4528 [0.3637, 0.5294] | 0.684 [0.640, 0.726] | 156 / 78 |
| UTIRIS | population mean | RMSE | 0.2025 [0.1689, 0.2367] | 0.884 [0.863, 0.903] | 156 / 78 |
| UTIRIS | population mean | MSE | 0.041 [0.02852, 0.05601] | 0.884 [0.863, 0.903] | 156 / 78 |
| UTIRIS | population mean | MAE | 0.1726 [0.1417, 0.2167] | 0.775 [0.746, 0.802] | 156 / 78 |
| UTIRIS | population mean | MSLE | 0.02233 [0.0161, 0.03151] | 0.879 [0.859, 0.899] | 156 / 78 |
| UTIRIS | population mean | PSNR | 12.22 [10.56, 13.3] | 0.869 [0.846, 0.891] | 156 / 78 |
| UTIRIS | population mean | PSNR_eps | 12.22 [10.56, 13.3] | 0.884 [0.863, 0.903] | 156 / 78 |
| UTIRIS | population mean | PSNR_01 | 13.87 [12.52, 15.45] | 0.884 [0.863, 0.903] | 156 / 78 |
| UTIRIS | population mean | PSNR_01_eps | 13.87 [12.52, 15.45] | 0.884 [0.863, 0.903] | 156 / 78 |
| UTIRIS | population mean | SSIM | 0.4904 [0.4237, 0.539] | 0.811 [0.784, 0.838] | 156 / 78 |
| UTIRIS | mirrored other side (fellow eye flipped) | RMSE | 0.2573 [0.1945, 0.3095] | 0.602 [0.533, 0.669] | 78 / 78 |
| UTIRIS | mirrored other side (fellow eye flipped) | MSE | 0.06621 [0.03783, 0.09578] | 0.602 [0.533, 0.669] | 78 / 78 |
| UTIRIS | mirrored other side (fellow eye flipped) | MAE | 0.2033 [0.1378, 0.2309] | 0.621 [0.552, 0.687] | 78 / 78 |
| UTIRIS | mirrored other side (fellow eye flipped) | MSLE | 0.03615 [0.02197, 0.05216] | 0.607 [0.540, 0.672] | 78 / 78 |
| UTIRIS | mirrored other side (fellow eye flipped) | PSNR | 9.947 [8.419, 11.82] | 0.585 [0.516, 0.652] | 78 / 78 |
| UTIRIS | mirrored other side (fellow eye flipped) | PSNR_eps | 9.826 [8.359, 11.59] | 0.602 [0.533, 0.669] | 78 / 78 |
| UTIRIS | mirrored other side (fellow eye flipped) | PSNR_01 | 11.79 [10.19, 14.22] | 0.602 [0.533, 0.669] | 78 / 78 |
| UTIRIS | mirrored other side (fellow eye flipped) | PSNR_01_eps | 11.79 [10.19, 14.22] | 0.602 [0.533, 0.669] | 78 / 78 |
| UTIRIS | mirrored other side (fellow eye flipped) | SSIM | 0.4014 [0.3464, 0.4943] | 0.611 [0.545, 0.674] | 78 / 78 |
| UTIRIS | stranger (matched) | RMSE | 0.2781 [0.2327, 0.3282] | 0.500 (по построению) | 9516 reference fills; не независимые люди |
| UTIRIS | stranger (matched) | MSE | 0.07736 [0.05413, 0.1077] | 0.500 (по построению) | 9516 reference fills; не независимые люди |
| UTIRIS | stranger (matched) | MAE | 0.2215 [0.1756, 0.276] | 0.500 (по построению) | 9516 reference fills; не независимые люди |
| UTIRIS | stranger (matched) | MSLE | 0.04306 [0.02995, 0.05995] | 0.500 (по построению) | 9516 reference fills; не независимые люди |
| UTIRIS | stranger (matched) | PSNR | 9.673 [7.972, 11.22] | 0.500 (по построению) | 9516 reference fills; не независимые люди |
| UTIRIS | stranger (matched) | PSNR_eps | 9.493 [7.592, 11.1] | 0.500 (по построению) | 9516 reference fills; не независимые люди |
| UTIRIS | stranger (matched) | PSNR_01 | 11.11 [9.676, 12.67] | 0.500 (по построению) | 9516 reference fills; не независимые люди |
| UTIRIS | stranger (matched) | PSNR_01_eps | 11.11 [9.676, 12.67] | 0.500 (по построению) | 9516 reference fills; не независимые люди |
| UTIRIS | stranger (matched) | SSIM | 0.3849 [0.3103, 0.4623] | 0.500 (по построению) | 9516 reference fills; не независимые люди |

Синтетические H19, pupil и population-null результаты остаются отдельными экспериментами: [label_robustness_report.md](label_robustness_report.md), [population_null_report.md](population_null_report.md), [pupil_report.md](pupil_report.md). Они не являются реальными КТ/MRI результатами.
