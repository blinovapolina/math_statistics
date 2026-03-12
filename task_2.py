import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns

# Настройка стиля
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Данные интервального ряда
intervals = ['29-34', '34-39', '39-44', '44-49', '49-54', '54-59', '59-64', '64-69']
frequencies = [9, 12, 10, 28, 19, 24, 10, 8]
lower_bounds = [29, 34, 39, 44, 49, 54, 59, 64]
upper_bounds = [34, 39, 44, 49, 54, 59, 64, 69]
midpoints = [(l + u) / 2 for l, u in zip(lower_bounds, upper_bounds)]

# Создаем DataFrame
df = pd.DataFrame({
    'interval': intervals,
    'frequency': frequencies,
    'lower': lower_bounds,
    'upper': upper_bounds,
    'midpoint': midpoints
})

# Общий объем выборки
n = sum(frequencies)
print("=" * 70)
print("АНАЛИЗ ИНТЕРВАЛЬНОГО ВАРИАЦИОННОГО РЯДА - ВАРИАНТ 9")
print("=" * 70)
print(f"\nДанные:")
print(df.to_string(index=False))
print(f"\nОбъем выборки: n = {n}")

# ============================================
# 1. ВЫЧИСЛЕНИЕ ХАРАКТЕРИСТИК
# ============================================

# 1.1 Среднее по группированной выборке
weighted_sum = sum(m * f for m, f in zip(midpoints, frequencies))
mean = weighted_sum / n
print(f"\n1. СРЕДНЕЕ ЗНАЧЕНИЕ:")
print(f"   x̄ = {weighted_sum:.2f} / {n} = {mean:.4f}")

# 1.2 Дисперсия и СКО
weighted_sum_sq = sum(m**2 * f for m, f in zip(midpoints, frequencies))
variance = weighted_sum_sq / n - mean**2
std = np.sqrt(variance)
print(f"\n2. ДИСПЕРСИЯ И СКО:")
print(f"   Смещенная дисперсия: s² = {variance:.4f}")
print(f"   СКО: s = {std:.4f}")
print(f"   Интервал [x̄-s, x̄+s] = [{mean-std:.2f}, {mean+std:.2f}]")

# 1.3 Медиана (для интервального ряда)
# Находим интервал, содержащий медиану
cumsum = 0
median_interval_idx = None
median_interval_lower = None
median_interval_upper = None
median_interval_freq = None
prev_cumsum = 0

for i, (freq, lower, upper) in enumerate(zip(frequencies, lower_bounds, upper_bounds)):
    prev_cumsum = cumsum
    cumsum += freq
    if cumsum >= n/2:
        median_interval_idx = i
        median_interval_lower = lower
        median_interval_upper = upper
        median_interval_freq = freq
        break

# Формула для медианы интервального ряда
# Me = x0 + h * (n/2 - S_prev) / f_me
h = median_interval_upper - median_interval_lower
median = median_interval_lower + h * (n/2 - prev_cumsum) / median_interval_freq

print(f"\n3. МЕДИАНА:")
print(f"   Медианный интервал: {df.loc[median_interval_idx, 'interval']}")
print(f"   Me = {median:.4f}")

# ============================================
# 2. АНАЛИЗ ФОРМЫ РАСПРЕДЕЛЕНИЯ
# ============================================

print(f"\n" + "=" * 70)
print("АНАЛИЗ ФОРМЫ РАСПРЕДЕЛЕНИЯ")
print("=" * 70)

# 2.1 Поиск пиков (мод)
# Для интервального ряда мода - интервал с наибольшей частотой
max_freq_idx = np.argmax(frequencies)
mode_interval = df.loc[max_freq_idx, 'interval']
mode_freq = frequencies[max_freq_idx]

# Ищем второй пик (если есть)
frequencies_array = np.array(frequencies)
second_peak_idx = None
second_peak_freq = 0
for i in range(len(frequencies)):
    if i != max_freq_idx and frequencies[i] > second_peak_freq:
        # Проверяем, что это локальный максимум
        left = frequencies[i-1] if i > 0 else 0
        right = frequencies[i+1] if i < len(frequencies)-1 else 0
        if frequencies[i] > left and frequencies[i] > right:
            second_peak_freq = frequencies[i]
            second_peak_idx = i

print(f"\n1. КОЛИЧЕСТВО МОД (ПИКОВ):")
print(f"   Основной пик: интервал {mode_interval}, частота {mode_freq}")
if second_peak_idx is not None:
    print(f"   Второй пик: интервал {df.loc[second_peak_idx, 'interval']}, частота {second_peak_freq}")
    print(f"   Вывод: Наблюдается БИМОДАЛЬНОСТЬ (два пика)")
    
    # Расстояние между пиками
    peak_distance = abs(midpoints[second_peak_idx] - midpoints[max_freq_idx])
    data_range = upper_bounds[-1] - lower_bounds[0]
    print(f"   Расстояние между пиками: {peak_distance:.2f}")
    print(f"   Диапазон данных: {data_range:.2f}")
    
    if peak_distance < data_range * 0.3:
        print(f"   Пики БЛИЗКО (сильное перекрытие) → соответствует варианту 9")
    else:
        print(f"   Пики ДАЛЕКО друг от друга")
else:
    print(f"   Вывод: Распределение УНИМОДАЛЬНОЕ (один пик)")

# 2.2 Оценка симметрии
# Для интервального ряда используем коэффициент Пирсона
# sk = (x̄ - Mo) / s
mode_value = midpoints[max_freq_idx]  # приближенно
pearson_skewness = (mean - mode_value) / std

print(f"\n2. СИММЕТРИЯ (коэффициент Пирсона):")
print(f"   x̄ = {mean:.4f}, Mo ≈ {mode_value:.2f}, s = {std:.4f}")
print(f"   sk = ({mean:.4f} - {mode_value:.2f}) / {std:.4f} = {pearson_skewness:.4f}")

if abs(pearson_skewness) < 0.3:
    print(f"   Вывод: Распределение близко к СИММЕТРИЧНОМУ")
elif pearson_skewness > 0:
    print(f"   Вывод: ПОЛОЖИТЕЛЬНАЯ асимметрия (правосторонняя)")
else:
    print(f"   Вывод: ОТРИЦАТЕЛЬНАЯ асимметрия (левосторонняя)")

# 2.3 Сравнение медианы и среднего
print(f"\n3. СРАВНЕНИЕ МЕДИАНЫ И СРЕДНЕГО:")
print(f"   Me = {median:.4f}, x̄ = {mean:.4f}")
print(f"   Разность: {median - mean:.4f}")

if abs(median - mean) < std * 0.1:
    print(f"   Me ≈ x̄ → подтверждает симметричность")
elif median > mean:
    print(f"   Me > x̄ → подтверждает ОТРИЦАТЕЛЬНУЮ асимметрию")
else:
    print(f"   Me < x̄ → подтверждает ПОЛОЖИТЕЛЬНУЮ асимметрию")

# ============================================
# 3. ВИЗУАЛИЗАЦИЯ
# ============================================

fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Анализ интервального вариационного ряда - Вариант 9', fontsize=16, fontweight='bold')

# 3.1 Гистограмма
ax1 = axes[0, 0]
bars = ax1.bar(range(len(intervals)), frequencies, width=0.8, color='skyblue', 
               edgecolor='navy', alpha=0.7, tick_label=intervals)
ax1.axhline(0, color='black', linewidth=0.5)
ax1.set_xlabel('Интервалы')
ax1.set_ylabel('Частота')
ax1.set_title('Гистограмма распределения')
ax1.grid(True, alpha=0.3, axis='y')

# Отмечаем пики
bars[max_freq_idx].set_color('red')
bars[max_freq_idx].set_alpha(0.9)
if second_peak_idx is not None:
    bars[second_peak_idx].set_color('orange')
    bars[second_peak_idx].set_alpha(0.9)

# Отмечаем среднее и медиану
ax1.axvline(x=np.interp(mean, midpoints, range(len(midpoints))), 
            color='green', linestyle='--', linewidth=2, label=f'Среднее ({mean:.2f})')
ax1.axvline(x=np.interp(median, midpoints, range(len(midpoints))), 
            color='purple', linestyle=':', linewidth=2, label=f'Медиана ({median:.2f})')
ax1.legend()

# Поворачиваем подписи для читаемости
ax1.set_xticklabels(intervals, rotation=45)

# 3.2 Кумулята (накопленные частоты)
ax2 = axes[0, 1]
cumulative = np.cumsum(frequencies)
ax2.plot(range(len(intervals)), cumulative, 'bo-', linewidth=2, markersize=8)
ax2.axhline(n/2, color='red', linestyle='--', alpha=0.7, label=f'n/2 = {n/2}')
ax2.axvline(x=np.interp(median, midpoints, range(len(midpoints))), 
            color='purple', linestyle=':', alpha=0.7, label=f'Медиана')
ax2.set_xlabel('Интервалы')
ax2.set_ylabel('Накопленная частота')
ax2.set_title('Кумулята (накопленные частоты)')
ax2.grid(True, alpha=0.3)
ax2.legend()
ax2.set_xticks(range(len(intervals)))
ax2.set_xticklabels(intervals, rotation=45)

# 3.3 Полигон частот
ax3 = axes[1, 0]
ax3.plot(midpoints, frequencies, 'ro-', linewidth=2, markersize=8, markerfacecolor='red')
ax3.fill_between(midpoints, 0, frequencies, alpha=0.3, color='skyblue')
ax3.axvline(mean, color='green', linestyle='--', linewidth=2, label=f'Среднее ({mean:.2f})')
ax3.axvline(median, color='purple', linestyle=':', linewidth=2, label=f'Медиана ({median:.2f})')
ax3.set_xlabel('Середины интервалов')
ax3.set_ylabel('Частота')
ax3.set_title('Полигон частот')
ax3.grid(True, alpha=0.3)
ax3.legend()

# 3.4 Ящик с усами (аппроксимация)
ax4 = axes[1, 1]
# Генерируем приближенные данные для box plot
approx_data = []
for m, f in zip(midpoints, frequencies):
    approx_data.extend([m] * f)
approx_data = np.array(approx_data)

bp = ax4.boxplot(approx_data, vert=False, patch_artist=True, showmeans=True)
bp['boxes'][0].set_facecolor('lightblue')
bp['medians'][0].set_color('green')
bp['means'][0].set_marker('o')
bp['means'][0].set_markerfacecolor('red')
bp['means'][0].set_markeredgecolor('red')
ax4.set_xlabel('Значение признака')
ax4.set_title('Ящик с усами (аппроксимация)')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ============================================
# 4. ИТОГОВЫЕ ОТВЕТЫ
# ============================================

print(f"\n" + "=" * 70)
print("ИТОГОВЫЕ ОТВЕТЫ НА ВОПРОСЫ")
print("=" * 70)

print(f"""
1. Количество мод (пиков): 
   {2 if second_peak_idx is not None else 1} пика
   Основной пик: интервал {mode_interval} (частота {mode_freq})
   {f'Второй пик: интервал {df.loc[second_peak_idx, "interval"]} (частота {second_peak_freq})' if second_peak_idx is not None else 'Второй пик не выражен'}

2. Симметрия распределения:
   Коэффициент Пирсона = {pearson_skewness:.4f} → {'положительная' if pearson_skewness > 0.3 else 'отрицательная' if pearson_skewness < -0.3 else 'симметричное'}

3. Сравнение медианы и среднего:
   Me ({median:.4f}) {'>' if median > mean else '<' if median < mean else '='} x̄ ({mean:.4f})
   {'Подтверждает' if (median > mean and pearson_skewness < 0) or (median < mean and pearson_skewness > 0) else 'Не подтверждает'} вывод о симметрии

4. Бимодальность и выбросы:
   - Бимодальность: {'ПРИСУТСТВУЕТ (два пика)' if second_peak_idx is not None else 'НЕ выражена'}
   - Выбросы: по интервальному ряду не определяются, но крайние интервалы (29-34 и 64-69) имеют малые частоты

5. Рекомендуемая мера центра:
   {'Среднее' if abs(pearson_skewness) < 0.3 else 'Медиана'} = {mean:.4f if abs(pearson_skewness) < 0.3 else median:.4f}

6. Предполагаемый закон распределения:
   По таблице вариантов - Вариант 9: Бимодальное распределение с близкими пиками
   Фактически: {'СООТВЕТСТВУЕТ (два близких пика)' if second_peak_idx is not None and abs(midpoints[second_peak_idx] - midpoints[max_freq_idx]) < (upper_bounds[-1]-lower_bounds[0])*0.3 else 'НЕ СООТВЕТСТВУЕТ'}
""")