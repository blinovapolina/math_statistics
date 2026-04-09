import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

# Загружаем данные
df_X = pd.read_csv('variant_2_sample_X.csv')
df_Y = pd.read_csv('variant_2_sample_Y.csv')

X = df_X['x_i'].values
Y = df_Y['y_j'].values

# Основные статистики
m = len(X)
n = len(Y)
mean_X = np.mean(X)
mean_Y = np.mean(Y)
var_X = np.var(X, ddof=1)
var_Y = np.var(Y, ddof=1)
std_X = np.std(X, ddof=1)
std_Y = np.std(Y, ddof=1)

print(f"Размер выборки X: m = {m}")
print(f"Размер выборки Y: n = {n}")
print(f"Среднее X̄ = {mean_X:.4f}")
print(f"Среднее Ȳ = {mean_Y:.4f}")
print(f"Дисперсия S_X² = {var_X:.4f}")
print(f"Дисперсия S_Y² = {var_Y:.4f}")
print(f"СКО S_X = {std_X:.4f}")
print(f"СКО S_Y = {std_Y:.4f}")

# Объединённая оценка дисперсии
S_pooled_sq = ((m - 1) * var_X + (n - 1) * var_Y) / (m + n - 2)
S_pooled = np.sqrt(S_pooled_sq)

print(f"\nОбъединённая дисперсия S_pooled² = {S_pooled_sq:.4f}")
print(f"Объединённое СКО S_pooled = {S_pooled:.4f}")

# Статистика критерия (t-наблюдаемое)
t_stat = (mean_X - mean_Y) / (S_pooled * np.sqrt(1/m + 1/n))
print(f"\nt-наблюдаемое = {t_stat:.4f}")

# Степени свободы
df = m + n - 2
print(f"Степени свободы = {df}")

# Критическое значение (двустороннее)
alpha = 0.05
t_crit = stats.t.ppf(1 - alpha/2, df)
print(f"t-критическое (двустороннее) = ±{t_crit:.4f}")

# p-value (двустороннее)
p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))
print(f"p-value = {p_value:.6f}")

# Вывод о гипотезе
if abs(t_stat) > t_crit:
    print("\nВывод: H0 отвергается. Средние различаются статистически значимо.")
else:
    print("\nВывод: H0 не отвергается. Нет оснований считать средние разными.")

# ========== ВИЗУАЛИЗАЦИЯ 1: Сравнение нормального распределения и t-распределений ==========
x_vals = np.linspace(-4.5, 4.5, 1000)

# Плотности распределений
normal_dist = stats.norm.pdf(x_vals, 0, 1)          # Стандартное нормальное N(0,1)
t_dist_5 = stats.t.pdf(x_vals, df=5)               
t_dist_15 = stats.t.pdf(x_vals, df=15)              

plt.figure(figsize=(12, 7))

# Рисуем все три распределения
plt.plot(x_vals, normal_dist, 'k-', linewidth=2.5, label='Стандартное нормальное')
plt.plot(x_vals, t_dist_5, 'r--', linewidth=2.5, label='t-распределение, ν = 5')
plt.plot(x_vals, t_dist_15, 'b-.', linewidth=2.5, label='t-распределение, ν = 15')

# Закрашиваем область под кривыми для наглядности
plt.fill_between(x_vals, 0, normal_dist, where=(x_vals >= 2), color='black', alpha=0.1)
plt.fill_between(x_vals, 0, t_dist_5, where=(x_vals >= 2), color='red', alpha=0.1)

# Настройка графика
plt.title('Сравнение нормального распределения и t-распределений', fontsize=16, fontweight='bold')
plt.xlabel('Значение статистики', fontsize=14)
plt.ylabel('Плотность вероятности', fontsize=14)

# Добавляем вертикальные линии для квантилей
plt.axvline(1.96, color='black', linestyle=':', linewidth=1.5, alpha=0.7)
plt.axvline(2.57, color='red', linestyle=':', linewidth=1.5, alpha=0.7)
plt.axvline(2.13, color='blue', linestyle=':', linewidth=1.5, alpha=0.7)

# Добавляем подписи к вертикальным линиям
plt.text(1.96, 0.05, 'z₀.₉₇₅ = 1.96', ha='center', fontsize=9, color='black')
plt.text(2.57, 0.08, 't₀.₉₇₅(5) = 2.57', ha='center', fontsize=9, color='red')
plt.text(2.13, 0.12, 't₀.₉₇₅(15) = 2.13', ha='center', fontsize=9, color='blue')

plt.legend(loc='upper right', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.3)

plt.tight_layout()
plt.show()

# ========== ВИЗУАЛИЗАЦИЯ 2: Распределение Стьюдента с критическими областями и t-наблюдаемым ==========
x_vals = np.linspace(-4.5, 4.5, 1000)
t_dist = stats.t.pdf(x_vals, df)

plt.figure(figsize=(10, 5))
plt.plot(x_vals, t_dist, 'k-', linewidth=2, label=f'Распределение Стьюдента (df={df})')

# Критические границы
plt.axvline(t_crit, color='red', linestyle='--', linewidth=2, label=f'Критическая граница +{t_crit:.3f}')
plt.axvline(-t_crit, color='red', linestyle='--', linewidth=2, label=f'Критическая граница -{t_crit:.3f}')

# Наблюдаемое значение t-статистики
plt.axvline(t_stat, color='blue', linestyle='-', linewidth=2.5, label=f't-набл = {t_stat:.3f}')

# Закрашиваем критические области (хвосты)
plt.fill_between(x_vals, 0, t_dist, where=(x_vals >= t_crit), color='red', alpha=0.3, label='Критическая область (правый хвост)')
plt.fill_between(x_vals, 0, t_dist, where=(x_vals <= -t_crit), color='red', alpha=0.3, label='Критическая область (левый хвост)')

# Добавляем текстовую информацию на график
textstr = f'|t_набл| = {abs(t_stat):.4f}\nt_крит = {t_crit:.4f}\np-value = {p_value:.4f}\nα = {alpha}'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=10,
         verticalalignment='top', bbox=props)

plt.title('t-критерий Стьюдента: критическая область и наблюдаемое значение', fontsize=14)
plt.xlabel('t-статистика', fontsize=12)
plt.ylabel('Плотность вероятности', fontsize=12)
plt.legend(loc='upper right')
plt.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout()
plt.show()