import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------
# ЗАДАНИЕ: Первичная обработка выборки
# -------------------------------------------------
data = np.loadtxt('variant_6.csv', delimiter=',')

print(f"Объём выборки: n = {len(data)}")
print()

# -------------------------------------------------
# ШАГ 1: Вариационный ряд
# -------------------------------------------------
sorted_data = np.sort(data)

print("Первые 5 значений вариационного ряда:")
print(sorted_data[:5])
print("Последние 5 значений вариационного ряда:")
print(sorted_data[-5:])
print()

# -------------------------------------------------
# ШАГ 2: Выборочные оценки
# -------------------------------------------------
x_bar = np.mean(data)
s2 = np.var(data, ddof=1)
s = np.std(data, ddof=1)
median = np.median(data)
x_min = np.min(data)
x_max = np.max(data)

print("Выборочные оценки:")
print(f"  Среднее:      x̄ = {x_bar:.3f}")
print(f"  Дисперсия:    s² = {s2:.3f}")
print(f"  Ст. откл.:    s = {s:.3f}")
print(f"  Медиана:      x̃ = {median:.3f}")
print(f"  Размах:       {x_max - x_min:.1f}")
print()

# -------------------------------------------------
# ШАГ 3: Полигон частот
# -------------------------------------------------

unique_vals, counts = np.unique(np.sort(data), return_counts=True)

for x, n_i in zip(unique_vals, counts):
    print(f"  x = {x:2.0f} → частота = {n_i}")

plt.figure(figsize=(17, 5)) 
plt.plot(unique_vals, counts, 'o-', color='coral', linewidth=2, markersize=10)
plt.bar(unique_vals, counts, width=0.5, alpha=0.3, color='coral')

for x, y in zip(unique_vals, counts):
    plt.text(x, y + max(counts)*0.03, str(int(y)), ha='center', fontsize=10)

plt.xlabel("Значение x", fontsize=12)
plt.ylabel("Частота", fontsize=12)
plt.title("Полигон частот", fontsize=14)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# -------------------------------------------------
# ШАГ 4: Эмпирическая функция распределения
# -------------------------------------------------
n = len(data)
ecdf = np.arange(1, n + 1) / n

plt.figure()
plt.step(sorted_data, ecdf, where='post')
plt.title("Эмпирическая функция распределения")
plt.xlabel("x")
plt.ylabel("F*(x)")
plt.grid(alpha=0.3)
plt.show()

# -------------------------------------------------
# ШАГ 5: Гистограмма + правило Скотта
# -------------------------------------------------
h_scott = 3.5 * s * n**(-1/3)
k_scott = int(np.ceil((x_max - x_min) / h_scott))

print("Правило Скотта:")
print(f"h = 3.5 · s · n⁻¹ᐟ³ = {h_scott:.2f}")
print(f"Число интервалов по Скотту: k = {k_scott}")
print(f"Фиксированное число интервалов: k = 5")
print()

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# По Скотту
axes[0].hist(data, bins=k_scott, edgecolor='black', color='steelblue', alpha=0.85)
axes[0].axvline(x_bar, color='red', linestyle='--',
                label=f'Среднее = {x_bar:.1f}')
axes[0].set_title(f"Гистограмма (Скотт, k={k_scott})")
axes[0].set_xlabel("x")
axes[0].set_ylabel("Частота")
axes[0].legend()
axes[0].grid(alpha=0.3)

# 5 интервалов
axes[1].hist(data, bins=5, edgecolor='black', color='coral', alpha=0.85)
axes[1].axvline(x_bar, color='red', linestyle='--',
                label=f'Среднее = {x_bar:.1f}')
axes[1].set_title("Гистограмма (фиксированно, k=5)")
axes[1].set_xlabel("x")
axes[1].set_ylabel("Частота")
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()

# -------------------------------------------------
# ШАГ 6: Сравнение с истинными параметрами
# -------------------------------------------------
mu_true = 75     
sigma2_true = 169  

print("Сравнение с истинными параметрами:")
print(f"  Истинное μ = {mu_true}, выборочное x̄ = {x_bar:.3f}")
print(f"  Истинное σ² = {sigma2_true}, выборочное s² = {s2:.3f}")
