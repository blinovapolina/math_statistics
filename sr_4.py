# Шаг 0. Импортируем библиотеки
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Шаг 1. Загрузка данных
data = pd.read_csv('variant_2_data.csv')
print(data.head())

x = data['x']
y = data['y']
n = len(x)

# Шаг 2. Диаграмма рассеяния
plt.scatter(x, y, color='blue', label='Данные')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Диаграмма рассеяния y vs x')
plt.grid(True)
plt.legend()
plt.show()

# Шаг 3. Оценка коэффициентов регрессии
x_mean = np.mean(x)
y_mean = np.mean(y)

Sxx = np.sum((x - x_mean)**2)
Sxy = np.sum((x - x_mean)*(y - y_mean))

b1 = Sxy / Sxx
b0 = y_mean - b1 * x_mean

print(f'Оценка b0: {b0}, b1: {b1}')
print(f'Уравнение регрессии: y_hat = {b0:.4f} + {b1:.4f} * x')

# Шаг 4. Прогнозные значения и остатки
y_hat = b0 + b1 * x
residuals = y - y_hat

results = pd.DataFrame({'i': range(1, n+1), 'x': x, 'y': y, 'y_hat': y_hat, 'residual': residuals})
print(results)

# Шаг 5. Остаточная дисперсия и стандартное отклонение
RSS = np.sum(residuals**2)
s2 = RSS / (n - 2)
s = np.sqrt(s2)

print(f'RSS: {RSS}, s^2: {s2}, s: {s}')

# Шаг 6. Стандартные ошибки и доверительные интервалы для коэффициентов
SE_b1 = s / np.sqrt(Sxx)
SE_b0 = s * np.sqrt(1/n + x_mean**2 / Sxx)

alpha = 0.05
t_crit = stats.t.ppf(1 - alpha/2, df=n-2)

CI_b0 = (b0 - t_crit*SE_b0, b0 + t_crit*SE_b0)
CI_b1 = (b1 - t_crit*SE_b1, b1 + t_crit*SE_b1)

print(f'95% CI b0: {CI_b0}')
print(f'95% CI b1: {CI_b1}')

# Шаг 7. Доверительный интервал для функции регрессии
x0 = np.linspace(min(x), max(x), 100)
y0_hat = b0 + b1 * x0
SE_y0 = s * np.sqrt(1/n + (x0 - x_mean)**2 / Sxx)

y_lower = y0_hat - t_crit * SE_y0
y_upper = y0_hat + t_crit * SE_y0

# График с доверительным интервалом
plt.scatter(x, y, color='blue', label='Данные')
plt.plot(x0, y0_hat, color='red', label='Регрессия')
plt.fill_between(x0, y_lower, y_upper, color='gray', alpha=0.3, label='95% доверительный интервал')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Линейная регрессия с доверительным интервалом')
plt.grid(True)
plt.legend()
plt.show()

# Шаг 8. Коэффициент детерминации
TSS = np.sum((y - y_mean)**2)
R2 = 1 - RSS / TSS
print(f'R^2: {R2}')