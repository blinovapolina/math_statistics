import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("RGR1_A-7_X1-X4.csv")

print("Размер выборки:", len(df))
print(df.head())


def analyze_column(data, name):

    print("\n=================================================")
    print("АНАЛИЗ СТОЛБЦА", name)
    print("=================================================")

    data = np.array(data)
    n = len(data)

    # ШАГ 1: Вариационный ряд
    sorted_data = np.sort(data)

    print("\nШАГ 1: Вариационный ряд")
    print("Исходная выборка:", data)
    print("Вариационный ряд:", sorted_data)

    plt.figure(figsize=(14,6))
    plt.plot(range(len(data)), data, 'o', label='Исходная', alpha=0.7, markersize=8)
    plt.plot(range(len(sorted_data)), sorted_data, 's', label='Вариационный ряд', markersize=8)
    plt.xlabel("Номер наблюдения", fontsize=12)
    plt.ylabel("Значение", fontsize=12)
    plt.title(f"Вариационный ряд ({name})", fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(alpha=0.3)
    plt.show()


    # ШАГ 2: Эмпирическая функция распределения
    print("\nШАГ 2: Эмпирическая функция распределения")

    x_sorted = np.sort(data)

    unique_vals, counts = np.unique(x_sorted, return_counts=True)
    cum_probs = np.cumsum(counts) / n
    prev_probs = np.concatenate(([0], cum_probs[:-1]))

    plt.figure(figsize=(14,6))

    for x, y0, y1 in zip(unique_vals, prev_probs, cum_probs):
        plt.plot([x,x],[y0,y1],'b--',linewidth=2)

    plt.plot([x_sorted[0]-1.5, unique_vals[0]],[0,0],'b-',linewidth=2)

    for i in range(len(unique_vals)-1):
        plt.plot([unique_vals[i],unique_vals[i+1]],
                 [cum_probs[i],cum_probs[i]],'b-',linewidth=2)

    plt.plot([unique_vals[-1],x_sorted[-1]+1.5],
             [cum_probs[-1],1],'b-',linewidth=2)

    plt.scatter(unique_vals, prev_probs, facecolors='none', edgecolors='blue', s=80)
    plt.scatter(unique_vals, cum_probs, color='blue', s=80)

    plt.xlabel("x", fontsize=12)
    plt.ylabel("Fn(x)", fontsize=12)
    plt.title(f"Эмпирическая функция распределения ({name})", fontsize=16)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.show()


    # ШАГ 3: Выборочные характеристики
    print("\nШАГ 3: Выборочные характеристики")

    x_bar = np.mean(data)

    S2 = np.mean((data - x_bar)**2)
    S = np.sqrt(S2)

    sigma2 = np.sum((data - x_bar)**2)/(n-1)
    sigma = np.sqrt(sigma2)

    median = np.median(data)

    q25 = np.quantile(data,0.25)
    q75 = np.quantile(data,0.75)

    print("Среднее X =", x_bar)
    print("Смещённая дисперсия S² =", S2)
    print("Стандартное отклонение S =", S)
    print("Несмещённая дисперсия σ² =", sigma2)
    print("Исправленное σ =", sigma)
    print("Медиана =", median)
    print("Q0.25 =", q25)
    print("Q0.75 =", q75)


    # ШАГ 4: Гистограммы
    print("\nШАГ 4: Гистограммы")

    s = np.std(data, ddof=1)
    x_min = data.min()
    x_max = data.max()

    # Правило Скотта
    h_scott = 3.5 * s * n**(-1/3)
    k_scott = int(np.ceil((x_max - x_min) / h_scott))

    # Правило Фридмана–Диакониса
    q25 = np.quantile(data,0.25)
    q75 = np.quantile(data,0.75)
    IQR = q75 - q25

    h_fd = 2 * IQR * n**(-1/3)
    k_fd = int(np.ceil((x_max - x_min) / h_fd))

    # Правило Стерджеса
    k_sturges = int(1 + np.floor(np.log2(n)))

    print("Скотт: k =", k_scott)
    print("Фридман–Диаконис: k =", k_fd)
    print("Стерджес: k =", k_sturges)

    # Построение гистограмм
    fig, axes = plt.subplots(2,2, figsize=(12,7))

    # Скотт
    axes[0,0].hist(data, bins=k_scott, edgecolor='black')
    axes[0,0].axvline(x_bar, color='red', linestyle='--', linewidth=2)
    axes[0,0].set_title(f"Скотт (k={k_scott})")
    axes[0,0].grid(alpha=0.3)

    # фиксированная
    axes[0,1].hist(data, bins=5, edgecolor='black')
    axes[0,1].axvline(x_bar, color='red', linestyle='--', linewidth=2)
    axes[0,1].set_title("Фиксированная (k=5)")
    axes[0,1].grid(alpha=0.3)

    # Фридман–Диаконис
    axes[1,0].hist(data, bins=k_fd, edgecolor='black')
    axes[1,0].axvline(x_bar, color='red', linestyle='--', linewidth=2)
    axes[1,0].set_title(f"Фридман–Диаконис (k={k_fd})")
    axes[1,0].grid(alpha=0.3)

    # Стерджес
    axes[1,1].hist(data, bins=k_sturges, edgecolor='black')
    axes[1,1].axvline(x_bar, color='red', linestyle='--', linewidth=2)
    axes[1,1].set_title(f"Стерджес (k={k_sturges})")
    axes[1,1].grid(alpha=0.3)

    plt.suptitle(f"Гистограммы для {name}", fontsize=14)
    plt.tight_layout()
    plt.show()



# =================================================
# АНАЛИЗ X1 X2 X3
# =================================================
analyze_column(df["X1"], "X1")
analyze_column(df["X2"], "X2")
analyze_column(df["X3"], "X3")
analyze_column(df["X4"], "X4")