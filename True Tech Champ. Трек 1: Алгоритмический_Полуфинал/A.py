import sys
input = sys.stdin.readline

t = int(input())

for _ in range(t):
    s = input().strip()
    
    # Вычисляем коэффициенты для каждой буквы по модулю 9
    coeffs = {'M': 0, 'T': 0, 'C': 0}
    power_mod = 1
    
    for i in range(len(s) - 1, -1, -1):
        coeffs[s[i]] = (coeffs[s[i]] + power_mod) % 9
        power_mod = (power_mod * 10) % 9
    
    # Получаем уникальные буквы
    unique_letters = list(set(s))
    
    # Строка стабильна, если все коэффициенты присутствующих букв равны 0 по модулю 9
    is_stable = all(coeffs[letter] == 0 for letter in unique_letters)
    
    print(1 if is_stable else 0)
