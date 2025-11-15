import sys
input = sys.stdin.readline

def solve():
    n = int(input())
    if n == 0:
        return 0
    
    arr = list(map(int, input().split()))
    
    pos = sorted([x for x in arr if x > 0])
    neg = sorted([-x for x in arr if x < 0])
    
    # Базовая проверка: количество должно совпадать
    if len(pos) != len(neg):
        return -1
    
    if len(pos) == 0:
        return 0
    
    # Жадный алгоритм: сопоставляем отсортированные массивы
    # Для каждого установления берем минимальный подходящий разрыв
    total_time = 0
    
    for i in range(len(pos)):
        if pos[i] > neg[i]:
            return -1
        total_time += neg[i] - pos[i]
    
    return total_time

t = int(input())
for _ in range(t):
    result = solve()
    print(result)
