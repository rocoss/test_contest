def gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def lcm(a, b):
    return a * b // gcd(a, b)

def count_valid_numbers(x, a, b, c):
    count = 0
    
    # Проверяем каждую пару
    pairs = [(a, b), (a, c), (b, c)]
    
    for x1, x2 in pairs:
        lcm_value = lcm(x1, x2)
        count += x // lcm_value  # Числа кратные НОК двух чисел
        
        # Исключаем числа кратные третьему числу
        third = [a, b, c]
        third.remove(x1)
        third.remove(x2)
        
        count -= x // lcm(lcm_value, third[0])
    
    return count

def find_nth_number(a, b, c, n):
    low = 1
    high = 10**18
    result = -1
    
    while low <= high:
        mid = (low + high) // 2
        if count_valid_numbers(mid, a, b, c) < n:
            low = mid + 1
        else:
            result = mid
            high = mid - 1
            
    return result if result <= 10**18 else -1

def find_nth_element():
    a, b, c = map(int, input().split())
    n = int(input())

    result = find_nth_number(a, b, c, n)
    print(result)

# Вызываем главную функцию
find_nth_element()
