import math

def is_prime_fast(n):
    """
    Детерминированный тест Миллера-Рабина на простоту.
    Гарантированно корректен для чисел до 3.3 * 10^14.
    """
    if n < 2:
        return False
    if n == 2 or n == 3:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    if n < 25:
        return True

    # Находим d и s такие, что n - 1 = 2^s * d
    d = n - 1
    s = 0
    while d % 2 == 0:
        d //= 2
        s += 1

    # Базы (свидетели), достаточные для чисел нашего диапазона
    bases = [2, 3, 5, 7, 11, 13, 17]

    for a in bases:
        if a >= n:
            break
        x = pow(a, d, n)  # x = a^d mod n
        if x == 1 or x == n - 1:
            continue
        
        # Цикл по r от 1 до s-1
        is_composite = True
        for _ in range(s - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                is_composite = False
                break
        
        if is_composite:
            return False
            
    return True


def find_special_numbers(prefix):
    """Находит номера через генерацию из квадратов с быстрой проверкой на простоту."""
    result = []
    prefix_int = int(prefix)
    
    # Диапазон 11-значных номеров с данным префиксом
    min_7 = 7 * 10**10 + prefix_int * 10**7
    max_7 = min_7 + 10**7 - 1
    min_8 = 8 * 10**10 + prefix_int * 10**7
    max_8 = min_8 + 10**7 - 1
    
    # Границы корней для квадратов (оптимизация: используем isqrt)
    sqrt_min_7 = math.isqrt(min_7)
    sqrt_max_7 = math.isqrt(max_7)
    sqrt_min_8 = math.isqrt(min_8)
    sqrt_max_8 = math.isqrt(max_8)
    
    seen = set()
    
    # Вариант 1: num_7 - квадрат, num_8 - простое
    for root in range(sqrt_min_7, sqrt_max_7 + 2):
        square_7 = root * root
        if min_7 <= square_7 <= max_7:
            square_8 = square_7 + 10**10
            if is_prime_fast(square_8):
                num_10_digits = square_7 % (10**10)
                if num_10_digits not in seen:
                    seen.add(num_10_digits)
                    result.append(num_10_digits)
    
    # Вариант 2: num_8 - квадрат, num_7 - простое
    for root in range(sqrt_min_8, sqrt_max_8 + 2):
        square_8 = root * root
        if min_8 <= square_8 <= max_8:
            square_7 = square_8 - 10**10
            if is_prime_fast(square_7):
                num_10_digits = square_8 % (10**10)
                if num_10_digits not in seen:
                    seen.add(num_10_digits)
                    result.append(num_10_digits)
    
    result.sort()
    return [str(num).zfill(10) for num in result]


def main():
    try:
        t = int(input())
        for _ in range(t):
            prefix = input().strip()
            special_numbers = find_special_numbers(prefix)
            print(len(special_numbers), *special_numbers)
    except (IOError, ValueError):
        pass

if __name__ == "__main__":
    main()
