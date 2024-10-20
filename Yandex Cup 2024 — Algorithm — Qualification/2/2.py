def count_hills(heights):
    n = len(heights)
    count = 0
    
    for i in range(1, n - 1):
        # Проверяем, является ли текущий элемент вершиной горки
        if heights[i] > heights[i - 1] and heights[i] > heights[i + 1]:
            # Проверяем наличие возрастающей части слева
            left = i - 1
            while left > 0 and heights[left - 1] < heights[left]:
                left -= 1
            
            # Проверяем наличие убывающей части справа
            right = i + 1
            while right < n - 1 and heights[right + 1] < heights[right]:
                right += 1
            
            # Увеличиваем счетчик на количество возможных горок
            count += (i - left) * (right - i)
    
    return count

def main():
    t = int(input())
    results = []
    
    for _ in range(t):
        n = int(input())
        heights = list(map(int, input().split()))
        results.append(count_hills(heights))
    
    for result in results:
        print(result)

# Запуск основной функции
main()
