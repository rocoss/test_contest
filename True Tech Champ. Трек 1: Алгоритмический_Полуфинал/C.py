import math
import sys

input = sys.stdin.read
data = input().split()

index = 0
t = int(data[index])
index += 1

for _ in range(t):
    p = int(data[index])
    q = int(data[index + 1])
    index += 2
    
    # Вычисляем gcd(2*p, 360)
    gcd_val = math.gcd(2 * p, 360)
    
    # k = 360 * q / gcd(2*p, 360)
    k = 360 * q // gcd_val
    
    # Отражений: k - 1, кроме k=1 (0)
    if k == 1:
        print(0)
    else:
        print(k - 1)
