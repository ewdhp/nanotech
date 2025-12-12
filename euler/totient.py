def gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def euler_totient(n):
    if n == 0:
        return 0
    count = 0
    for i in range(1, n + 1):
        if gcd(n, i) == 1:
            count += 1
    return count

# Example usage
for num in range(1, 21):
    print(f"φ({num}) = {euler_totient(num)}")