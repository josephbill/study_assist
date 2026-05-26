# fib sequence 
# 1,1,2,3,5,8,13.....
def fib_memo(n, cache={}):
    if n in cache:
        return cache[n]
    if n == 1 or n == 2:
        return 1
    cache[n] = fib_memo(n-1, cache) + fib_memo(n-2, cache)
    return cache[n]


for n in range(1,101):
    print(f"{n} is {fib_memo(n)}")