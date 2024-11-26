from sympy import is_primitive_root, primerange

def find_primitive_root(M): #为模数求原根
    primes = list(primerange(1, 100))  # 获取小于n的所有质数
    for i in primes:
        if is_primitive_root(i, M):
            return i
    return None

M = 0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000001
g = find_primitive_root(M)

def qpow(a, b):   # 快速幂，计算 a**b % M
    ans = 1
    while b:
        if b & 1: ans = ans*a % M
        a = a*a % M
        b >>= 1
    return ans

def helper(i: int, power: int):
    '''
    i: 要逆序的整数
    power: 2**(r-2)
    '''
    res = 0
    while i:
        if i&1: 
            res += power
        i >>= 1
        power >>= 1
    return res

def NTT(x: list, inv=1):
    l = 0
    while (1<<l) < len(x): l += 1

    for r in range(1, l+1):
        mid = 1 << (l-r)
        t = qpow(g, (M-1)>>r)
        if inv == -1: 
            t = qpow(t, M-2)
        group_count = 1 << (r-1)
        for i in range(group_count):
            power = helper(i, group_count>>1)
            a_p = qpow(t, power)
            base = i<<(l-r+1)
            for j in range(mid):
                u = x[base+j]
                v = x[base+j+mid]*a_p % M
                x[base+j] = (u+v) % M
                x[base+j+mid] = (u-v) % M

    rev = [0] * len(x)
    for i in range(len(x)):
        rev[i] = (rev[i >> 1] >> 1) | ((i&1) << (l-1))
        if i < rev[i]:
            x[i], x[rev[i]] = x[rev[i]], x[i]
    return x

a = [0x0000000800000007000000060000000500000004000000030000000200000001] * (1 << 24)
result = NTT(a)
print(hex(result[0]))
print(hex(result[1]))