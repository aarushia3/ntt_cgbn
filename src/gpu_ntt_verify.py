def find_primitive_root(order, modulus):
    primitive_roots = []
    for candidate in range(2, modulus):
        is_primitive = True
        if pow(candidate, order, modulus) == 1:
            for exp in range(1, order):
                if pow(candidate, exp, modulus) == 1:
                    is_primitive = False
                    break
        else:
            is_primitive = False
        if is_primitive:
            primitive_roots.append(candidate)
    return primitive_roots

# print(find_primitive_root(8, 2013265921)) [211723194, 420899707, 1592366214, 1801542727]
# print(31**8 % 2013265921)

def modinv(a, p):
    """Compute modular inverse of a modulo p."""
    return pow(a, p - 2, p)  # Fermat’s little theorem (p must be prime)

def ntt(a, root, mod):
    """
    Compute the Number Theoretic Transform (NTT) of a list `a`
    under modulus `mod` using given primitive root of unity `root`.
    """
    n = len(a)
    if n & (n - 1):
        raise ValueError("Length of input must be a power of 2")

    # Bit-reversal permutation
    j = 0
    for i in range(1, n):
        bit = n >> 1
        while j & bit:
            j ^= bit
            bit >>= 1
        j ^= bit
        if i < j:
            a[i], a[j] = a[j], a[i]

    length = 2
    while length <= n:
        wlen = pow(root, n // length, mod)
        for i in range(0, n, length):
            w = 1
            for j in range(i, i + length // 2):
                u = a[j]
                v = a[j + length // 2] * w % mod
                a[j] = (u + v) % mod
                a[j + length // 2] = (u - v + mod) % mod
                w = w * wlen % mod
        length <<= 1
    return a

def intt(a, root, mod):
    """Compute the inverse NTT (INTT)."""
    n = len(a)
    inv_n = modinv(n, mod)
    inv_root = modinv(root, mod)
    a = ntt(a, inv_root, mod)
    for i in range(n):
        a[i] = a[i] * inv_n % mod
    return a

if __name__ == "__main__":
    # Ideally we want that data, mod and order can be input
    # maybe the root as well if we don't want to call find_primitive_root
    data = [1, 2, 3, 4]
    mod = 7681
    order = len(data)
    
    roots = find_primitive_root(order, mod)
    print(roots)

    print("Input:", data)
    ntt_result = ntt(data.copy(), roots[0], mod)
    print("NTT:", ntt_result)

    recovered = intt(ntt_result.copy(), roots[0], mod)
    print("Inverse NTT:", recovered)