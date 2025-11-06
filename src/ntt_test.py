# find the primitive root of unity
def find_primitive_root(order, mod):
    for g in range(2, mod):
        if all(pow(g, order // p, mod) != 1 for p in prime_factors(order)):
            return g
    return None

find_primitive_root(4, 7681)