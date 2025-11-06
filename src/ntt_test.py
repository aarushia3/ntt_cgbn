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
print(31**8 % 2013265921)