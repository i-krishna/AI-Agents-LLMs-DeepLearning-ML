import pdb
import logging
def mystery_fn(z):
    if z < 0:
        return '-' + mystery_fn(-z)
    s = []
    print(s)
    while z:
        print(z)
        if len(s) % 4 == 3:
            s.append(',')
            print(s)
        s.append(str(z % 10))
        z //= 10
    return ''.join(reversed(s))

print(mystery_fn(1234567))
