import pdb
import logging
import traceback

# Try icecream, fallback to print if not installed
try:
    from icecream import ic
except ImportError:
    ic = lambda *args: print(*args)

# --- logging setup ---
logging.basicConfig(level=logging.DEBUG, format="%(levelname)s: %(message)s")

def mystery_fn(z):
    # --- assert: validate input type ---
    assert isinstance(z, int), f"Expected int, got {type(z)}"

    logging.debug(f"mystery_fn called with z={z}")

    if z < 0:
        return '-' + mystery_fn(-z)

    s = []

    # --- icecream: inspect initial state ---
    ic(z, s)

    while z:
        logging.debug(f"Top of loop: z={z}, s={s}")

        if len(s) % 4 == 3:
            s.append(',')
            # --- print: track comma insertion ---
            print(f"[print] Comma inserted, s={s}")

        s.append(str(z % 10))
        z //= 10

        # --- icecream: track each digit appended ---
        ic(z, s)

    result = ''.join(reversed(s))
    logging.debug(f"Final result: {result}")
    return result


# --- pdb: uncomment to step through interactively (won't work in CoderPad) ---
# pdb.set_trace()

# --- traceback: catch unexpected errors gracefully ---
try:
    print(mystery_fn(1234567))
    print(mystery_fn(-987))
    print(mystery_fn(0))       # edge case
    print(mystery_fn("oops"))  # triggers assert
except AssertionError as e:
    print(f"[assert failed] {e}")
except Exception:
    traceback.print_exc()
