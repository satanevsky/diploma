import sys


def forward_out(filename):
    def decorate(func):
        def result(*args, **kwargs):
            stdout = sys.stdout
            stderr = sys.stderr
            try:
                with open(filename, 'a') as out:
                    sys.stdout = sys.stderr = out
                    ans = func(*args, **kwargs)
            finally:
                sys.stdout = stdout
                sys.stderr = stderr
            return ans
        return result
    return decorate
