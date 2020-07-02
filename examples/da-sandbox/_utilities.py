
LEFT_PAD = 3

def print_start_of(module_name):
    print(" " * LEFT_PAD, end='')
    print(str(module_name) + ": started")

def print_end_of(module_name):
    print(" " * LEFT_PAD, end='')
    print(str(module_name) + ": ended\n")

def print_info(
        module_name,
        *args,
        **kwargs):
    print(" " * LEFT_PAD + str(module_name) + ": ", end='')
    print(*args, **kwargs)

