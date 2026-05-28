def calc(first: float, second: float, operation: str) -> float:
    if operation == "a":
        return first + second
    elif operation == "s":
        return first - second
    elif operation == "m":
        return first * second
    elif operation == "d":
        if(second == 0):
            raise ValueError("Cannot divide by zero!")
        return first / second
    else:
        raise ValueError("Invalid operation")
