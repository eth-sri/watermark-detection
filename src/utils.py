import numpy as np
import random
import string


def logit(p):
    return np.log((p + 1e-10) / (1 - p + 1e-10))


def generate_random_prefix():
    letters = string.ascii_letters
    digits = string.digits
    special_chars = "!@#$%^&*()-_=+{}[];:<>,.?|`~"

    password = [random.choice(special_chars)]

    remaining_chars = random.choices(letters + digits + special_chars, k=4)

    password.extend(remaining_chars)

    random.shuffle(password)

    return "".join(password)


def get_2_random_fruits():
    fruits = [
        "apple",
        "banana",
        "cherry",
        "date",
        "elderberry",
        "fig",
        "grape",
        "honeydew",
        "kiwi",
        "lemon",
        "mango",
        "nectarine",
        "orange",
        "papaya",
        "quince",
        "raspberry",
        "strawberry",
        "tangerine",
        "ugli",
        "watermelon",
    ]
    return random.sample(fruits, 2)
