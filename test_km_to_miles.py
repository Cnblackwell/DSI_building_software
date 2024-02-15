import math

def km_to_miles(km):
    # Check if the input is valid (non-negative integer)
    if km < 0:
        raise ValueError("Input must be a non-negative integer")

    # Conversion factor from kilometers to miles
    conversion_factor = 0.621371

    # Convert kilometers to miles
    miles = km * conversion_factor

    return miles