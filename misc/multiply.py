# x is double (float in Python)
# y is double
x = 38.172
y = 2.555

exact = x * y                    # full-precision product
rounded = round(exact, 4)        # 4 digits of exactitude on decimals
abs_error = abs(exact - rounded)
rel_error = abs_error / abs(exact)

print(f"Exact product : {exact}")
print(f"Rounded (4 d) : {rounded:.4f}")
print(f"Absolute error: {abs_error:.10f}")
print(f"Relative error: {rel_error:.3e}")
