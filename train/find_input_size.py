def f(x, kernel_size=2):
    return (x + 2 - 1*(kernel_size - 1) - 1)/2 + 1

img_h = 354
img_w = 472

print(f"h = {f(f(img_h))}")
print(f"w = {f(f(img_w))}")