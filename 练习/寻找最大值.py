max_length = None
for i in range(0,1001):
    if i % 5 == 0 and i % 7 == 0:
        max_length = i
print(f"Max length={max_length}")