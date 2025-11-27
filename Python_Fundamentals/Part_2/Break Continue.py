# break example
items = ["apple", "banana", "mango", "orange"]

for item in items:
    if item == "mango":
        print("Found:", item)
        break
    print("Checking:", item)


# continue example
for num in range(1, 10):
    if num % 2 == 0:
        continue
    print("Odd number:", num)


# combined example
numbers = [5, -3, 7, -1, 9, 0, 2]

for n in numbers:
    if n < 0:
        continue
    if n == 0:
        break
    print("Processing:", n)
