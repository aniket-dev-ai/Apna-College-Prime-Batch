a = 5
b = 6
sum = a+b

# normal formating
print("sum of {} & {} is {}".format(a,b,sum))
print("we are learning {} language".format("python"))

# index based formatting 
print("sum of {1} & {0} is {2}".format(a,b,sum))

# value based formatting
print("valyes of var {a} & {b}".format(a=5,b=10))

# f string formatting

a = 10
b= 20

print(f"The sum of {a} and {b} is = {a+b}")

