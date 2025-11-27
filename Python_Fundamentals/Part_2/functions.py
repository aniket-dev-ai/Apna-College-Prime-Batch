def hello(): #fxn definition
    print("Hello")
    
hello() #fxn call


def sum(a,b=1):
    return a + b

print(sum(4,5))
print(sum(4),"using default parameter")


def avg(a,b,c):
    sum = a+b+c
    return sum//3

print(avg(14,15,16))

# Types of functions
# |
# |___ built-in print() , input() , range() , type()
# |
# |___ User defined sum() , xyz(a,b)