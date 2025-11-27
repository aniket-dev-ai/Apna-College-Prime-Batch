# marks1 = 100
# marks2 = 65
# marks3 = 74
# marks4 = 98
# marks5 = 79

# Array hi kehlo yaar

marks = [100,65,74,98,79]

print(marks)
print(len(marks))
print(marks[4])

# Strings are immutable but lists are mutable

marks[4] = 88
print(marks[4])
print(marks)

# list can have mutliple types of value

marks = [100,65,74,98,79,True,"Hello"]
print(marks)
print(marks[6])

# Slicing is possible
print(marks[3:6])

# Method of lists
# list.append(val)      Add one element at the end
# l.insert(idx,val)     Insert element at idx
# l.sort()              arranges in increasing order
# l.reverse()           reverses order

num = [1,2,3,4,5,6]

print(num)

num.append(7)
print(num)

num.insert(4,3)
print(num)

num.sort()
print(num)

num.reverse()
print(num)

#  Using Loops with list

for val in num:
    print(val)

