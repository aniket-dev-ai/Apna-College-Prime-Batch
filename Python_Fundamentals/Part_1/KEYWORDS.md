# 🧭 Python Keywords — Quick Reference
 
  

[![Python Version](https://img.shields.io/badge/python-3.14-blue.svg)](https://www.python.org)

[![Topics](https://img.shields.io/badge/topics-keywords%20%7C%20examples-brightgreen)](https://www.python.org)

---

## Table of Contents

- [🧭 Python Keywords — Quick Reference](#-python-keywords--quick-reference)
  - [Table of Contents](#table-of-contents)
  - [How to use this file](#how-to-use-this-file)
    - [and](#and)
    - [as](#as)
    - [assert](#assert)
    - [async](#async)
    - [await](#await)
    - [break](#break)
    - [case](#case)
    - [class](#class)
    - [continue](#continue)
    - [def](#def)
    - [del](#del)
    - [elif](#elif)
    - [else](#else)
    - [except](#except)
    - [False](#false)
    - [finally](#finally)
    - [for](#for)
    - [from](#from)
    - [global](#global)
    - [if](#if)
    - [import](#import)
    - [in](#in)
    - [is](#is)
    - [lambda](#lambda)
    - [match](#match)
    - [None](#none)
    - [nonlocal](#nonlocal)
    - [not](#not)
    - [or](#or)
    - [pass](#pass)
    - [raise](#raise)
    - [return](#return)
    - [True](#true)
    - [try](#try)
    - [while](#while)
    - [with](#with)
    - [yield](#yield)

---

## How to use this file

Each keyword below has:

- **Explanation:** what the keyword does
- **Short example:** a small code snippet to illustrate typical use

Copy examples into a `.py` file or run interactively to learn by doing.

---

### and

**Explanation:** Logical AND operator. Evaluates left-to-right and returns the first falsy operand or the last truthy operand.


**Example:**
```python
a = True and False
print(a)  # False
```

---

### as

**Explanation:** Gives an alias when importing modules or binds an exception to a name in except clauses.


**Example:**
```python
import math as m
print(m.sqrt(16))

try:
    1/0
except ZeroDivisionError as e:
    print('caught', e)
```

---

### assert

**Explanation:** Debugging aid that tests a condition and raises AssertionError if it fails.


**Example:**
```python
assert 2+2==4
# assert 1==0  # would raise AssertionError
```

---

### async

**Explanation:** Marks a function as asynchronous (a coroutine). Use with await.


**Example:**
```python
async def coro():
    return 42
# run with asyncio.run(coro())
```

---

### await

**Explanation:** Used in async functions to suspend execution until the awaited awaitable completes.


**Example:**
```python
import asyncio
async def sleepy():
    await asyncio.sleep(0.1)
    return 'done'
```

---

### break

**Explanation:** Exits the nearest enclosing loop immediately.


**Example:**
```python
for i in range(5):
    if i==2:
        break
    print(i)  # prints 0,1
```

---

### case

**Explanation:** Used inside match blocks as a branch.


**Example:**
```python
See the 'match' example above.
```

---

### class

**Explanation:** Defines a new class (custom type).


**Example:**
```python
class Person:
    def __init__(self,name):
        self.name = name
p = Person('Ani')
print(p.name)
```

---

### continue

**Explanation:** Skips the rest of the current loop iteration and continues with next iteration.


**Example:**
```python
for i in range(4):
    if i%2==0:
        continue
    print(i)  # prints 1,3
```

---

### def

**Explanation:** Defines a function.


**Example:**
```python
def add(a,b):
    return a+b
print(add(2,3))
```

---

### del

**Explanation:** Deletes a name binding or an item from a container.


**Example:**
```python
x = [1,2,3]
del x[1]
print(x)  # [1,3]
```

---

### elif

**Explanation:** Used after if for additional conditional checks.


**Example:**
```python
x=0
if x>0:
    print('positive')
elif x==0:
    print('zero')
else:
    print('negative')
```

---

### else

**Explanation:** Fallback block used after if, for/while (loop else), or try (try/except/else).


**Example:**
```python
for i in range(3):
    pass
else:
    print('loop finished')
```

---

### except

**Explanation:** Catches exceptions thrown in a try block.


**Example:**
```python
try:
    1/0
except ZeroDivisionError:
    print('division by zero')
```

---

### False

**Explanation:** Boolean constant representing false. Used in logical expressions and conditionals.


**Example:**
```python
if not False:
    print('False is falsy')
```

---

### finally

**Explanation:** Always-run block after try/except, useful for cleanup.


**Example:**
```python
try:
    x=1
finally:
    print('always runs')
```

---

### for

**Explanation:** Iterates over items of an iterable (lists, tuples, strings, generators...).


**Example:**
```python
for x in [10,20]:
    print(x)
```

---

### from

**Explanation:** Imports specific attributes from a module.


**Example:**
```python
from math import sqrt
print(sqrt(9))
```

---

### global

**Explanation:** Declares that a variable inside a function refers to a module-level variable.


**Example:**
```python
x=0

def inc():
    global x
    x += 1
inc(); print(x)
```

---

### if

**Explanation:** Conditional branching based on truthiness of an expression.


**Example:**
```python
if True:
    print('yes')
```

---

### import

**Explanation:** Imports a module or package.


**Example:**
```python
import math
print(math.pi)
```

---

### in

**Explanation:** Membership test (x in container) and used in for loops.


**Example:**
```python
print(2 in [1,2,3])  # True
for ch in 'hi':
    print(ch)
```

---

### is

**Explanation:** Identity test: True if two references point to the same object.


**Example:**
```python
a=[1]
b=a
print(a is b)  # True
```

---

### lambda

**Explanation:** Creates a small anonymous function (single expression).


**Example:**
```python
f = lambda x: x*2
print(f(3))
```

---

### match

**Explanation:** Structural pattern matching (Python 3.10+). Similar to switch/case but more powerful.


**Example:**
```python
def show(x):
    match x:
        case 0:
            return 'zero'
        case [a,b]:
            return f'list of two: {a},{b}'
print(show(0))
print(show([1,2]))
```

---

### None

**Explanation:** Special constant representing absence of a value. Commonly used as default return value for functions that don't return anything.


**Example:**
```python
def f():
    return None
print(f())  # prints None
```

---

### nonlocal

**Explanation:** Declares that a variable refers to the nearest enclosing (non-global) scope in nested functions.


**Example:**
```python
def outer():
    x=0
    def inner():
        nonlocal x
        x+=1
    inner()
    return x
print(outer())
```

---

### not

**Explanation:** Logical NOT operator; inverts truthiness.


**Example:**
```python
print(not True)  # False
```

---

### or

**Explanation:** Logical OR operator; returns first truthy operand or the last operand.


**Example:**
```python
print(False or 'ok')  # 'ok'
```

---

### pass

**Explanation:** No-op placeholder used where a statement is syntactically required.


**Example:**
```python
def todo():
    pass
```

---

### raise

**Explanation:** Raises an exception intentionally.


**Example:**
```python
def f():
    raise ValueError('bad')
# f()  # would raise ValueError
```

---

### return

**Explanation:** Exits a function and optionally returns a value.


**Example:**
```python
def f():
    return 5
print(f())
```

---

### True

**Explanation:** Boolean constant representing true. Used in logical expressions.


**Example:**
```python
if True:
    print('This runs')
```

---

### try

**Explanation:** Starts an exception handling block used with except/finally/else.


**Example:**
```python
try:
    x=1/0
except Exception:
    print('error')
```

---

### while

**Explanation:** Repeats a block while a condition is true.


**Example:**
```python
i=0
while i<2:
    print(i)
    i+=1
```

---

### with

**Explanation:** Context manager protocol; ensures proper setup and teardown (e.g., open files).


**Example:**
```python
with open(__file__,'r') as f:
    print('file opened for reading')
```

---

### yield

**Explanation:** Used in generator functions to yield values lazily.


**Example:**
```python
def gen():
    yield 1
    yield 2
for v in gen():
    print(v)
```

---

 