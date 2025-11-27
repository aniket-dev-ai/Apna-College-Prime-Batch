name = "Aniket"

for val in name:
    print(val)
    
if 'e' in name:
    print("yes")
    
###############################

for i in range(5):
    print("Hello world "+ repr(i+1))
    
check = "qwsadcfvgbhjuasdfghuiokjhbgfdswertyuiokjhgfdxcvbnm"
i = 0
for ch in check:
    if ch in "aeiou":
        i += 1
        
print(i)

