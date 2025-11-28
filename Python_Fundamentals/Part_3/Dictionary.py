# Dictionary is key value pair just like a json

info ={
    "name":"Aniket",
    "cgpa":9.9,
    "subjects":["maths","science"]
    }

print(info)
print(type(info))
print(info["cgpa"])

# it is mutable
info["cgpa"] = 9.6
print(info["cgpa"])

