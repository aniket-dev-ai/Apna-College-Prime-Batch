#  if , elif , else  => Just highly focus of indentation i.e tabs

   
    
#  check between child , teenager , adult

# age = int(input("Pleasse enter your age : "))

# if age < 13 :
#     print("You are still a child")
# elif age<=18 and age >= 13:
#     print("You are a teenager now")
# else:
#     print("Congrats You are an adult")

userName = input("Please enter UserName : ")
passWord = input("Please enter password : ")

if(userName=="admin" and passWord=="pass"):
    print("Login Successfull")
elif (userName!="admin"):
    print("You are not a admin")
else:
    print("Wrong password")