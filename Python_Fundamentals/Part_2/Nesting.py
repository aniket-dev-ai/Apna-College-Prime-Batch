age = int(input("Enter your age: "))
income = int(input("Enter your monthly income: "))
credit_score = int(input("Enter your credit score: "))

if age >= 18:
    if income >= 25000:
        if credit_score >= 700:
            print("Loan Approved: You qualify for the premium loan.")
        else:
            print("Loan Approved: But only for a basic plan due to lower credit score.")
    else:
        print("Loan Rejected: Income too low.")
else:
    print("Loan Rejected: You must be 18 or older.")
