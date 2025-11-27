# Match case is alternate for if - elif - else

color = input("Enter color: ")

match color:
    case "Green":
        print("Go")
    case "Red":
        print("Stop")
    case "Yellow":
        print("Look")
    case _:
        print("Wrong color")