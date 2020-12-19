from recognisePerson import recognisePerson
from registerNewPerson import registerNewPerson

choice = int(input("Enter Choice: \n1. Register new person. \n2. Recognise person.\n"))
if choice == 1:
	registerNewPerson()
elif choice == 2:
	print(recognisePerson())