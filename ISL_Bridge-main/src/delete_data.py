import csv

def main ():

    sign = input("Input signs you want to delete (separated by commas) : ")

    signs = [s.strip().upper() for s in set(sign.split(","))]
   
    with open("../data/landmarks_own.csv", "r") as file:
        reader = csv.reader(file)
        list = []
        for row in reader :
            list.append(row)
    
    updated_list = []
    for row in list:
        if row and  row[-1] not in signs:
            updated_list.append(row)
    
    with open("../data/landmarks_own.csv", "w", newline="") as file:
        writer = csv.writer(file)
        for row in updated_list:
            writer.writerow(row)


if __name__ == "__main__":
    main()