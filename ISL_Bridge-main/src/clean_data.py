import csv
from collections import Counter

def main ():

    single = ['C','L','O','U','V','0','1','2','3','4','5','6','7','8','9','SPACE']   
    double = ['A','B','D','E','F','G','I','K','M','N','P','Q','R','S','T','W','X','Z']

    with open("../data/landmarks_own.csv", "r") as file:
        reader = csv.reader(file)
        list = []
        labels = []
        for row in reader :
            list.append(row)
            labels.append(row[-1])
    
    print("Before cleaning : ",len(list)) 

    updated_list = []
    updated_list.append(list[0])
    for row in list:
        if row[-1] != 'sign':
            if len(row) == 127:
                #non empty rows 
                if not all(float(x) == 0 for x in row[0:126]):
                    if row[-1] in single:
                        updated_list.append(row)
                    elif not all(float(x) == 0 for x in row[0:63]) and not all(float(x) == 0 for x in row[63:126]) and row[-1] in double:
                        updated_list.append(row)
    
    print("After cleaning : ",len(updated_list)) 

    with open("../data/landmarks_own.csv", "w", newline="") as file:
        writer = csv.writer(file)
        for row in updated_list:
            writer.writerow(row)

    labels_dict = Counter(labels)

    for _ in labels_dict.keys() :
        print(f"{_} : {labels_dict[_]}")

if __name__ == "__main__":
    main()
