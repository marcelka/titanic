import csv as csv
import numpy as np

class Passenger:
    """
    PassengerId: 891
    Survived: 0
    Pclass: 3
    Name: Dooley, Mr. Patrick
    Sex: male
    Age: 32
    SibSp: 0
    Parch: 0
    Ticket: 370376
    Fare: 7.75
    Cabin: C148
    Embarked: Q
    """
    def __init__(self, header, row):
        self.header = header
        for i in range(len(header)):
            if header[i] == 'Survived':
                self.Survived = row[i] == 1
            elif header[i] == 'Fare':
                try:
                    self.Fare = float(row[i])
                except ValueError:
                    self.Fare = None
            elif header[i] in ['PassengerId', 'Pclass', 'Age', 'SibSp',
                               'Parch']:
                try:
                    setattr(self, header[i], int(row[i]))
                except ValueError:
                    setattr(self, header[i], None)
            else:
                setattr(self, header[i], row[i])

    def row(self):
        return [getattr(self, attr) for attr in self.header]


def to_array(passengers):
    return np.array([p.row() for p in passengers])


def test_strategy(strategy, passengers):
    correct = [p for p in passengers if strategy(p) == p.Survived]
    return len(correct) / len(passengers)


def get_csv(strategy, passengers):
    with open('results/my_strategy.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',',)
        writer.writerow(['PassengerId', 'Survived'])
        for p in passengers:
            writer.writerow([p.PassengerId, int(strategy(p))])


def cheaty(passenger):
    return passenger.Survived

def model(passenger):
    return passenger.Sex == 'female'

passengers = []
with open('data/train.csv') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    header = next(reader)
    passengers = [Passenger(header, row) for row in reader]

# fill missing data (Age, Fare)
classes = {p.Pclass for p in passengers}
fares = {c: np.median([p.Fare for p in passengers if p.Pclass == c and
         p.Fare is not None]) for c in classes}
age = np.median([p.Age for p in passengers if p.Age is not None])
for p in passengers:
    if p.Fare is None:
        p.Fare = fares[p.Pclass]
    if p.Age is None:
        p = age

print('cheaty', test_strategy(cheaty, passengers))
print('model', test_strategy(model, passengers))
get_csv(model, passengers)

with open('data/test.csv') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    header = next(reader)
    passengers = [Passenger(header, row) for row in reader]
    get_csv(model, passengers)
    print(to_array(passengers))


#from sklearn.ensemble import RandomForestClassifier
#Forest = RandomForestClassifier(n_estimators = 100)
#Forest = Forest.fit(train_data[0::,1::],train_data[0::,0])
#Output = Forest.predict(test_data)
