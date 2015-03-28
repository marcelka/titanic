import csv as csv
import numpy as np
import random
from collections import namedtuple

Feature = namedtuple("Feature", ["text", "length", "type", "fn"])
Function = namedtuple("Function", ["text", "fn"])

functions = { #{{{
    ('n', 'n', 'n') : [
        Function('+', lambda x,y: x+y), 
        Function('*', lambda x,y: x*y), 
        Function('-', lambda x,y: x-y),
    ],

    ('n', 'n', 'b'): [
        Function('>', lambda x,y: x>y),
        Function('<', lambda x,y: x<y),
        Function('==', lambda x,y: x==y),
        Function('!=', lambda x,y: x!=y),
    ],

    ('b', 'b', 'b'): [
        Function('and', lambda x,y: x and y),
        Function('or', lambda x,y: x or y),
        Function('==', lambda x,y: x == y),
        Function('!=', lambda x,y: x != y),
    ],

    ('n', 'n'): [
        Function('-', lambda x: -x),
    ],

    ('n', 'b'): [
        Function('>', lambda x: x>0),
        Function('<', lambda x: x<0),
        Function('==', lambda x: x==0),
    ],

    ('b', 'b'): [
        Function('not', lambda x: not x),
    ],

    ('b', 'n'): [
        Function('int', lambda x: 1 if x else 0),
    ],

    ('b', 'n', 'n', 'n'): [
        Function('if', lambda x,y,z: y if x else z),
    ],
} #}}}

def get_feature(pool):
    """
       returns Feature
    """
    def get_num(pool):
        while True:
            if random.random()<0.1:
                res = random.randint(0,1000)
                return Feature(str(res), 1, 'n', lambda p: res)
            res = random.choice(pool)
            if res.type == 'n':
                return res

    def get_bool(pool):
        while True:
            res = random.choice(pool)
            if res.type == 'b':
                return res

    get_pool = {'n': get_num, 'b': get_bool}
    c = random.choice(list(functions.keys()))
    fn = random.choice(functions[c])
    args_types = c[:-1]
    res_type = c[-1]
    args = [get_pool[arg_type](pool) for arg_type in args_types]
    return Feature('%s(%s)'%(fn.text, ', '.join(f.text for f in args)), 
                   1 + sum(f.length for f in args),
                   res_type, 
                   lambda p: fn(*[arg.fn(p) for arg in args]))
feature_count = 20

def feature_set_gen(score):
    result = [Feature('1', 1, 'n', lambda p: 1)]*feature_count
    pool = []
    base_pool = [
        Feature('is_male', 1, 'b', lambda p: p.is_male),
        Feature('price', 1, 'n', lambda p: p.price),
    ]
    while True:
        npool = list(pool)
        npool.extend(base_pool)
        new_feature = get_feature(npool)
        pool.append(new_feature)
        if len(pool) > 1000:
            pool.pop(0)
        if new_feature.type == 'n':
            while True:
                i = random.randint(0, len(result) - 1)
                if i>0:
                    break
            new_result = list(result)
            new_result[i] = new_feature
            if score(new_result) > score(result):
                result = new_result
                yield result

count = 0
for feature_set in feature_set_gen(lambda feature_set: random.random()):
    for f in feature_set:
        print(f.text)
        print('\n next feature \n')
    print('\n\n')
    count += 1
    if count == 100:
        break
    
    
class Passenger:
    """
    PassengerId: 891
    Survived: 0 -> bool
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

def save_csv(strategy, filename, passengers):
    with open('results/%s' % filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',',)
        writer.writerow(['PassengerId', 'Survived'])
        for p in passengers:
            writer.writerow([p.PassengerId, int(strategy(p))])

def cheaty(passenger):
    return passenger.Survived

def gender(passenger):
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
    if p.Fare is None: p.Fare = fares[p.Pclass]
    if p.Age is None: p = age

print('cheaty', test_strategy(cheaty, passengers))
print('gender', test_strategy(gender, passengers))

save_csv(gender, "my-gender-train.csv", passengers)

with open('data/test.csv') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    header = next(reader)
    passengers = [Passenger(header, row) for row in reader]
    save_csv(gender, "my-gender-test.csv", passengers)
    print(to_array(passengers))


#from sklearn.ensemble import RandomForestClassifier
#Forest = RandomForestClassifier(n_estimators = 100)
#Forest = Forest.fit(train_data[0::,1::],train_data[0::,0])
#Output = Forest.predict(test_data)
