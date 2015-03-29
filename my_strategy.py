import csv as csv
import numpy as np
import random
from datetime import datetime
from collections import namedtuple
from sklearn import svm
import math

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
        Function('>0', lambda x: x>0),
        Function('<0', lambda x: x<0),
        Function('==0', lambda x: x==0),
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

pax_properties = [ #{{{
  ("pclass", 'n', lambda p: p.Pclass),
  ("has_posh_name", 'b', lambda p: any([part in p.Name for part in ["van "]])),
  ("name_parts", 'n', lambda p: len(p.Name.split(" "))),
  ("is_male", 'b', lambda p: p.Sex == "male"),
  ("age", 'n', lambda p: p.Age),
  ("has_age", 'b', lambda p: p.hasAge),
  ("relatives", 'n', lambda p: p.SibSp),
  ("parch", 'n', lambda p: p.Parch),
  ("fare", 'n', lambda p: p.Fare),
  ("embarked_s", 'b', lambda p: p.Embarked == "S"),
  ("embarked_q", 'b', lambda p: p.Embarked == "Q"),
  ("embarked_c", 'b', lambda p: p.Embarked == "C"),
  ("has_cabin", 'b', lambda p: p.Cabin != ""),
  ("cabin_count", 'n', lambda p: 0 if p.Cabin == "" else len(p.Cabin.split(" "))),
  ("survived", 'b', lambda p: p.Survived),
] #}}}

def getname(name): return lambda p: getattr(p, name)
take_only = ['has_age', 'age', 'has_fare', 'fare', 'is_male', 'relatives']
base_pool = [Feature(name, 1, typee, getname(name)) 
        #for (name, typee, factory) in pax_properties if name not in ['id', 'survived']]
        for (name, typee, factory) in pax_properties if name in take_only]

feature_count = 20
pool_size = 50
length_penalty_coef = 1000000
max_feature_length = 10

logfile = open('log.txt', 'w', 5)

def log(message):
    logfile.write(message)
    logfile.write('\n')
    logfile.flush()

def get_feature(pool): #{{{
    """
       returns Feature
    """
    def get_num(pool):
        while True:
            if random.random()<0.1:
                res = random.randint(0,10)*(10**random.randint(0,3))
                return Feature(str(res), 1, 'n', lambda p: res)
            res = random.choice(pool)
            if res.type == 'n':
                return res

    def get_bool(pool):
        while True:
            res = random.choice(pool)
            if res.type == 'b':
                return res

    if random.random()<0.2:
        return get_num(pool)
    get_pool = {'n': get_num, 'b': get_bool}
    c = random.choice(list(functions.keys()))
    fn = random.choice(functions[c])
    args_types = c[:-1]
    res_type = c[-1]
    args = [get_pool[arg_type](pool) for arg_type in args_types]
    return Feature('%s(%s)'%(fn.text, ', '.join(f.text for f in args)), 
                   1 + sum(f.length for f in args),
                   res_type, 
                   lambda p: fn.fn(*[arg.fn(p) for arg in args]))
#}}}

def feature_set_gen(score_fn, pax): #{{{
    result = [Feature('1', 1, 'n', lambda p: 1)]*feature_count
    pool = []
    score = score_fn(result)
    yield (score, result)
    while True:
        log('main loop')
        npool = list(pool)
        npool.extend(base_pool)
        
        avg_len = sum(f.length for f in npool)/len(npool)
        log('avg pool len: %s'%str(avg_len))

        new_feature = get_feature(npool)
        try:
            is_good = True
            for p in pax:
                if new_feature.fn(p) == None:
                    is_good = False
        except TypeError:
            is_good = False
        log('got new feature, good: %s'%str(is_good))
        if random.random() > (0.9 ** new_feature.length):
            log('new feature too long')
            continue
        if is_good or random.random()<0.5:
            pool.append(new_feature)
            if len(pool) > pool_size:
                pool.pop(0)
        if not is_good:
            continue
        if new_feature.type == 'n':
            while True:
                i = random.randint(0, len(result) - 1)
                if i>0:
                    break
            new_result = list(result)
            new_result[i] = new_feature
            new_score = score_fn(new_result)
            if new_score > score:
                score = new_score
                result = new_result
                yield (score, result)
            else:
                log('new feature does not help')
        else:
            log('new feature is not num')
#}}}

def score(fset, passengers): #{{{ score probabilistic

    classifiers = {0: set(), 1: set()}
    result = [1 if p.survived else 0 for p in passengers]

    def get_classifier(feature, mean, std):
        def res(p):
            v = feature.fn(p)
            return math.exp(-abs(mean-v)/max(std, 1e-6))
        return res

    def classify(p):
        vs = [(classifier(p) for classifier in classifiers[outcome]) for outcome in [0, 1]]
        rs = [2, 1]
        for outcome in [0, 1]:
            r = rs[outcome]
            for v in vs[outcome]:
                r *= v
            rs[outcome]=r
        return 0 if rs[0]>rs[1] else 1

    for feature in fset:
        for outcome in [0,1]:
            data = [feature.fn(p) for p in passengers if p.survived == (outcome == 1)]
            mean = np.mean(data)
            std = np.std(data)
            classifiers[outcome].add(get_classifier(feature, mean, std))

   
    predict = [classify(p) for p in passengers]
    accuracy = sum(1 if p==r else 0 for p,r in zip(predict, result))/len(passengers)
    length_penalty = sum(f.length for f in fset)/length_penalty_coef
    return accuracy - length_penalty
            
# }}}
    
#def score(fset, passengers): #{{{ score using SVM
#    X = [[feature.fn(p) for feature in fset] for p in passengers]
#    result = [1 if p.survived else 0 for p in passengers]
#    before_time = datetime.now()
#    clf = svm.SVC(kernel='rbf', degree=3, max_iter=1000000)
#    #clf = svm.SVC(kernel='linear', degree=3, max_iter=1000000)
#    #clf = svm.SVR(kernel='linear', degree=1, cache_size=1, max_iter=100000)
#    log('gonna fit the model')
#    clf.fit(X, result)
#    after_time = datetime.now()
#    log('fitting model took ' + str(after_time - before_time))
#    _predict = clf.predict(X)
#    predict = [round(p) for p in _predict]
#    accuracy = sum(1 if p==r else 0 for p,r in zip(predict, result))/len(result)
#    length_penalty = sum(f.length for f in fset)/length_penalty_coef
#    return accuracy - length_penalty
# }}}
    

# {{{ data preprocessing; computes pax
class Passenger: #{{{
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
                self.Survived = row[i] == '1'
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
#}}}

def read_passengers(filename): #{{{
    with open(filename) as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        header = next(reader)
        passengers = [Passenger(header, row) for row in reader]
    return passengers
#}}}

def fill_missing_data(passengers): #{{{
    # Age and Fare is incomplete
    #classes = {p.Pclass for p in passengers}
    #fares = {c: np.median([p.Fare for p in passengers if p.Pclass == c and
    #         p.Fare is not None]) for c in classes}
    #age = np.median([p.Age for p in passengers if p.Age is not None])
    for p in passengers:
        if p.Fare is None: 
            #p.Fare = fares[p.Pclass]
            #p.Fare = 0
            p.hasFare = False
        else:
            p.hasFare = True
        if p.Age is None: 
            #p.Age = age
            #p.Age = 0
            p.hasAge = False
        else:
            p.hasAge = True
#}}}

def save_csv(strategy, filename, passengers): #{{{
    with open('results/%s' % filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',',)
        writer.writerow(['PassengerId', 'Survived'])
        for p in passengers:
            writer.writerow([p.PassengerId, int(strategy(p))])
#}}}

NormPax = namedtuple("NormPax", [name for (name,_,_) in pax_properties])

def normalized(passenger):
    vals = dict(list(map(lambda i: (i[0],i[2](passenger)), pax_properties)))
    return NormPax(**vals)

passengers = read_passengers('data/train.csv')
fill_missing_data(passengers)
pax = [normalized(p) for p in passengers]
#}}}

def pretty_format_fs(fset):
    return '\n\n'.join(feature.text for feature in fset)

run = 1
if run == 1:
    for sc, fset in feature_set_gen(lambda fset: score(fset, pax), pax):
        print(pretty_format_fs(fset))
        print(sc)
        print('-----')


def to_int_feature(f):
    if f.type=='n':
        return f
    else:
        return Feature(f.text, f.length, 'n', lambda p: 1 if f.fn(p) else 0)

one_f= Feature('one', 1, 'n', lambda p: 1)
gender_f = Feature('gender', 1, 'n', lambda p: 1 if p.is_male else 0)
fare_f = Feature('fare', 1, 'n', lambda p: p.fare)
age_f = Feature('age', 1, 'n', lambda p: p.age)
survived_f = Feature('survived', 1, 'n', lambda p: 1 if p.survived else 0)
basic = [to_int_feature(f) for f in base_pool]
basic.append(one_f)
print('all', score(basic, pax))
print('o', score([one_f], pax))
print('g', score([one_f, gender_f], pax))
print('f', score([one_f, fare_f], pax))
print('a', score([one_f, age_f], pax))
print('s', score([one_f, survived_f], pax))
print('gf', score([one_f, gender_f, fare_f], pax))
print('gfa', score([one_f, gender_f, fare_f, age_f], pax))
print('gfas', score([one_f, gender_f, fare_f, age_f, survived_f], pax))



#{{{ poor mans git
#def test_strategy(strategy, passengers):
#    correct = [p for p in passengers if strategy(p) == p.Survived]
#    return len(correct) / len(passengers)
#
#def cheaty(passenger): return passenger.Survived
#def gender(passenger): return passenger.Sex == 'female'

#print('cheaty', test_strategy(cheaty, passengers))
#print('gender', test_strategy(gender, passengers))
#save_csv(gender, "my-gender-test.csv", passengers)
#def to_array(passengers):
#    return np.array([p.row() for p in passengers])
#print(to_array(passengers))
#for i in range(20):
#    print(normalized(passengers[i]))
#for k,v in domains(passengers).items():
#        print(k, '\n', v, '\n')
#save_csv(gender, "my-gender-train.csv", passengers)
#passengers = read_passengers('data/test.csv')
#fill_missing_data(passengers)

#count = 0
#for feature_set in feature_set_gen(lambda feature_set: random.random()):
#    for f in feature_set:
#        print(f.text)
#        print('\n next feature \n')
#    print('\n\n')
#    count += 1
#    if count == 100:
#        break

#def domains(passengers): 
#    domains = {}
#    for h in passengers[0].header:
#        domains[h] = set([getattr(p, h) for p in passengers])
#    return domains

#}}}
