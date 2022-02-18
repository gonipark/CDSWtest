import pickle
import argparse

model = pickle.load(open('model.pkl', 'rb'))


def predict(args):
 
  #petal_length = float(args.get('petal_length'))
#  result = model.predict([[petal_length]])
 
  Pclass=int(args.get('Pclass'))
  Sex=int(args.get('Sex'))
  Age=float(args.get('Age'))
  Fare=float(args.get('Fare'))
  Cabin=float(args.get('Cabin'))
  Embarked=int(args.get('Embarked'))
  Title=int(args.get('Title'))

  result = model.predict([[ Pclass, Sex ,Age, Fare,Cabin , Embarked ,Title]])

  
  return result



