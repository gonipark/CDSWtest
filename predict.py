import pickle
import argparse

model = pickle.load(open('model.pkl', 'rb'))

def predict():
  #make csv
#  test=pd.read_csv('preprocessed_test')
#  test_data = test.drop("PassengerId", axis=1).copy()
#  prediction = model.predict(test_data)
#
#  submission = pd.DataFrame({
#        "PassengerId": test["PassengerId"],
#        "Survived": prediction
#    })
#
#  submission.to_csv('submission.csv', index=False)


  args=parser.parse_args()
  result = model.predict([[ args.Pclass,  args.Sex , args.Age,  args.Fare,  args.Cabin , args.Embarked , args.Title]])
  return result

parser=argparse.ArgumentParser()
parser.add_argument('--Pclass',default=3)
parser.add_argument('-Sex',default=0)
parser.add_argument('--Age',default=1.0)
parser.add_argument('--Fare',default=0.0)
parser.add_argument('--Cabin',default=2.0)
parser.add_argument('--Embarked',default=0)
parser.add_argument('--Title',default=0)

print(predict())
