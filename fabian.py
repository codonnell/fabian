from flask import Flask, request
from flask_restful import Resource, Api
import pickle
import numpy as np

app = Flask(__name__)
app.config.from_object('fabian.cfg')
app.config.from_envvar('FABIAN_SETTINGS')
api = Api(app)

with open(app.config['CLASSIFIER'],'rb') as f:
    clf = pickle.load(f)

class Difficulty(Resource):
    def get(self):
        data = request.get_json()['data']
        X = np.array(data).astype(np.float)
        y = clf.predict_proba(X)
        return {'data': [probs for probs in y.tolist()]}

api.add_resource(Difficulty, '/difficulty/')

if __name__ == "__main__":
    app.run(debug=app.config['DEBUG'])
