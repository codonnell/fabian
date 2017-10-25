# fabian

Flask endpoint providing difficulty data with some utility code for creating classifiers.

## Deployment

Install the python packages `flask-api` and `scikit-learn` using `pip`, preferably in a virtualenv. Create a scikit-learn classifier with the `predict-proba` method, like `SVC`, `LogisticRegression`, `RandomForestClassifier`, etc. Pickle it and store it in this directory. By default, `fabian` will look for this classifier to be stored as `classifier.pkl`, but you can customize this by creating a new config file whose name you store in the environment variable `FABIAN_SETTINGS`.

With the proper packages installed and classifier in place, start the server with ```python fabian.py```. For production use, be sure to set `DEBUG=False` in the config file whose name is stored in the environment variable `FABIAN_SETTINGS`.
