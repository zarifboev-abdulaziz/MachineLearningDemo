# jupyter notebook - to start and go to your files

# import pandas as pd
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
#
# music_data = pd.read_csv('music.csv')
# X = music_data.drop(columns = ['genre'])
# y = music_data['genre']
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
#
# model = DecisionTreeClassifier()
# model.fit(X_train, y_train)
# predictions = model.predict(X_test)
#
# score = accuracy_score(y_test, predictions)
# score

# ///////////////////////////////////////////////////////////// -- extract .joblib file using joblib.dump()
# import pandas as pd
# from sklearn.tree import DecisionTreeClassifier
# # from sklearn.externals import joblib
# import joblib
#
#  music_data = pd.read_csv('music.csv')
#  X = music_data.drop(columns = ['genre'])
#  y = music_data['genre']
#
#  model = DecisionTreeClassifier()
#  model.fit(X, y)
#
# joblib.dump(model, 'music-recommender.joblib')
#
# #predictions = model.predict([ [21, 1], [22, 0] ])

# //////////////////////////////////////////////////////////// -- loading .joblib file using load() method
# import pandas as pd
# from sklearn.tree import DecisionTreeClassifier
# # from sklearn.externals import joblib
# import joblib
#
# # music_data = pd.read_csv('music.csv')
# # X = music_data.drop(columns = ['genre'])
# # y = music_data['genre']
#
# # model = DecisionTreeClassifier()
# # model.fit(X, y)
#
# model = joblib.load('music-recommender.joblib')
#
# predictions = model.predict([ [21, 1], [22, 0] ])
# predictions

# ///////////////////////////////////////////////////////// -- exrtaxting .dot file to visualize
#use Graphviz extension in vscode to visualize

# import pandas as pd
# from sklearn.tree import DecisionTreeClassifier
# from sklearn import tree
#
#
# music_data = pd.read_csv('music.csv')
# X = music_data.drop(columns = ['genre'])
# y = music_data['genre']
#
# model = DecisionTreeClassifier()
# model.fit(X, y)
#
# tree.export_graphviz(model, out_file='music-recommender.dot',
#                      feature_names=['age', 'gender'],
#                     class_names=sorted(y.unique()),
#                     label='all',
#                     rounded=True,
#                     filled=True)
