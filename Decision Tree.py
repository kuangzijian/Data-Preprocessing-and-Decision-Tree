import pandas as pd
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.tree import export_graphviz
import six
import sys
sys.modules['sklearn.externals.six'] = six
import pydotplus

feature_cols = ['A', 'B', 'C', 'D']
X_train = [[0,1,0,0],[1,1,0,0],[1,1,0,0],[1,1,0,0],[1,1,0,0],[0,1,0,0],[0,1,0,0],[1,1,0,0],[1,1,1,0],[1,1,1,1],[1,1,1,1],[1,0,0,0],[1,0,1,1],[1,0,1,0],[1,1,1,1],[0,0,0,0]]
y_train = [0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1]

# Create Decision Tree classifer object
clf = DecisionTreeClassifier()

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

dot_data = six.StringIO()
export_graphviz(clf, out_file=dot_data,
                filled=True, rounded=True,
                special_characters=True,feature_names = feature_cols,class_names=['0','1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('diabetes.png')
#(graph.create_png())
