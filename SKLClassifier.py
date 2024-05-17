import collections
import gc
import logging
import random
from tqdm import tqdm
import numpy as np
from sklearn import neural_network, ensemble
from sklearn.metrics import PrecisionRecallDisplay, confusion_matrix, RocCurveDisplay, ConfusionMatrixDisplay
from imblearn.over_sampling import BorderlineSMOTE, RandomOverSampler, SMOTE
from imblearn.under_sampling import TomekLinks, NeighbourhoodCleaningRule
import joblib
import ToxProtParse
import pickle
import matplotlib.pyplot as plt
import pandas as pd
from xgboost import XGBClassifier, XGBRFClassifier

plt.rcParams.update({'font.size': 18})
random.seed(123)
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def lmap(imp,cut):
    if imp>cut:
        return 1
    else:
        return 0





###Training
"""with open("100_train_embedded.pickle","rb") as tp:
    training_data = pickle.load(tp)
#with open("100_test_embedded.pickle", "rb") as td:
#    test_data = pickle.load(td)
#    training_data[0]=np.concatenate((training_data[0], test_data[0]))
#    training_data[1]=np.concatenate((training_data[1], test_data[1]))
print("Training started")
model = neural_network.MLPClassifier(solver="adam", verbose=True, early_stopping=True,
                                     hidden_layer_sizes=(500, 250), alpha=1e-8, tol=0.0001)
model.fit(training_data[0], training_data[1])
joblib.dump(model, "MLP_default_100.joblib")
print("Training ended")
del training_data
gc.collect()"""


###Testing
"""with open("ToxPredN.pickle", "rb") as tp:
    test_data: list = pickle.load(tp)
with open("ToxPredP.pickle", "rb") as tp:
    test_data2 = pickle.load(tp)
test_data[0] += test_data2[0]
test_data[1] = np.concatenate((test_data[1], test_data2[1]))
test_data[2] += test_data2[2]"""
###OR"""
"""with open("diffseqs_100.pickle", "rb") as tp:
    test_data = pickle.load(tp)"""
"""model: neural_network.MLPClassifier = joblib.load("MLP_full.joblib")
print("Predicting")
model.score(test_data[0], test_data[1])
predicted_vector = model.predict_proba(test_data[0])[:,1]
pr_cl = model.predict(test_data[0])
cm = confusion_matrix(test_data[1],pr_cl, labels=model.classes_)
MCC = (cm[0][0]*cm[1][1]-cm[0][1]*cm[1][0])\
      /np.sqrt((cm[1][1]+cm[0][1])*(cm[1][1]+cm[1][0])*(cm[0][0]+cm[0][1])*(cm[0][0]+cm[1][0]))
ConfusionMatrixDisplay.from_predictions(test_data[1], pr_cl, colorbar=False, display_labels=model.classes_,
                                        cmap="Blues", normalize="true", values_format=".4f")
plt.show()"""

"""for p, r, a in zip(pr_cl, test_data[1], test_data[2]):
    if p == 1 and r == 0:
        acc = a.split("|")[1]
        ToxProtParse.retrieveAccessionFromUniprot(acc, "temp")"""

"""print(cm)
print(MCC)
pr_cl001=[lmap(pred,0.00001) for pred in predicted_vector]
cm001=confusion_matrix(test_data[1],pr_cl001, labels=model.classes_)
ConfusionMatrixDisplay.from_predictions(test_data[1],pr_cl001, colorbar=False, display_labels=model.classes_,
                                        cmap="Blues", normalize="true", values_format=".4f")
plt.show()
MCC1 = (cm001[0][0]*cm001[1][1]-cm001[0][1]*cm001[1][0])\
       /np.sqrt((cm001[1][1]+cm001[0][1])*(cm001[1][1]+cm001[1][0])*(cm001[0][0]+cm001[0][1])*(cm001[0][0]+cm001[1][0]))
print(cm001)
print(MCC1)
RocCurveDisplay.from_predictions(test_data[1],predicted_vector)
plt.plot([0,1], [0,1], linestyle="--", label="No skill")
plt.legend()
plt.show()
PrecisionRecallDisplay.from_predictions(test_data[1],predicted_vector)
noskill=collections.Counter(test_data[1])[1]/len(test_data[1])
plt.plot([0,1],[noskill,noskill],linestyle="--",label="No skill")
plt.legend()
plt.show()"""


###UnderSampling
"""
with open("pickled_embedded_train.pickle","rb") as tp:
    training_data = pickle.load(tp)
XRes, YRes = NeighbourhoodCleaningRule(n_neighbors=5).fit_resample(training_data[0],training_data[1])
print(collections.Counter(YRes))
"""


###OverSampling
"""
with open("100_test_embedded.pickle","rb") as tp:
    training_data = pickle.load(tp)
with open("100_train_embedded.pickle","rb") as tp:
    test_data = pickle.load(tp)

alldata=[training_data[0]+test_data[0],training_data[1]+test_data[1]]
#alldata=[training_data[0],training_data[1]]
XRes, YRes = RandomOverSampler(shrinkage=0.25).fit_resample(alldata[0],alldata[1])
print(collections.Counter(YRes))
"""

###XGBoost
"""
with open("pickled_embedded_train.pickle", "rb") as tp:
    training_data = pickle.load(tp)
model = ensemble.HistGradientBoostingClassifier(verbose=True, early_stopping=True, class_weight={0: 1, 1: 70},
                                                max_iter=300)
model.fit(training_data[0], training_data[1])
with open("pickled_embedded_test.pickle", "rb") as tp:
    test_data = pickle.load(tp)
model.score(test_data[0], test_data[1])
pr_cl = model.predict(test_data[0])
cm = confusion_matrix(test_data[1], pr_cl).ravel()
print(cm)"""

###Figs
###ToxPred3CM
"""#difftest = pd.read_csv("ToxinPred3_diffseqs_filt_predictions.csv")
postest = pd.read_csv("ToxinPred3_Toxin_predictions.csv")
negtest = pd.read_csv("ToxinPred3_Non-toxin_predictions.csv")
nlist = list(negtest["ML Score"])
plist = list(postest["ML Score"])
#difflist = list(difftest["ML Score"])
preds = [nlist + plist, np.concatenate((np.zeros(len(nlist)), np.ones(len(plist))))]
#dpreds = [difflist, np.ones(len(difflist))]
RocCurveDisplay.from_predictions(preds[1], preds[0])
plt.plot([0,1], [0,1], linestyle="--", label="No skill")
plt.legend()
plt.show()
#PrecisionRecallDisplay.from_predictions(preds[1],dpreds[0])
#noskill=collections.Counter(dpreds[1])[1]/len(dpreds[1])
#plt.plot([0,1],[noskill,noskill],linestyle="--",label="No skill")
#plt.legend()
#plt.show()
pr_cl=[lmap(pred, 0.5) for pred in preds[0]]

cm = confusion_matrix(preds[1],pr_cl)
print(cm)
ConfusionMatrixDisplay.from_predictions(preds[1],pr_cl, colorbar=False,
                                        cmap="Blues", normalize="true", values_format=".4g")
plt.show()"""


"""preds = ["Не токсин"]*123575 + ["Токсин"]*56 + ["Не токсин"]*723 + ["Токсин"]*1313
reals = ["Не токсин"]*123631 + ["Токсин"]*2036

ConfusionMatrixDisplay.from_predictions(reals,preds, colorbar=False,
                                        cmap="Blues", normalize="true", values_format=".4f")
plt.title("Розмірність 100, голосування")
plt.xlabel("Передбачений клас")
plt.ylabel("Реальний клас")
plt.show()

preds = ["Не токсин"]*123437 + ["Токсин"]*194 + ["Не токсин"]*342 + ["Токсин"]*1694
ConfusionMatrixDisplay.from_predictions(reals,preds, colorbar=False,
                                        cmap="Blues", normalize="true", values_format=".4f")

plt.title("Розмірність 100, найподібніший")
plt.xlabel("Передбачений клас")
plt.ylabel("Реальний клас")
plt.show()

preds = ["Не токсин"]*123504 + ["Токсин"]*127 + ["Не токсин"]*513 + ["Токсин"]*1523
ConfusionMatrixDisplay.from_predictions(reals,preds, colorbar=False,
                                        cmap="Blues", normalize="true", values_format=".4f")
plt.title("Розмірність 200, голосування")
plt.xlabel("Передбачений клас")
plt.ylabel("Реальний клас")
plt.show()


preds = ["Не токсин"]*123186 + ["Токсин"]*445 + ["Не токсин"]*215 + ["Токсин"]*1821
ConfusionMatrixDisplay.from_predictions(reals,preds, colorbar=False,
                                        cmap="Blues", normalize="true", values_format=".4f")
plt.title("Розмірність 200, найподібніший")
plt.xlabel("Передбачений клас")
plt.ylabel("Реальний клас")
plt.show()

"""
preds = ["Не токсин"]*122940 + ["Токсин"]*691 + ["Не токсин"]*230 + ["Токсин"]*1806
reals = ["Не токсин"]*123631 + ["Токсин"]*2036
ConfusionMatrixDisplay.from_predictions(reals,preds, colorbar=False,
                                        cmap="Blues", normalize="true", values_format=".4f")
plt.title("Передбачення на основі BLASTp")
plt.xlabel("Передбачений клас")
plt.ylabel("Реальний клас")
plt.show()

"""preds = ["Не токсин"]*123718 + ["Токсин"]*146 + ["Не токсин"]*426 + ["Токсин"]*1377
reals = ["Не токсин"]*123864 + ["Токсин"]*1803

ConfusionMatrixDisplay.from_predictions(reals,preds, colorbar=False,
                                        cmap="Blues", normalize="true", values_format=".4f")
plt.title("Порогове значення 0.5")
plt.xlabel("Передбачений клас")
plt.ylabel("Реальний клас")
plt.show()

preds = ["Не токсин"]*121255 + ["Токсин"]*2609 + ["Не токсин"]*120 + ["Токсин"]*1683
ConfusionMatrixDisplay.from_predictions(reals,preds, colorbar=False,
                                        cmap="Blues", normalize="true", values_format=".4f")
plt.title("Порогове значення 0.00001")
plt.xlabel("Передбачений клас")
plt.ylabel("Реальний клас")
plt.show()"""

"""preds = ["Не токсин"]*5607 + ["Токсин"]*61 + ["Не токсин"]*1998 + ["Токсин"]*4743
reals = ["Не токсин"]*5668 + ["Токсин"]*6741

ConfusionMatrixDisplay.from_predictions(reals,preds, colorbar=False,
                                        cmap="Blues", normalize="true", values_format=".4f")
plt.title("ToxinPred3.0")
plt.xlabel("Передбачений клас")
plt.ylabel("Реальний клас")
plt.show()"""

"""preds = ["Не токсин"]*718 + ["Токсин"]*386 + ["Не токсин"]*139 + ["Токсин"]*965
reals = ["Не токсин"]*1104 + ["Токсин"]*1104

ConfusionMatrixDisplay.from_predictions(reals,preds, colorbar=False,
                                        cmap="Blues", normalize="true", values_format=".4f")
plt.title("Модель, порогове значення 0.00001")
plt.xlabel("Передбачений клас")
plt.ylabel("Реальний клас")
plt.show()"""
