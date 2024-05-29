import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from svm_visualization import draw_boundary
from players import aaron_judge, jose_altuve, david_ortiz

fig, ax = plt.subplots()

# print(aaron_judge.columns)
aaron_judge["type"] = aaron_judge["type"].map({"S": 1, "B": 0})
print(aaron_judge.type.unique())

aaron_judge = aaron_judge.dropna(subset = ["type", "plate_x", "plate_z"])

plt.scatter(aaron_judge["plate_x"], aaron_judge["plate_z"], c = aaron_judge["type"], cmap = plt.cm.coolwarm, alpha = 0.25)

training_set, validation_set = train_test_split(aaron_judge, random_state = 1)
classifier = SVC(kernel = "rbf", gamma = 1, C = 15)
classifier.fit(training_set[["plate_x", "plate_z"]], training_set["type"])
draw_boundary(ax, classifier)

plt.show()
plt.clf()

print(classifier.score(validation_set[["plate_x", "plate_z"]], validation_set["type"]))
