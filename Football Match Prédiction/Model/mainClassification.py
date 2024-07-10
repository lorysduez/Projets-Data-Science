from Classification import (load_data, shape_matrice, preprocess_data,
                            model_test, train_model, plot_confusion_matrix,
                            load_data_Forme, preprocess_data_Forme, plot_training_history,
                            model_conv2dTranspose)
from sklearn.model_selection import train_test_split
import tensorflow as tf

"Test SANS inclure les formes des équipes"
data = load_data()
# Verif = shape_matrice()
X_combined, Y = preprocess_data()
X_train, X_test, Y_train, Y_test = train_test_split(X_combined, Y, test_size=0.2, random_state=42)
# trained_model, trained_history = train_model(X_combined, Y, model_test())
# trained_model.save("C:/Users/LD/Desktop/Projet P5/ModelClassification")
loaded_model = tf.keras.models.load_model("C:/Users/LD/Desktop/Projet P5/ModelClassification")
plot_confusion_matrix(loaded_model, X_test, Y_test)
# plot_training_history(loaded_model)

"Test Avec inclure les formes des équipes"
# data = load_data_Forme()
# X_combined, Y = preprocess_data_Forme()
# X_train, X_test, Y_train, Y_test = train_test_split(X_combined, Y, test_size=0.2, random_state=42)
# trained_model, trained_history = train_model(X_combined, Y, model_test())
# plot_confusion_matrix(trained_model, X_test, Y_test)
# plot_training_history(trained_history)

