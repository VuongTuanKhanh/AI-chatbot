import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import plot_roc_curve
from sklearn.metrics import classification_report

class Model():
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def extract_training_info(self, file_name, probs):
        """
        A helper function helps generating training information files
        
        :param file_name: file name
        :param probs: list of probabilities for test samples
        
        :return: None
        """
        data = pd.DataFrame()
        data['TARGET'] = probs
        data.to_csv(file_name, index = False)
        print('Done writting data file: ', file_name)

    def plot_bar_chart(self, title, x_labels, y, y_label):
        """
        A helper function helps plotting bar graphs

        :param title: title of the graph
        :param x_labels: x labels
        :param y: data for each x label
        :param y_labels: label for y-axis

        :return: None
        """
        index = np.arange(len(x_labels))
        plt.bar(index, y)
        plt.ylabel(y_label, fontsize=15)
        plt.xticks(index, x_labels, fontsize=15, rotation=90)
        plt.title(title)
        plt.show()

    def print_confusion(self, y_test, y_test_pred):
        """ This function prints out the confusion matrix in pretty format
            for a given pair of true labels and predicting labels.
            
        :param y_test: A list of true labels.
        :param y_test_pred: A list of prediction labels.
        
        :return: None
        """
        
        unique_label = np.unique(y_test)
        confusion = confusion_matrix(y_test, y_test_pred)
        print(pd.DataFrame(confusion,
                        index=['true:{:}'.format(x) for x in unique_label], 
                        columns=['pred:{:}'.format(x) for x in unique_label]))

    def train_classifiers(self, classifier_names, classifiers):
        """
        Train an array of classifers

        :param classifier_names: A list of classifer names
        :param classifiers: A list of scikit-learn estimators (classifers)

        :return: None
        """

        train_accuracies = []
        train_roc_auc_scores = []
        train_times = []
        test_prob_predictions = []

        for idx, clf in enumerate(classifiers):
            classifier_name = classifier_names[idx]
            print('Training for classifier: ', classifier_name)
            start_time = time.time()
            classifiers[idx].fit(self.X_train, self.y_train)
            train_time = time.time() - start_time
            print(f"Training time for {classifier_name}: {train_time} secs")

            # Predict on the train, test dataset after training the classifier
            y_train_pred = clf.predict(self.X_train)
            y_test_pred = clf.predict(self.X_test)       

            print(f"***** START Report for classifer {classifier_name} *****")
            from sklearn.metrics import roc_auc_score
            train_accuracy = classifiers[idx].score(self.X_train, self.y_train, zero_division=0)
            test_accuracy = classifiers[idx].score(self.X_test, self.y_test, zero_division=0)
            # train_probs = classifiers[idx].predict_proba(self.X_train)[:, 1]
            # train_roc_auc = roc_auc_score(self.y_train, train_probs)
            # test_probs = classifiers[idx].predict_proba(self.X_test)[:, 1]
            print(classification_report(self.y_test, y_test_pred))

            # print(f'Train time: {train_time} seconds')
            # print(f'Train accuracy: {train_accuracy}')
            # # print(f'Train roc auc score: {train_roc_auc}')
            # print('*****************************')

            # train_accuracies.append(train_accuracy)
            # # train_roc_auc_scores.append(train_roc_auc)
            # train_times.append(train_time)
            # # test_prob_predictions.append(test_probs)

            # # for idx, classifer_name in enumerate(classifier_names):
            # #     self.extract_training_info(
            # #         '{}/{}_{}.csv'.format(self.prefix_url, classifer_name, self.suffix_url),
            # #         test_prob_predictions[idx]
            # #     )

            # self.plot_bar_chart('Train accuracies', classifier_names, train_accuracies, 'Train accuracy')
            # self.plot_bar_chart('Train roc_auc scores', classifier_names, train_roc_auc_scores, 'Train roc_auc')
            # self.plot_bar_chart('Train times', classifier_names, train_times, 'Train time')

            # print("- Train confusion matrix")
            # self.print_confusion(self.y_train, y_train_pred)
            # print("- Test confusion matrix")                                                 
            # self.print_confusion(self.y_test, y_test_pred)

            # # Use the precision_recall_fscore_support the get train_results, test_results.
            # train_results = precision_recall_fscore_support(self.y_train, y_train_pred)
            # test_results = precision_recall_fscore_support(self.y_test, y_test_pred)

            # # Get train/test precision/recall from train/test results above
            # train_precision = train_results[0]
            # train_recall = train_results[1]
            # test_precision = test_results[0]
            # test_recall = test_results[1]             
            # print("- Train precision: ", train_precision)
            # print("- Train recall: ", train_recall)
            # print("- Test precision: ", test_precision)
            # print("- Test recall: ", test_recall)

            # # # Calculate train/test f1 using the f1_score function
            # # # 2 * (precision * recall) / (precision + recall)
            # # train_f1 = f1_score(y_train, y_train_pred)
            # # test_f1 = f1_score(y_test, y_test_pred)
            # # print("- Train f1: ", train_f1)
            # # print("- Test f1: ", test_f1)

            # # # Calculate train/test roc_auc using the roc_auc_score function
            # # train_probs = clf.predict_proba(self.X_train)[:, 1]      
            # # train_roc_auc = roc_auc_score(self.y_train, train_probs)
            # # test_probs = clf.predict_proba(self.X_test)[:, 1]
            # # test_roc_auc = roc_auc_score(self.y_test, test_probs)
            # # print("- Train auc: ", train_roc_auc)
            # # print("- Test auc: ", test_roc_auc)
            
            # # Plot the train/test roc curve using the plot_roc_curve from scikitplot lib
            # # print("- Plot train roc")
            # # plot_roc_curve(clf, self.X_train, self.y_train)
            # # plt.show()
            # # print("- Plot test roc")
            # # plot_roc_curve(clf, self.X_test, self.y_test)
            # # plt.show()
            
            # print (f"***** END Report for classifer {classifier_name} *****")

    @property
    def prefix_url(self):
        import os
        return os.getcwd()
    
    @property
    def suffix_url(self):
        return 'models'