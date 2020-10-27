import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import os
import logging
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
 
# nn
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
 
# predefine
from tensorflow.keras.applications import ResNet50V2
 
# callbacks
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, Callback, ReduceLROnPlateau, TensorBoard

# Prepare the training dataset.
batch_size = 64
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
x_train = np.reshape(x_train, (-1, 32, 32, 3))
x_test = np.reshape(x_test, (-1, 32, 32, 3))

# define architecture
baseModel = ResNet50V2(weights = "imagenet", include_top = False, input_shape = (32, 32, 3))
headModel = baseModel.output
headModel = GlobalAveragePooling2D()(headModel)
headModel = Dropout(0.50)(headModel)
headModel = Dense(10, activation = 'softmax', name = "resnet50v2_dense")(headModel)

model = Model(inputs = baseModel.input, outputs = headModel, name = "ResNet50V2")
model.trainable = True

def get_batch(batch_size, x, y):
  train_dataset = tf.data.Dataset.from_tensor_slices((x, y))
  train_dataset = train_dataset.shuffle(buffer_size = 1024).batch(batch_size)
  return train_dataset

def get_logs(logs, epoch, model, x, y):
  y_pred = model.predict(x).argmax(axis = 1)
  conf_mat = confusion_matrix(y, y_pred)
  class_label = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
  df = pd.DataFrame(conf_mat, index = class_label, columns = class_label)
  sns.heatmap(df, annot = True, cmap = "YlGnBu", fmt = "d")
  plt.title("Confusion Matrix")
  plt.xlabel("Predicted Label")
  plt.ylabel("True Label")
  plt.show()

  temp = classification_report(y, y_pred, target_names = class_label, output_dict = True)

  epoch_dict, precision_dict, recall_dict, f1score_dict = {}, {}, {}, {}
  
  for label in class_label:
    precision_dict[label] = np.round(temp[label]["precision"], 2)
    recall_dict[label] = np.round(temp[label]["recall"], 2)
    f1score_dict[label] = np.round(temp[label]["f1-score"], 2)

  epoch_dict["precision"] = precision_dict
  epoch_dict["recall"] = recall_dict
  epoch_dict["f1score"] = f1score_dict
  epoch_dict["confusion_matrix"] = conf_mat
  logs["epoch_{}".format(epoch)] = epoch_dict

  return logs

model.stop_training = False
model.compile(loss = "sparse_categorical_crossentropy", optimizer = tf.keras.optimizers.Adam(), metrics = ["accuracy"])

earlystop = EarlyStopping(monitor = "val_loss", patience = 20, verbose = 1)
earlystop.set_model(model)
earlystop.on_train_begin()

modelcheckpoint = ModelCheckpoint(filepath="weights/", monitor="val_loss", verbose = 1, save_best_only=True)
modelcheckpoint.set_model(model)
modelcheckpoint.on_train_begin()

reduce_lr = ReduceLROnPlateau(monitor = "val_loss", patience = 10, verbose = 1)
reduce_lr.set_model(model)
reduce_lr.on_train_begin()

tensorboard = TensorBoard(log_dir = "logs/")
tensorboard.set_model(model)
tensorboard.on_train_begin()

epochs = 3
train_logs_dict = {}
test_logs_dict = {}
for epoch in range(epochs):
    training_acc, testing_acc, training_loss, testing_loss = [], [], [], []
    print("\nStart of epoch %d" % (epoch+1,)) 
    # Iterate over the batches of the dataset.
    modelcheckpoint.on_epoch_begin(epoch)
    earlystop.on_epoch_begin(epoch)
    reduce_lr.on_epoch_begin(epoch)
    tensorboard.on_epoch_begin(epoch)
    for x_batch_train, y_batch_train in get_batch(batch_size, x_train, y_train):

        train_loss, train_accuracy = model.train_on_batch(x_batch_train, y_batch_train)
        training_acc.append(train_accuracy)
        training_loss.append(train_loss)
    
    for x_batch_test, y_batch_test in  get_batch(batch_size, x_test, y_test):

        test_loss, test_accuracy = model.test_on_batch(x_batch_test, y_batch_test)
        testing_acc.append(test_accuracy)
        testing_loss.append(test_loss)
    train_logs_dict = get_logs(train_logs_dict, epoch, model, x_train, y_train)
    test_logs_dict = get_logs(test_logs_dict, epoch, model, x_test, y_test)
    logs = {'acc': np.mean(training_acc), 'loss': np.mean(training_loss), 'val_loss': np.mean(testing_loss), 'val_acc': np.mean(testing_acc)}
    modelcheckpoint.on_epoch_end(epoch, logs)
    earlystop.on_epoch_end(epoch, logs)
    reduce_lr.on_epoch_end(epoch, logs)
    tensorboard.on_epoch_end(epoch, logs)
    print("accuracy: {}, loss: {}, validation accuracy: {}, validation loss: {}".format(np.mean(training_acc), np.mean(training_loss), np.mean(testing_acc), np.mean(testing_loss)))
    if model.stop_training:
      break
earlystop.on_train_end()
modelcheckpoint.on_train_end()
reduce_lr.on_train_end()
tensorboard.on_train_end()

# confusion metric for training
y_train_pred = model.predict(x_train).argmax(axis = 1)

conf_mat = confusion_matrix(y_train, y_train_pred)
class_label = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
df = pd.DataFrame(conf_mat, index = class_label, columns = class_label)
sns.heatmap(df, annot = True, cmap = "YlGnBu", fmt = "d")
plt.title("Confusion Matrix for Training data")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

# confusion metric for testing
y_test_pred = model.predict(x_test).argmax(axis = 1)

conf_mat = confusion_matrix(y_test, y_test_pred)
class_label = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
df = pd.DataFrame(conf_mat, index = class_label, columns = class_label)
sns.heatmap(df, annot = True, cmap = "YlGnBu", fmt = "d")
plt.title("Confusion Matrix for Testing data")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

# classification report for training
print(classification_report(y_train, y_train_pred, target_names = class_label))

# classification report for testing
print(classification_report(y_test, y_test_pred, target_names = class_label))

# training dict
print(train_logs_dict)

# testing dict
print(test_logs_dict)

json_dict = {}
json_dict["training"] = train_logs_dict
json_dict["validation"] = test_logs_dict

import json

class NumpyEncoder(json.JSONEncoder):
  def default(self, obj):
    if isinstance(obj, np.ndarray):
      return obj.tolist()
    return json.JSONEncoder.default(self, obj)

with open('training_logs.json', 'w') as tl:
  json.dump(json_dict, tl, cls=NumpyEncoder)