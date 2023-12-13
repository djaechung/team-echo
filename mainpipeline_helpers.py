import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import os
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

class r3dmodel(nn.Module):
  def __init__(self, model1, regression = False):
    super(r3dmodel, self).__init__()
    self.regression = regression
    self.preloaded_model = model1
    self.new_layer1 = nn.Linear(400,1)
    if self.regression == False:
        self.new_layer2 = nn.Sigmoid()
    
  def forward(self, x):
    x = self.preloaded_model(x)
    x = self.new_layer1(x)
    if self.regression == False:
        x = self.new_layer2(x)
    return x

class swin3dmodel(nn.Module):
  def __init__(self, model2, regression = False):
    super(swin3dmodel, self).__init__()
    self.regression = regression
    self.preloaded_model = model2
    self.new_layer1 = nn.Linear(400,1)
    if self.regression == False:
        self.new_layer2 = nn.Sigmoid()

  def forward(self, x):
    x = self.preloaded_model(x)
    x = self.new_layer1(x)
    if self.regression == False:
        x = self.new_layer2(x)
    return x

class CustomBinaryCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(CustomBinaryCrossEntropyLoss, self).__init__()

    def forward(self, logits, targets):
        # Convert logits to probabilities using sigmoid function
        probabilities = logits
        
        # Calculate  binary cross-entropy loss
        loss = -(targets * torch.log(probabilities) + (1 - targets) * torch.log(1 - probabilities))
        # print("loss is: ", loss)
        return loss

class CustomMSELoss(nn.Module):
    def __init__(self):
        super(CustomMSELoss, self).__init__()
    
    def forward(self, predicted, target):
        difference = predicted - target

        mse_loss = difference ** 2
        return mse_loss

def calculate_metrics(outputs, labels):
  ### Update correct scores
  running_corrects = 0
  running_total = 0
  TP, TN, FP, FN = 0, 0, 0, 0

  for i, prob in enumerate(outputs):
    ### Determining Prediction
    if prob > 0.5:
      pred = 1
    else:
      pred = 0
    
    ### Label
    label = labels[i]
    if pred == label:
      running_corrects += 1
    running_total += 1

    ### TP, TN, FP, FN calculation
    if pred == 1 and label == 1:
      TP += 1
    elif pred == 0 and label == 0:
      TN += 1
    elif pred == 1 and label == 0:
      FP += 1
    elif pred == 0 and label == 1:
      FN += 1
  
  return running_corrects, running_total, TP, TN, FP, FN

def epoch_evaluation(model, data_dir, loss_fn, optimizer, batch_size_fake, batch_size_effective, device, training=True):
    running_loss = 0
    running_corrects = 0
    running_total = 0
    running_probabilities = []
    running_labels = []
    running_video_ids = []
    running_efs = []

    loss = None

    running_losses = []
    num_additions = 0

    for j, file in enumerate(os.listdir(data_dir)):
    #   print("File #: ", j)
      dataset = torch.load(f'{data_dir}/{file}')
      dataloader = DataLoader(dataset, batch_size=batch_size_fake, shuffle=training)
      optimizer.zero_grad()

      for i, (inputs, labels, video_id, ef_actual) in enumerate(dataloader):
        # print("i is: ", i)
        # print("inputs.shape is: ", inputs.shape)
        # print("labels is: ", labels)
        # print("video_id is: ", video_id)
        inputs = inputs.permute(0, 2, 1, 3, 4)
        # print("inputs.shape is: ", inputs.shape)

        inputs = inputs.to(device)
        labels = labels.to(device)

        if training == True:
            with torch.set_grad_enabled(True):
              if (i+1) % batch_size_effective == 0:
                # print("Updating parameters")
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                loss = None

        if training == False:
          with torch.no_grad():
            outputs = model(inputs)
            if outputs.item() == np.nan:
              print("issue happening")
              print("data_dir is: ", data_dir)
              print("file is: ", file)
              print("i is: ", i)
            # print("outputs.shape is: ", outputs.shape)
            outputs = outputs.squeeze(0)

            if loss == None:
              loss = loss_fn(outputs, labels)
              running_losses.append(loss.item())
              num_additions += 1
              running_loss += loss.item()
            else:
              loss_temp = loss_fn(outputs, labels)
              running_losses.append(loss_temp.item())
              num_additions += 1
              running_loss += loss_temp.item()
              loss += loss_temp
            
        elif training == True:
          outputs = model(inputs)
          # print("outputs.shape is: ", outputs.shape)          
          outputs = outputs.squeeze(0)

          if loss == None:
            loss = loss_fn(outputs, labels)
            running_losses.append(loss.item())
            num_additions += 1
            running_loss += loss.item()
          else:
            loss_temp = loss_fn(outputs, labels)
            running_losses.append(loss_temp.item())
            num_additions += 1
            running_loss += loss_temp.item()
            loss += loss_temp
          
        running_labels.append(labels.item())
        running_probabilities.append(outputs.item())
        running_video_ids.append(video_id[0])
        running_efs.append(ef_actual)
    # print("running_probabilities is: ", running_probabilities)
    # print("running_labels is: ", running_labels)
    running_corrects, running_total, TP, TN, FP, FN = calculate_metrics(running_probabilities, running_labels)
    # calculated_loss = sum([-1*(running_labels[i]*math.log(running_probabilities[i]) + (1-running_labels[i])*math.log(1-running_probabilities[i])) for i in range(len(running_labels))])/len(running_labels)
    return running_loss / len(running_probabilities), running_corrects, running_total, running_probabilities, running_labels, running_video_ids, running_efs, TP, TN, FP, FN

def optimize_threshold(probs, labels):
    best_threshold = 0
    best_accuracy = 0

    thresholds = []
    accuracies = []

    for threshold in np.arange(0, 1.001, 0.001):
        predictions = (probs > threshold).astype(int)
        accuracy = accuracy_score(labels, predictions)
        thresholds.append(threshold)
        accuracies.append(accuracy)
        if accuracy > best_accuracy:
            best_threshold = threshold
            best_accuracy = accuracy
    
    return best_threshold, thresholds, accuracies

def accuracy_score_threshold(threshold, probs, labels):
    threshold = np.float64(threshold)
    predictions = (probs > threshold).astype(int)
    accuracy = accuracy_score(labels, predictions)
    return accuracy


def metrics(probabilities, labels, accuracy_v_threshold_plot, roc_auc_plot, phase, threshold_to_use=None):
    print(phase)

    best_threshold, thresholds, accuracies = optimize_threshold(probabilities, labels)      
    roc_auc = roc_auc_score(labels, probabilities)

    if threshold_to_use == None:
        accuracy = accuracy_score_threshold(best_threshold, probabilities, labels)
        print(f"best_threshold (used) is {best_threshold}")
    else:
        print(f"best_threshold (not used) is {best_threshold}")
        print(f"theoretical accuracy (with best_threshold) is: {accuracy_score_threshold(best_threshold, probabilities, labels)}")
        print(f"threshold used is {threshold_to_use}")
        accuracy = accuracy_score_threshold(threshold_to_use, probabilities, labels)
    
    ax1 = accuracy_v_threshold_plot
    ax2 = roc_auc_plot

    ax1.plot(thresholds, accuracies, label=f'{phase} Curve')
    ax1.set_title("Accuracy vs Threshold Plot")
    ax1.set_xlabel("Threshold")
    ax1.set_ylabel("Accuracy")
    ax1.axhline(y=0.875, color='red', linestyle='--')
    ax1.legend(loc="lower right")

    fpr, tpr, _ = roc_curve(labels, probabilities)
    ax2.plot(fpr, tpr, label=f'{phase} Curve (Area=%0.2f)' % roc_auc)
    ax2.plot([0, 1], [0, 1], 'k--')  # dashed diagonal line
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate')
    ax2.set_title('ROC Curve')
    ax2.legend(loc="lower right")

    print("minority class prevalence is: ", sum([1 for elem in labels if elem == 1])/len(labels))
    print("majority class prevalence is: ", sum([1 for elem in labels if elem == 0])/len(labels))
    print("AUC is: ", roc_auc)
    print("Accuracy is: ", accuracy)

    return best_threshold


def epoch_evaluation_regression(model, data_dir, loss_fn, optimizer, batch_size_fake, batch_size_effective, device, training=True):
    running_preds = []
    running_efs = []
    running_video_ids = []
    running_losses = []

    running_loss = 0
    loss = None

    for j, file in enumerate(os.listdir(data_dir)):
    #   print("File #: ", j)
      dataset = torch.load(f'{data_dir}/{file}')
      dataloader = DataLoader(dataset, batch_size=batch_size_fake, shuffle=training)
      optimizer.zero_grad()

      for i, (inputs, labels, video_id, ef_actual) in enumerate(dataloader):
        # print("i is: ", i)
        # print("inputs.shape is: ", inputs.shape)
        # print("ef_actual is: ", ef_actual)
        # print("video_id is: ", video_id)
        inputs = inputs.permute(0, 2, 1, 3, 4)
        # print("inputs.shape is: ", inputs.shape)

        inputs = inputs.to(device)
        ef_actual = ef_actual.to(device)

        if training == True:
            with torch.set_grad_enabled(True):
              if (i+1) % batch_size_effective == 0:
                # print("Updating parameters")
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                loss = None

        if training == False:
          with torch.no_grad():
            outputs = model(inputs)
            if outputs.item() == np.nan:
              print("issue happening")
              print("data_dir is: ", data_dir)
              print("file is: ", file)
              print("i is: ", i)
            # print("outputs.shape is: ", outputs.shape)
            outputs = outputs.squeeze(0)

            if loss == None:
              loss = loss_fn(outputs, ef_actual)
              running_losses.append(loss.item())
              running_loss += loss.item()
            else:
              loss_temp = loss_fn(outputs, ef_actual)
              running_losses.append(loss_temp.item())
              running_loss += loss_temp.item()
              loss += loss_temp
            
        elif training == True:
          outputs = model(inputs)
          # print("outputs.shape is: ", outputs.shape)          
          outputs = outputs.squeeze(0)

          if loss == None:
            loss = loss_fn(outputs, ef_actual)
            running_losses.append(loss.item())
            running_loss += loss.item()
          else:
            loss_temp = loss_fn(outputs, ef_actual)
            running_losses.append(loss_temp.item())
            running_loss += loss_temp.item()
            loss += loss_temp
        
        running_video_ids.append(video_id[0])
        running_efs.append(ef_actual.item())
        running_preds.append(outputs.item())
        # print("outputs.item() is: ", outputs.item())

    assert abs(sum(running_losses) / len(running_efs) - running_loss / len(running_efs)) < 1e-2
    return running_loss / len(running_efs), running_preds, running_efs, running_video_ids

def metrics_regression(predicted_ef, actual_ef):
    # print("predicted_ef is: ", predicted_ef)
    # print("actual_ef is: ", actual_ef)
    R_2_score = r2_score(actual_ef, predicted_ef)
    MAE_score = mean_absolute_error(actual_ef, predicted_ef)
    RMSE_score = mean_squared_error(actual_ef, predicted_ef)**0.5

    return R_2_score, MAE_score, RMSE_score
