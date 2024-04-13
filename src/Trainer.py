import torch

from util.evaluations import mean_columnwise_log_loss_torch


class Trainer:
    def __init__(self, model, device, optimizer):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optimizer

    def train_epoch(self, trainloader):
        self.model.train()
        total_loss = 0
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = mean_columnwise_log_loss_torch(labels.float(), outputs)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(trainloader)

    def validate_epoch(self, validloader):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for inputs, labels in validloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = mean_columnwise_log_loss_torch(labels.float(), outputs)
                total_loss += loss.item()
        return total_loss / len(validloader)

    def make_predictions(self, testloader):
        self.model.eval()
        all_predictions = []
        with torch.no_grad():
            for inputs in testloader:
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                all_predictions.extend(outputs.cpu().numpy())
        return all_predictions
