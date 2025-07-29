import torch
import torch.nn as nn
import torch.optim as optim
import copy



class CNN_LSTM_Model(nn.Module):
    def __init__(self):
        super(CNN_LSTM_Model, self).__init__()


        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d((2, 2)),

            nn.Conv2d(32, 32, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(),
            # nn.BatchNorm2d(32),   Taking out the batch norm and the Dropout to try to force overfitting
            nn.MaxPool2d((2, 2)),

            nn.Conv2d(32, 32, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(),
            # nn.BatchNorm2d(32),
            nn.MaxPool2d((2, 2)),

            nn.Flatten()
        )


        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, 43, 180)
            cnn_output = self.cnn(dummy_input)
            self.lstm_input_size = cnn_output.shape[1]


        self.lstm = nn.LSTM(self.lstm_input_size, 128, num_layers=2, batch_first=True, bidirectional=True)


        self.fc = nn.Sequential(
            nn.Linear(128 * 2, 128),
            nn.ReLU(),
            # nn.Dropout(0.5),
            nn.Linear(128, 4)
        )

    def forward(self, x):
        batch_size, timesteps, C, H, W = x.size()  # x shape: [batch_size, timesteps, C, H, W]

        c_in = x.view(batch_size * timesteps, C, H, W)


        c_out = self.cnn(c_in)

        r_in = c_out.view(batch_size, timesteps, -1)

        r_out, _ = self.lstm(r_in)

        out = self.fc(r_out[:, -1, :])
        return out


def train_model(train_loader, val_loader, device, model_architecture):
    model = model_architecture().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, verbose=True)

    # Training the model for longer from 100 to 200 and increasing patience from 10 to 20
    num_epochs = 200
    best_val_accuracy = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    patience = 20
    counter = 0

    print(f"Starting training with {len(train_loader)} training batches and {len(val_loader)} validation batches")

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        avg_train_loss = running_loss / len(train_loader)
        train_accuracy = 100 * train_correct / train_total

        model.eval()
        val_correct = 0
        val_total = 0
        val_running_loss = 0.0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_accuracy = 100 * val_correct / val_total
        avg_val_loss = val_running_loss / len(val_loader)

        print(f'Epoch [{epoch + 1}], Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, '
              f'Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%')

        scheduler.step(avg_val_loss)

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model_wts = copy.deepcopy(model.state_dict())
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f'Early stopping at epoch {epoch + 1}')
                print(f'Best validation accuracy: {best_val_accuracy:.2f}%')
                break

    model.load_state_dict(best_model_wts)
    return model