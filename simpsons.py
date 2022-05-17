# Import what we need.
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import pandas as pd
import matplotlib.pyplot as plt
from skimage import io
import os
from PIL import Image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Custom class to load our dataset. We give it the created text file (it counts as a text file).
class SimpsonsDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    # Gets the image and label from the .txt file.
    def __getitem__(self, index):
        img_path = self.annotations.iloc[index, 0]
        image = io.imread(img_path)
        # Convert the label to a tensor.
        y_label = torch.tensor(int(self.annotations.iloc[index, 1]))

        if self.transform:
            # Transform the image into a tensor.
            image = self.transform(image)
        return (image, y_label)


# Custom Neural Net class (MLP).
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        return out


# MLP with 2 hidden layers.
class NeuralNet2(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet2, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l2(out)
        return out


# Custom CNN class. (w-f + 2p) / s + 1
# 2 CONV layers, filter size 3.
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 12, 3)
        self.conv3 = nn.Conv2d(12, 12, 3)
        self.fc1 = nn.Linear(12*26*26, 300)
        self.fc2 = nn.Linear(300, 100)
        self.fc3 = nn.Linear(100, 4)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 12*26*26)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# CNN with 3 CONV layers of filter size 3.
class ConvNet2(nn.Module):
    def __init__(self):
        super(ConvNet2, self).__init__()

        self.conv1 = nn.Conv2d(3, 6, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 12, 3)
        self.fc1 = nn.Linear(12*54*54, 300)
        self.fc2 = nn.Linear(300, 100)
        self.fc3 = nn.Linear(100, 4)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 12*54*54)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Resizes all images in the dataset.
def resize():
    dataset_dir = 'dataset/train'
    for folder in os.listdir(dataset_dir):
        if folder == 'homer_simpson' or folder == 'bart_simpson' or folder == 'lisa_simpson' or folder == 'marge_simpson':
            for img in os.listdir(dataset_dir + '/' + folder):
                im = Image.open(dataset_dir + '/' + folder + '/' + img)
                new_im = im.resize((224, 224))
                new_im.save(dataset_dir + '/' + folder + '/' + img)


# Function to save our model.
def save_checkpoint(state, filename):
    torch.save(state, filename)


# Function to load our model.
def load_checkpoint(checkpoint, mod, opt):
    mod.load_state_dict(checkpoint['state_dict'])
    opt.load_state_dict(checkpoint['optimizer'])


# Prints more data about the MLP classifications/misclassifications.
def MLP_data():
    model = NeuralNet2(image_size * image_size * 3, hidden_neurons, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    load_checkpoint(torch.load("MLP-25-50-300-0.0001.pth.tar"), model, optimizer)
    model.eval()

    bart_total = 0
    bart_correct = 0
    bart_incorrect = 0
    bart_homer = 0
    bart_lisa = 0
    bart_marge = 0
    homer_total = 0
    homer_correct = 0
    homer_incorrect = 0
    homer_bart = 0
    homer_lisa = 0
    homer_marge = 0
    lisa_total = 0
    lisa_correct = 0
    lisa_incorrect = 0
    lisa_bart = 0
    lisa_homer = 0
    lisa_marge = 0
    marge_total = 0
    marge_correct = 0
    marge_incorrect = 0
    marge_bart = 0
    marge_homer = 0
    marge_lisa = 0

    for images, labels in test_loader:
        for i, img in enumerate(images):
            img_re = img.reshape(1, 224*224*3)
            output = torch.argmax(model(img_re))
            if output.data == 0:
                bart_total += 1
                if output == labels[i]:
                    bart_correct += 1
                else:
                    bart_incorrect += 1
                    if labels[i].data == 1:
                        bart_homer += 1
                    elif labels[i].data == 2:
                        bart_lisa += 1
                    elif labels[i].data == 3:
                        bart_marge += 1
            elif output.data == 1:
                homer_total += 1
                if output == labels[i]:
                    homer_correct += 1
                else:
                    homer_incorrect += 1
                    if labels[i].data == 0:
                        homer_bart += 1
                    elif labels[i].data == 2:
                        homer_lisa += 1
                    elif labels[i].data == 3:
                        homer_marge += 1
            elif output.data == 2:
                lisa_total += 1
                if output == labels[i]:
                    lisa_correct += 1
                else:
                    lisa_incorrect += 1
                    if labels[i].data == 0:
                        lisa_bart += 1
                    elif labels[i].data == 1:
                        lisa_homer += 1
                    elif labels[i].data == 3:
                        lisa_marge += 1
            elif output.data == 3:
                marge_total += 1
                if output == labels[i]:
                    marge_correct += 1
                else:
                    marge_incorrect += 1
                    if labels[i].data == 0:
                        marge_bart += 1
                    elif labels[i].data == 1:
                        marge_homer += 1
                    elif labels[i].data == 2:
                        marge_lisa += 1

            if output != labels[i]:
                img = img.swapaxes(0, 1)
                img = img.swapaxes(1, 2)
                fig = plt.gcf()
                fig.canvas.set_window_title(str(i))
                plt.imshow(img)
                plt.show()

    print("Bart Accuracy:", 100.0 * (bart_correct / bart_total))
    print("Bart Inaccuracy:", 100.0 * (bart_incorrect / bart_total))
    print("Bart-Homer:", bart_homer)
    print("Bart-Lisa:", bart_lisa)
    print("Bart-Marge:", bart_marge)
    print("Homer Accuracy:", 100.0 * (homer_correct / homer_total))
    print("Homer Inaccuracy:", 100.0 * (homer_incorrect / homer_total))
    print("Homer-Bart:", homer_bart)
    print("Homer-Lisa:", homer_lisa)
    print("Homer-Marge:", homer_marge)
    print("Lisa Accuracy:", 100.0 * (lisa_correct / lisa_total))
    print("Lisa Inaccuracy:", 100.0 * (lisa_incorrect / lisa_total))
    print("Lisa-Bart:", lisa_bart)
    print("Lisa-Homer:", lisa_homer)
    print("Lisa-Marge:", lisa_marge)
    print("Marge Accuracy:", 100.0 * (marge_correct / marge_total))
    print("Marge Inaccuracy:", 100.0 * (marge_incorrect / marge_total))
    print("Marge-Bart:", marge_bart)
    print("Marge-Homer:", marge_homer)
    print("Marge-Lisa:", marge_lisa)
    print("Total Accuracy:", 100.0 * ((bart_correct + homer_correct + lisa_correct + marge_correct) / (
                bart_total + homer_total + lisa_total + marge_total)))

# Prints more data about the CNN classifications/misclassifications.
def CNN_data():

    model = ConvNet2().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    load_checkpoint(torch.load("CNN13conv-25-50-0.001.pth.tar"), model, optimizer)
    model.eval()

    bart_total = 0
    bart_correct = 0
    bart_incorrect = 0
    bart_homer = 0
    bart_lisa = 0
    bart_marge = 0
    homer_total = 0
    homer_correct = 0
    homer_incorrect = 0
    homer_bart = 0
    homer_lisa = 0
    homer_marge = 0
    lisa_total = 0
    lisa_correct = 0
    lisa_incorrect = 0
    lisa_bart = 0
    lisa_homer = 0
    lisa_marge = 0
    marge_total = 0
    marge_correct = 0
    marge_incorrect = 0
    marge_bart = 0
    marge_homer = 0
    marge_lisa = 0

    for images, labels in test_loader:
        for i, img in enumerate(images):
            img_re = img
            output = torch.argmax(model(img_re))
            if output.data == 0:
                bart_total += 1
                if output == labels[i]:
                    bart_correct += 1
                else:
                    bart_incorrect += 1
                    if labels[i].data == 1:
                        bart_homer += 1
                    elif labels[i].data == 2:
                        bart_lisa += 1
                    elif labels[i].data == 3:
                        bart_marge += 1
            elif output.data == 1:
                homer_total += 1
                if output == labels[i]:
                    homer_correct += 1
                else:
                    homer_incorrect += 1
                    if labels[i].data == 0:
                        homer_bart += 1
                    elif labels[i].data == 2:
                        homer_lisa += 1
                    elif labels[i].data == 3:
                        homer_marge += 1
            elif output.data == 2:
                lisa_total += 1
                if output == labels[i]:
                    lisa_correct += 1
                else:
                    lisa_incorrect += 1
                    if labels[i].data == 0:
                        lisa_bart += 1
                    elif labels[i].data == 1:
                        lisa_homer += 1
                    elif labels[i].data == 3:
                        lisa_marge += 1
            elif output.data == 3:
                marge_total += 1
                if output == labels[i]:
                    marge_correct += 1
                else:
                    marge_incorrect += 1
                    if labels[i].data == 0:
                        marge_bart += 1
                    elif labels[i].data == 1:
                        marge_homer += 1
                    elif labels[i].data == 2:
                        marge_lisa += 1

            if output != labels[i]:
                img = img.swapaxes(0, 1)
                img = img.swapaxes(1, 2)
                fig = plt.gcf()
                fig.canvas.set_window_title(str(i))
                plt.imshow(img)
                plt.show()

    print("Bart Accuracy:", 100.0 * (bart_correct / bart_total))
    print("Bart Inaccuracy:", 100.0 * (bart_incorrect / bart_total))
    print("Bart-Homer:", bart_homer)
    print("Bart-Lisa:", bart_lisa)
    print("Bart-Marge:", bart_marge)
    print("Homer Accuracy:", 100.0 * (homer_correct / homer_total))
    print("Homer Inaccuracy:", 100.0 * (homer_incorrect / homer_total))
    print("Homer-Bart:", homer_bart)
    print("Homer-Lisa:", homer_lisa)
    print("Homer-Marge:", homer_marge)
    print("Lisa Accuracy:", 100.0 * (lisa_correct / lisa_total))
    print("Lisa Inaccuracy:", 100.0 * (lisa_incorrect / lisa_total))
    print("Lisa-Bart:", lisa_bart)
    print("Lisa-Homer:", lisa_homer)
    print("Lisa-Marge:", lisa_marge)
    print("Marge Accuracy:", 100.0 * (marge_correct / marge_total))
    print("Marge Inaccuracy:", 100.0 * (marge_incorrect / marge_total))
    print("Marge-Bart:", marge_bart)
    print("Marge-Homer:", marge_homer)
    print("Marge-Lisa:", marge_lisa)
    print("Total Accuracy:", 100.0 * ((bart_correct + homer_correct + lisa_correct + marge_correct) / (
            bart_total + homer_total + lisa_total + marge_total)))


# Loop through the dataset and create an text file with image paths and labels.
dataset_dir = 'dataset/train'
file = open("simpsons.txt", "w+")
label_num = 0
img_count = 1
for folder in os.listdir(dataset_dir):
      if folder == 'homer_simpson' or folder == 'bart_simpson' or folder == 'lisa_simpson' or folder == 'marge_simpson':
        label = folder
        for img in os.listdir(dataset_dir + '/' + folder):
            if img_count < 6202:
              file.write(dataset_dir + "/" + label + "/" + img + "," + str(label_num))
              file.write("\n")
              img_count += 1
        label_num += 1
file.close()

# Hyperparameters
image_size = 224
num_classes = 4

epochs = 25
batch_size = 50
hidden_neurons = 100
learning_rate = 0.001

# Use the custom class to load our dataset.
dataset = SimpsonsDataset(csv_file='simpsons.txt', transform=transforms.ToTensor())

# Split the dataset into test and train data.
train_set, test_set = torch.utils.data.random_split(dataset, [6000, 200])

# Load the test and train data, shuffle it, and specify batch size.
train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)

model = NeuralNet(image_size * image_size * 3, hidden_neurons, num_classes).to(device)
# Specify criterion and optimizer.
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# MLP loop.
writer = SummaryWriter("runs/simpsonsMLP")
n_total_steps = len(train_loader)
running_loss = 0.0
running_correct = 0
step = 0
for epoch in range(epochs):
    for i, (images, labels) in enumerate(train_loader):
        step = i
        images = images.reshape(batch_size, image_size * image_size * 3).to(device)
        labels = labels.to(device)

        outputs = model(images.float())
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f'epoch {epoch + 1} / {epochs}, step {i + 1} / {n_total_steps}, loss = {loss.item():.4f}')

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        running_correct += (predicted == labels).sum().item()
        running_samples = labels.size(0)
        acc = 100.0 * running_correct / running_samples
        writer.add_scalar('training loss', running_loss, epoch * n_total_steps + i)
        writer.add_scalar('accuracy', acc, epoch * n_total_steps + i)
        running_loss = 0.0
        running_correct = 0


    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        for images, labels in test_loader:
          images = images.reshape(batch_size, image_size * image_size * 3).to(device)
          labels = labels.to(device)
          outputs = model(images.float())
          _, predicted = torch.max(outputs.data, 1)
          n_samples += labels.size(0)
          n_correct += (predicted == labels).sum().item()


        acc = 100.0 * n_correct / n_samples
        writer.add_scalar('accuracy', acc, epoch * n_total_steps + step)
        print(f'Accuracy: {acc} %')

# Save the MLP.
name = "MLP-" + str(epochs) + "-" + str(batch_size) + "-" + str(hidden_neurons) + "-" + str(learning_rate) + ".pth.tar"
checkpoint = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
save_checkpoint(checkpoint, name)

# Create our CNN with a new criterion and optimizer.
learning_rate = 0.0001
cnn = ConvNet2().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)
running_loss = 0.0
running_correct = 0
step = 0

# CNN loop.
writer = SummaryWriter("runs/simpsonsCNN")
n_total_steps = len(train_loader)
for epoch in range(epochs):
    for i, (images, labels) in enumerate(train_loader):

        outputs = cnn(images.float())
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f'epoch {epoch + 1} / {epochs}, step {i + 1} / {n_total_steps}, loss = {loss.item():.4f}')

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        running_correct += (predicted == labels).sum().item()
        running_samples = labels.size(0)
        acc = 100.0 * running_correct / running_samples
        writer.add_scalar('training loss', running_loss, epoch * n_total_steps + i)
        writer.add_scalar('accuracy', acc, epoch * n_total_steps + i)
        running_loss = 0.0
        running_correct = 0

    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        for images, labels in test_loader:
            outputs = cnn(images.float())
            _, predicted = torch.max(outputs.data, 1)
            n_samples += labels.size(0)
            n_correct += (predicted == labels).sum().item()

        acc = 100.0 * n_correct / n_samples
        writer.add_scalar('accuracy', acc, epoch * n_total_steps + step)
        print(f'Accuracy: {acc} %')

# Save the CNN.
name = "CNN-" + str(epochs) + "-" + str(batch_size) + "-" + str(learning_rate) + ".pth.tar"
checkpoint = {'state_dict': cnn.state_dict(), 'optimizer': optimizer.state_dict()}
save_checkpoint(checkpoint, name)

writer.close()