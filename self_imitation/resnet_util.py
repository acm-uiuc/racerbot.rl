import torch, os, torchvision, torch.nn as nn, torchvision.transforms as transforms, torchvision.datasets as datasets, pandas as pd, numpy as np, matplotlib.pyplot as plt

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}

def get_accuracy_loss(net, device, data_loader, criterion):
    with torch.no_grad():
        net.eval()
        correct = 0
        total = 0
        loss = 0
        distr = {}
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            loss += criterion(outputs, labels) * labels.size(0)
            labels1 = pd.Series(labels.cpu()).value_counts()
    
    accuracy = correct / total
    loss = loss.item() / total
    return (round(accuracy, 4), round(loss, 4))

def get_MC_accuracy_loss(net, device, data_loader, criterion, num_MC_runs = 10):
    with torch.no_grad():
        correct = 0
        total = 0
        loss = 0
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = 0
            for i in range(num_MC_runs):
                outputs += net(images)/num_MC_runs
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            loss += criterion(outputs, labels) * labels.size(0)
    
    accuracy = correct / total
    loss = loss.item() / total
    return (round(accuracy, 4), round(loss, 4))

class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride = 1, downsample = None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size = 3, stride = stride, padding = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace = True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
    
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes = 100, dropout_p = 0.3, fc_dim = 3, dataset = "cifar100"):
        super(ResNet, self).__init__()
        self.inplanes = 32
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace = True)
        # self.maxpool = nn.MaxPool2d(kernel_size = 2, stride = 1, padding = 0)
        self.layer1 = self._make_layer(block, 32, layers[0])
        self.layer2 = self._make_layer(block, 64, layers[1], stride = 2)
        self.layer3 = self._make_layer(block, 128, layers[2], stride = 2)
        self.layer4 = self._make_layer(block, 256, layers[3], stride = 2)
        self.maxpool1 = nn.MaxPool2d(kernel_size = 4, stride = 1, padding = 0)
        self.maxpool2 = nn.MaxPool2d(kernel_size = 3, stride = 1, padding = 0)
        self.fc = nn.Linear(256 * fc_dim * fc_dim, num_classes)
        self.dropout = nn.Dropout2d(p = dropout_p)
        self.dataset = dataset
    
    def _make_layer(self, block, planes, blocks, stride = 1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size = 1, stride = stride, bias = False),
                nn.BatchNorm2d(planes),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)
    
    def forward(self, x):
    	x = self.conv1(x)
    	x = self.bn1(x)
    	x = self.relu(x)
    	x = self.dropout(x)
    	x = self.layer1(x)
    	x = self.layer2(x)
    	x = self.layer3(x)
    	x = self.layer4(x)
    	if self.dataset == "tinyimagenet":
	    	x = self.maxpool1(x)
	    	x = self.maxpool2(x)
    	x = x.view(x.size(0), -1)
    	x = self.fc(x)
    	return x

def cifar10_loader(data_path = "./data",
    transforms_list = [transforms.RandomHorizontalFlip(), transforms.RandomCrop(size = [32, 32], padding = 4), transforms.ToTensor()]):
    data_aug = transforms.Compose(transforms_list)
    to_tensor = transforms.Compose([transforms.ToTensor()])
    train_dat = torchvision.datasets.CIFAR10(root='./data', train = True, download = True, transform = data_aug)
    tr_loader = torch.utils.data.DataLoader(train_dat, batch_size = 256, shuffle = True, num_workers = 2)
    test_dat = torchvision.datasets.CIFAR10(root = './data', train = False, download = True, transform = to_tensor)
    tst_loader = torch.utils.data.DataLoader(test_dat, batch_size = 256, shuffle = False, num_workers = 2)
    return tr_loader, tst_loader

def cifar100_loader(data_path = "./data/", batch_size = 256,
	transform_list = [transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding = 4), transforms.ToTensor()]):
	transform_train = transforms.Compose(transform_list)
	trainset = torchvision.datasets.CIFAR100(root = data_path, train = True, download = False, transform = transform_train)
	tr_loader = torch.utils.data.DataLoader(trainset, batch_size = batch_size, shuffle = True, num_workers = 8)
	# For testing data
	transform_test = transforms.Compose([transforms.ToTensor()])
	testset = torchvision.datasets.CIFAR100(root = data_path, train = False, download = False, transform = transform_test)
	tst_loader = torch.utils.data.DataLoader(testset, batch_size = batch_size, shuffle = False, num_workers = 8)
	return tr_loader, tst_loader

def create_val_folder(val_dir):
    """
    This method is responsible for separating validation
    images into separate sub folders
    """
    # path where validation data is present now
    path = os.path.join(val_dir, 'images')
    # file where image2class mapping is present
    filename = os.path.join(val_dir, 'val_annotations.txt')
    fp = open(filename, "r") # open file in read mode
    data = fp.readlines() # read line by line
    '''
    Create a dictionary with image names as key and
    corresponding classes as values
    '''
    val_img_dict = {}
    for line in data:
        words = line.split("\t")
        val_img_dict[words[0]] = words[1]
    fp.close()
    # Create folder if not present, and move image into proper folder
    for img, folder in val_img_dict.items():
        newpath = (os.path.join(path, folder))
        if not os.path.exists(newpath): # check if folder exists
            os.makedirs(newpath)
        # Check if image exists in default directory
        if os.path.exists(os.path.join(path, img)):
	        os.rename(os.path.join(path, img), os.path.join(newpath, img))
    return

def tinyimagenet_loader(batch_size = 256, train_dir = 'data/tiny-imagenet-200/train', val_dir = 'data/tiny-imagenet-200/val/images',
	transforms_list = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomCrop(64, padding = 4), transforms.ToTensor()])):
	train_dataset = datasets.ImageFolder(train_dir, transform = transforms_list)
	train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, shuffle = True, num_workers = 8)
	if 'val_' in os.listdir(val_dir)[0]:
	    create_val_folder(val_dir)
	else:
	    pass
	val_dataset = datasets.ImageFolder(val_dir, transform = transforms.ToTensor())
	val_loader = torch.utils.data.DataLoader(val_dataset, batch_size = batch_size, shuffle = False, num_workers = 8)
	return train_loader, val_loader

def resnet18(pretrained = True):
	model = torchvision.models.resnet.ResNet(torchvision.models.resnet.BasicBlock, [2, 2, 2, 2])
	if pretrained:
		model.load_state_dict(torch.utils.model_zoo.load_url(model_urls['resnet18'], model_dir = './'))
	return model


def upsample_resnet(num_classes = 100, use_pretrained = True, dataset = "cifar100"):
    resnet_18_model = resnet18(pretrained = True)
    feature_count = resnet_18_model.fc.in_features
    resnet_18_model.fc = nn.Linear(feature_count, num_classes)
    if dataset == "cifar100":
    	upsample = nn.Upsample(scale_factor = 7, mode = 'bilinear')
    elif dataset == "tinyimagenet":
    	upsample = nn.Upsample(scale_factor = 3.5, mode = 'bilinear')
    resnet_18_model = nn.Sequential(upsample, resnet_18_model)
    return resnet_18_model

def get_dataset_from_loader(data_loader):
	x = []
	y = np.array([])
	for images, labels in data_loader:
		images = images.to('cpu')
		images = np.array(images)
		x.append(images)
		labels = labels.to('cpu')
		labels = np.array(labels)
		y = np.append(y, labels)
	x = np.vstack(x)
	print(y.shape)
	# y = y.reshape((y.shape[0] * y.shape[1], ))
	return x, y

def load_cifar100_numpy(data_path = "./data/", batch_size = 256,
	transform_list = [transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding = 4), transforms.ToTensor()]):
	transform_train = transforms.Compose(transform_list)
	trainset = torchvision.datasets.CIFAR100(root = data_path, train = True, download = False, transform = transform_train)
	tr_loader = torch.utils.data.DataLoader(trainset, batch_size = batch_size, shuffle = True, num_workers = 8)
	transform_test = transforms.Compose([transforms.ToTensor()])
	testset = torchvision.datasets.CIFAR100(root = data_path, train = False, download = False, transform = transform_test)
	tst_loader = torch.utils.data.DataLoader(testset, batch_size = batch_size, shuffle = False, num_workers = 8)
	x_train, y_train = get_dataset_from_loader(tr_loader)
	x_test, y_test = get_dataset_from_loader(tst_loader)
	return {"X_train": x_train, "Y_train": y_train, "X_test": x_test, "Y_test": y_test}

def plot_sync_accuracies(txt_file):
	df = pd.read_csv(txt_file, sep = " ", header = None)
	df.columns = ['epoch', 'test_accuracy', 'train_accuracy', 'time']
	df = df[['epoch', 'test_accuracy', 'train_accuracy']].groupby(['epoch']).mean().reset_index(drop = False)
	df = df.drop(['epoch'], axis = 1)
	df.plot()
	plt.savefig("train_test_cifar100_sync_accuracy.png")
	plt.show()
