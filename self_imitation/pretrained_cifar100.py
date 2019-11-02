import torch, os, torchvision, torchvision.transforms as transforms
import torch.nn as nn, torch.nn.functional as F
import time, torch.optim as optim
from util import plot_train_test_accuracy
from torch.autograd import Variable
from resnet_util import BasicBlock, upsample_resnet, get_accuracy_loss, cifar100_loader

batch_size = 128
torch.manual_seed(1)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
criterion = nn.CrossEntropyLoss()
print(device)

tr_loader, tst_loader = cifar100_loader(batch_size = batch_size)

if not(os.path.isfile("pretrained.pt")):
	net = upsample_resnet(dataset = "cifar100").to(device)
	optimizer = optim.Adam(net.parameters(), lr = 0.001, betas = (0.9, 0.999), eps = 1e-08)
	scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor = 0.3162, mode = 'max', patience = 1)

	num_epochs = 5
	tr_acc_loss = []
	tst_acc_loss = []
	tm = time.time()

	for epoch in range(num_epochs):
	    for inputs, labels in tr_loader:
	    	net.train()
	    	inputs = Variable(inputs).to(device)
	    	labels = Variable(labels).to(device)
	    	optimizer.zero_grad()
	    	outputs = net(inputs)
	    	loss = criterion(outputs, labels)
	    	loss.backward()
	    	optimizer.step()
	    tr_acc_loss.append(get_accuracy_loss(net, device, tr_loader, criterion))
	    tst_acc_loss.append(get_accuracy_loss(net, device, tst_loader, criterion))
	    scheduler.step(tst_acc_loss[len(tst_acc_loss) - 1][0])
	    print('Training accuracy and loss after ' + str(epoch + 1) + ' epochs: ' + str(tr_acc_loss[len(tr_acc_loss) - 1]))
	    print('Testing accuracy and loss after ' + str(epoch + 1) + ' epochs: ' + str(tst_acc_loss[len(tst_acc_loss) - 1]))

	print('Finished Training in ' + str(round(time.time() - tm, 4)))
	tr_acc = [tpl[0] for tpl in tr_acc_loss]
	tst_acc = [tpl[0] for tpl in tst_acc_loss]
	tr_loss = [tpl[1] for tpl in tr_acc_loss]
	tst_loss = [tpl[1] for tpl in tst_acc_loss]
	plot_train_test_accuracy(train_acc = tr_acc, test_acc = tst_acc, label = "_cifar100_pretrained_accuracy")
	plot_train_test_accuracy(train_acc = tr_loss, test_acc = tst_loss, label = "_cifar100_pretrained_loss")
	torch.save(net, "pretrained.pt")
else:
	net = torch.load("pretrained.pt")

tr_acc_loss_1 = get_accuracy_loss(net, device, tr_loader, criterion)
tst_acc_loss_1 = get_accuracy_loss(net, device, tst_loader, criterion)
print('Final training accuracy and loss: ' + str(tr_acc_loss_1))
print('Final testing accuracy and loss: ' + str(tst_acc_loss_1))