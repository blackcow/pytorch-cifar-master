import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import cifar10my
bs = 10
transform=transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
trainset=cifar10my.CIFAR10MY(root='./data',train=True,download=True,transform=transform)
# trainloader=torch.utils.data.DataLoader(trainset,batch_size=bs,shuffle=True,num_workers=2)
trainloader=torch.utils.data.DataLoader(trainset,batch_size=bs,shuffle=False,num_workers=2)

testset = cifar10my.CIFAR10MY(root='./data', train=False,download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=bs,shuffle=False, num_workers=1)

classes = ('airplane', 'automobile', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
print('ok')

def imshow(img):
    img=img/2+0.5
    npimg=img.numpy()
    plt.imshow(np.transpose(npimg,(1,2,0)))
    plt.show()

# dataiter=iter(trainloader)
# images,labels=dataiter.next()
#
# imshow(torchvision.utils.make_grid(images))
# print(''.join('%5s' % classes[labels[j]] for j in range(4)))

if __name__ == '__main__':
    dataiter=iter(trainloader)
    # dataiter=iter(testloader)

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        # 500 轮为一个 label
        if batch_idx == 500:
            # images, labels = dataiter.next()
            print(' '.join('%5s' % classes[targets[j]] for j in range(bs)))
            imshow(torchvision.utils.make_grid(inputs))

