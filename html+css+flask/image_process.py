import io
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as tv

class ResidualBlock(nn.Module):
    def __init__(self,in_channels,out_channels,stride=1):
        super(ResidualBlock,self).__init__()
        self.conv1=nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=stride,padding=1,bias=False)
        self.bn1=nn.BatchNorm2d(out_channels)
        self.conv2=nn.Conv2d(out_channels,out_channels,kernel_size=3,stride=1, padding=1, bias=False)
        self.bn2=nn.BatchNorm2d(out_channels)
        self.relu=nn.ReLU(inplace=True)
        self.stride=stride
        if stride!=1 or in_channels!=out_channels:
            self.residual=nn.Sequential(
                nn.Conv2d(in_channels,out_channels,kernel_size=1,stride=stride,bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.residual=nn.Identity()

    def forward(self,x):
        identity=self.residual(x)
        out=self.conv1(x)
        out=self.bn1(out)
        out=self.relu(out)
        out=self.conv2(out)
        out=self.bn2(out)
        out+=identity
        out=self.relu(out)
        return out

class ConvResNet(nn.Module):
    def __init__(self,in_channels,num_classes):
        super(ConvResNet,self).__init__()
        self.conv1=nn.Conv2d(in_channels,64,kernel_size=3,stride=1,padding=1,bias=False)
        self.bn1=nn.BatchNorm2d(64)
        self.relu=nn.ReLU(inplace=True)
        self.residual1=ResidualBlock(64,64)
        self.conv2=nn.Conv2d(64,128,kernel_size=3,stride=2,padding=1,bias=False)
        self.bn2=nn.BatchNorm2d(128)
        self.residual2=ResidualBlock(128,128)
        self.conv3=nn.Conv2d(128,256,kernel_size=3,stride=2,padding=1,bias=False)
        self.bn3=nn.BatchNorm2d(256)
        self.residual3=ResidualBlock(256,256)
        self.avg_pool=nn.AdaptiveAvgPool2d((1,1))
        self.fc=nn.Linear(256,num_classes)

    def forward(self, x):
        x=self.conv1(x)
        x=self.bn1(x)
        x=self.relu(x)
        x=self.residual1(x)
        x=self.conv2(x)
        x=self.bn2(x)
        x=self.relu(x)
        x=self.residual2(x)
        x=self.conv3(x)
        x=self.bn3(x)
        x=self.relu(x)
        x=self.residual3(x)
        x=self.avg_pool(x)
        x=torch.flatten(x, 1)
        x=self.fc(x)
        return x

def predict_class(img_bytes):
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    model_wt=torch.load(r'residual-model-final.pth',map_location=torch.device('cpu'))
    model=ConvResNet(3,3)
    model.load_state_dict(model_wt)
    model.eval()
    transform = tv.Compose([
        tv.Resize((224,224)),
        tv.ToTensor(),
        tv.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))
    ])
    img_t=transform(img)
    img_t=img_t.unsqueeze(0)
    with torch.no_grad():
        out=model(img_t.float())
        probabilities=F.softmax(out,dim=1)
        predicted_class=torch.argmax(probabilities).item()
    if predicted_class==0:
        return "Water Asphalt Smooth"
    elif predicted_class==1:
        return "Water Concrete Smooth"
    else:
        return "Water Gravel"

if __name__ == "__main__":
    print('This is a helper script for the Flask app.')
