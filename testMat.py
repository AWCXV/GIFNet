import scipy.io as scio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def showLossChart(path,savedName):
    plt.cla();
    plt.clf();
    if (path == ""):
        return;
    #path = 'savedLossMat/LossMat_Epoch:_(5_20)_Step:_(450_590).mat'
    
    data = scio.loadmat(path)
    #type(data)

    #print(dict)  # data是字典格式

    #print(data.keys())  # 查看字典的键

    #loss =data['finalLoss'][0];
    loss =data['Loss'][0];

    #print(loss);

    x_data = range(0,len(loss));
    y_data = loss;

    plt.plot(x_data,y_data);
    plt.xlabel("Step");
    plt.ylabel("Loss");
    plt.savefig(savedName);


showLossChart("","test.png");