import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# input: dict
def barplot(result,savepath=""):
    plt.figure(figsize=(15,15))
    plt.bar(result.keys(),result.values(),width=0.25,facecolor='lightblue',edgecolor='white',label="vggface2")
    plt.ylim([60,100])
    plt.xlabel("Backbone",fontsize=25)
    plt.ylabel("wrong predicted id in same race(%)",fontsize=25)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(fontsize=20)
    plt.show()

scores={

}

barplot(scores)