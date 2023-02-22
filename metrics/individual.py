import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

class indivisualFairness():
    def __init__(self,maad_path,test_path,target_attribution=None):
        if target_attribution:
            self.target_attributions=target_attribution
        else:
            self.target_attributions= ['Male','Young', 'Middle_Aged', 'Senior','Asian','White','Black','Rosy_Cheeks',
                'Shiny_Skin','Bald','Wavy_Hair','Receding_Hairline','Bangs','Sideburns',
                'Black_Hair','Blond_Hair','Brown_Hair','Gray_Hair','No_Beard','Mustache',
                '5_o_Clock_Shadow','Goatee','Oval_Face','Square_Face','Round_Face','Double_Chin',
                'High_Cheekbones','Chubby','Obstructed_Forehead', 'Fully_Visible_Forehead','Brown_Eyes',
                'Bags_Under_Eyes','Bushy_Eyebrows','Arched_Eyebrows','Mouth_Closed','Smiling',
                'Big_Lips','Big_Nose','Pointy_Nose','Heavy_Makeup','Wearing_Hat','Wearing_Earrings',
                'Wearing_Necktie','Wearing_Lipstick','No_Eyewear','Eyeglasses','Attractive']

        maad=pd.read_csv(maad_path).drop('id',axis=1)
        test_id=pd.read_csv(test_path)

        # self.maad=maad
        self.group_attr=None
        self.test_data=maad.merge(test_id,on='Filename')


    def update_group(self,group_attr):
        self.group_attr=group_attr

    def EqualOppotunity(self,test_pre_path):
        test_pre_data=self.test_data.merge(pd.read_csv(test_pre_path),on='Filename')
        test_pre_data["result"]=test_pre_data["id"]==test_pre_data["pre_id"]
        
        test_group_p_1=test_pre_data.groupby("id")[self.target_attributions].mean()
        test_group_p_0=1-test_group_p_1
        
        test_maad_content=test_pre_data[self.target_attributions]
        test_pre_content=test_pre_data["result"].values
        
        result_1=test_pre_content.reshape(test_pre_content.shape[0],1)*test_maad_content
        result_1.insert(0,"id",test_pre_data["id"])
        result_1_groupby=result_1.groupby("id")[self.target_attributions]
        acc_1=(result_1_groupby.mean()/test_group_p_1).mean()
        
        result_0=test_pre_content.reshape(test_pre_content.shape[0],1)*(1-test_maad_content)
        result_0.insert(0,"id",test_pre_data["id"])
        result_0_groupby=result_0.groupby("id")[self.target_attributions]
        acc_0=(result_0_groupby.mean()/test_group_p_0).mean()
        
        score_dict={}
        for gn,attrlist in self.group_attr.items():
            if len(attrlist)==1:
                score_dict[gn]=np.array([acc_1[attrlist[0]],acc_0[attrlist[0]]]).std()
            else:
                score_np=acc_1[attrlist].values
                score_dict[gn]=score_np.std()
        return score_dict


    def plot_result(self,result,filename=""):
        plt.figure(figsize=(40, 40))
        font = {
            'weight': 'normal',
            'size': 20
        }
        plt.xlabel(type, font)
        # if type=="multi":
        #     xlabel=self.group_attr.keys()
        # else:
        #     xlabel=self.target_attributions
        plt.plot(result.values(), result.keys(), linewidth=3, marker='o', markersize=15)
        plt.yticks(size=25)
        plt.xticks(size=15)
        plt.savefig(filename+"_"+'groupeo_vggface2.png')
        plt.show()
        
    def plot_bar_result(self,result,filename=""):
        plt.figure(figsize=(60, 50))
        plt.bar(result.keys(), result.values())
        plt.ylabel("EO",size=40)
        
        plt.xlabel("Attributes",size=40)
        plt.yticks(size=40)
        plt.xticks(size=35,rotation=60)
        plt.savefig("indivisual_bar_"+filename+'.png')
        plt.show()