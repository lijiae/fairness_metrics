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
        test_pre_data_0=test_pre_data[test_pre_data["result"]==False]
        test_pre_data_1=test_pre_data[test_pre_data["result"]==True]

        if(self.group_attr==None):
            # single attritbutes
            return 
        else:
            groupnames=self.group_attr.keys()
            scores={}
            for groupname in groupnames:
                attr_names=self.group_attr[groupname]
                
                if(len(attr_names)>=1):
                    test_pre_group_0=test_pre_data_0.groupby(['pre_id'])[attr_names]
                    y0=test_pre_group_0.sum()/test_pre_group_0.count()-(test_pre_group_0.count()-test_pre_group_0.sum())/test_pre_group_0.count()

                    test_pre_group_1=test_pre_data_1.groupby(['pre_id'])[attr_names]
                    y1=test_pre_group_1.sum()/test_pre_group_1.count()-(test_pre_group_1.count()-test_pre_group_1.sum())/test_pre_group_1.count()

                    scores[groupname]=(abs(y1.mean())+abs(y0.mean()))/2



                