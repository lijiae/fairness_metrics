import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


class ReverseFairness():
    def __init__(self,maad_path,train_path,test_path,target_attribution=None):
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
        train_id=pd.read_csv(train_path)
        test_id=pd.read_csv(test_path)
        self.maad=maad
        self.train_data=maad.merge(train_id,on='Filename')
        self.test_data=maad.merge(test_id,on='Filename')
        self.group_attr=None
    
    def update_group(self,group_attr):
        self.group_attr=group_attr
        
    
    def RF(self,test_pre_path):
        test_pre_data=self.test_data.merge(pd.read_csv(test_pre_path),on='Filename')
        test_pre_data["result"]=test_pre_data["id"]==test_pre_data["pre_id"]
        test_pre_data=test_pre_data[test_pre_data["result"]==False]