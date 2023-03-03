import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


class BiasAm():
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
            
        maad=pd.read_csv(maad_path)
        if 'id' in maad.columns:
            maad=maad.drop('id',axis=1)
        train_id=pd.read_csv(train_path)
        test_id=pd.read_csv(test_path)
        self.maad=maad
        self.train_data=maad.merge(train_id,on='Filename')
        self.test_data=maad.merge(test_id,on='Filename')
        self.group_attr=None
    
    def update_group(self,group_attr):
        self.group_attr=group_attr

    def BA(self,test_pre_path):
        test_pre_data=pd.read_csv(test_pre_path)
        test_pre_data=self.maad.merge(test_pre_data,on='Filename')
        test_pre_group=test_pre_data.groupby('pre_id')[self.target_attributions]
        train_group=self.train_data.groupby('id')[self.target_attributions]
        pat=train_group.sum()/train_group.count()
        yat=pat.copy()
        yat[yat>0.5]=1
        yat[yat<=0.5]=0
        pat_hat=test_pre_group.sum()/test_pre_group.count()
        result=(pat_hat-pat)*yat
        result=result.dropna(axis=0,how='all')
        # result_table=pd.DataFrame(result.sum(axis=0)/result.shape[0])
        return result.sum(axis=0)/result.shape[0]

    def DBA_A_to_T(self,test_pre_path):
        test_pre_data=pd.read_csv(test_pre_path)
        test_pre_data=self.maad.merge(test_pre_data,on='Filename')
        test_pre_group=test_pre_data.groupby('pre_id')[self.target_attributions]
        # get ids in test data
        train_data_filter=self.train_data[self.train_data["id"].isin(test_pre_group.indices.keys())]
        train_group=train_data_filter.groupby('id')[self.target_attributions]

        p_at=train_group.sum()/train_data_filter.shape[0]
        p_a=(train_data_filter[self.target_attributions].sum()/train_data_filter.shape[0]).values
        p_t=(train_group.count()/train_data_filter.shape[0]).values[:,0]
        y_at=p_at.values-p_t.reshape(p_t.shape[0],1).dot(p_a.T.reshape(1,p_a.shape[0]))
        y_at[y_at>0]=1
        y_at[y_at<=0]=0

        # test_group=self.test_data.groupby('pre_id')[self.target_attributions]
        # get ids in test data
        p_haty_a=(test_pre_group.sum()[self.target_attributions]/test_pre_data[self.target_attributions].sum()).values
        p_y_a=(test_pre_data.sum()[self.target_attributions]/test_pre_data[self.target_attributions].sum()).values
        delta=p_haty_a-p_y_a

        result=y_at*delta-(1-y_at)*delta
        result=result.sum(axis=0)/(len(test_pre_group)*2)
        #

        return result

    def plot_result(self,result,type='ba'):
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
        plt.savefig('biasmetric_{}.png'.format(type))
        plt.show()

    def multi_ba(self,test_pre_path):
        test_pre_data=pd.read_csv(test_pre_path)
        test_pre_data=test_pre_data.merge(self.test_data,on="Filename")
        test_pre_data=test_pre_data[test_pre_data["id"]==test_pre_data["pre_id"]].drop("id",axis=1)
        # test_pre_data=(test_pre_data[test_pre_data["id"]==test_pre_data["pre_id"]]).drop("id",axis=1)
        # test_pre_data=self.maad.merge(test_pre_data,on='Filename')

        groupnames=self.group_attr.keys()

        scores={}
        for groupname in groupnames:
            attr_names=self.group_attr[groupname]
            test_pre_group=test_pre_data.groupby(['pre_id'])[attr_names]
            train_group=self.train_data.groupby('id')[attr_names]

            propotion=1.0/len(attr_names)
            if(propotion==1):
                propotion=propotion/2

            if(len(attr_names)<=1):
                pat=train_group.mean()
                yat=pat.copy()
                yat[yat>propotion]=1
                yat[yat<=propotion]=0
                pat_hat=test_pre_group.mean()
                result=(pat_hat-pat)*yat
                ba_score_1=result.mean()
                
                # pat_0=1-pat
                # yat_0=pat_0.copy()
                # yat_0[yat_0>propotion]=1
                # yat_0[yat_0<=propotion]=0
                # pat_hat_0=1-pat_hat
                # result_0=(pat_hat_0-pat_0)*yat_0
                # ba_score_0=result_0.mean()
                
                # scores[groupname]=(ba_score_1+ba_score_0).values[0]
                scores[groupname]=ba_score_1.values[0]
                
                
            else:
                pat=train_group.sum()/train_group.count()
                yat=pat.copy()
                yat[yat>propotion]=1
                yat[yat<=propotion]=0
                pat_hat=test_pre_group.sum()/test_pre_group.count()
                result=(pat_hat-pat)*yat
                # result=result.dropna(axis=0,how='all')
                # result_table=pd.DataFrame(result.sum(axis=0)/result.shape[0])
                # ba_score=result.sum(axis=0)/result.shape[0]
                # ba_score=result.mean()
                scores[groupname]=result.mean().values.sum()

        return scores

    def multi_dba(self,test_pre_path,amplify_factor=100):
        # print("multi-group is working")
        test_pre_data=pd.read_csv(test_pre_path)
        test_pre_data=test_pre_data.merge(self.test_data,on="Filename")
        test_pre_data=test_pre_data[test_pre_data["id"]==test_pre_data["pre_id"]].drop("id",axis=1)

        groupnames=self.group_attr.keys()

        scores={}
        for groupname in groupnames:
            attr_names=self.group_attr[groupname]
            test_pre_group=test_pre_data.groupby(['pre_id'])[attr_names]
            train_group=self.train_data.groupby('id')[attr_names]
            train_data=self.train_data[attr_names]

            propotion=1.0/len(attr_names)
            if(propotion==1):
                propotion=propotion/2

            if(len(attr_names)<=1):
                p_at=train_group.sum()/train_data.shape[0]
                p_a=(train_data.sum()/train_data.shape[0]).values
                p_t=(train_group.count()/train_data.shape[0]).values[:,0]
                y_at=p_at.values-p_t.reshape(p_t.shape[0],1).dot(p_a.T.reshape(1,p_a.shape[0]))
                y_at[y_at>0]=1
                y_at[y_at<=0]=0
                p_haty_a=test_pre_group.sum()/test_pre_group.sum().sum()
                p_y_a=train_group.sum()/train_data.sum()
                delta=(p_haty_a-p_y_a)
                result=y_at*delta-(1-y_at)*delta
                ba_score_1=result.mean()
                scores[groupname]=ba_score_1[0]

            else:
                p_at=train_group.sum()/train_data.shape[0]
                p_a=(train_data.sum()/train_data.shape[0]).values
                p_t=(train_group.count()/train_data.shape[0]).values[:,0]
                y_at=p_at.values-p_t.reshape(p_t.shape[0],1).dot(p_a.T.reshape(1,p_a.shape[0]))
                y_at[y_at>0]=1
                y_at[y_at<=0]=0
                p_haty_a=test_pre_group.sum()/test_pre_group.sum().sum()
                p_y_a=train_group.sum()/train_data.sum()
                delta=(p_haty_a-p_y_a)
                result=y_at*delta-(1-y_at)*delta
                # result_table=pd.DataFrame(result.sum(axis=0)/result.shape[0])
                # ba_score=result.sum(axis=0)/result.shape[0]
                scores[groupname]=result.mean().sum()

        return scores
    
    def plot_bar_result(self,result,filename="",isBA=True):
        plt.figure(figsize=(60, 50))
        plt.bar(result.keys(), result.values())
        
        nametype=""
        if isBA:
            plt.ylabel("BA",size=40)
            nametype="BA"
        else:
            plt.ylabel("DBA",size=40)
            nametype="DBA"
        plt.xlabel("Attributes",size=40)
        plt.yticks(size=40)
        plt.xticks(size=35,rotation=60)
        plt.savefig(nametype+"_bar_"+filename+'.png')
        plt.show()
        