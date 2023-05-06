from metrics.BiasAmplification import BiasAm
from metrics.groupbias import groupFairness as gb
from metrics.individual import indivisualFairness as ind
from metrics.ReverseFairness import ReverseFairness as rf
import numpy as np
import pandas as pd
import argparse


def makeargs():
    parse=argparse.ArgumentParser()
    parse.add_argument('--maad_path',type=str,default='/media/lijia/DATA/lijia/data/vggface2/anno/maad_id.csv')
    parse.add_argument('--train_csv',type=str,default="/media/lijia/DATA/lijia/data/vggface2/anno/train_id_sample_8615.csv")
    parse.add_argument('--test_csv',type=str,default="/media/lijia/DATA/lijia/data/vggface2/anno/test_id_sample_8615.csv")
    parse.add_argument('--test_pre_csv',type=str,default='/home/lijia/codes/202302/lijia/face-recognition/data/result/caam_sex/caam_sex_result_test.csv')
    parse.add_argument('--dataset_type',type=str,choices=["celeba","vggface2"],default='vggface2')
    args=parse.parse_args()
    return args

def main():
    args=makeargs()
    target_attr=['Male','Young', 'Middle_Aged', 'Senior','Asian','White','Black','Rosy_Cheeks',
                'Shiny_Skin','Bangs','Sideburns',
                'Black_Hair','Blond_Hair','Brown_Hair','Gray_Hair','No_Beard','Mustache',
                '5_o_Clock_Shadow','Goatee','Oval_Face','Square_Face','Round_Face','Double_Chin',
                'High_Cheekbones','Chubby','Obstructed_Forehead', 'Fully_Visible_Forehead','Brown_Eyes',
                'Bags_Under_Eyes','Bushy_Eyebrows','Arched_Eyebrows','Mouth_Closed','Smiling',
                'Big_Lips','Big_Nose','Pointy_Nose','Heavy_Makeup',
                'Wearing_Lipstick','Eyeglasses','Attractive']

    feature_groups = {'Gender': ['Male'],
    'Age':['Young', 'Middle_Aged', 'Senior'],
    'Race':['Asian','White','Black'],
    'Rosy_Cheeks': ['Rosy_Cheeks'],
    'Shiny_Skin': ['Shiny_Skin'],
    'Bangs':['Bangs'],
    'Sideburns':['Sideburns'],
    'HairColor': ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair'],
    'Beard': ['No_Beard','Mustache','5_o_Clock_Shadow','Goatee'],
    'FaceShape': ['Oval_Face', 'Square_Face', 'Round_Face'],
    'Double_Chin': ['Double_Chin'],
    'High_Cheekbones': ['High_Cheekbones'],
    'Chubby': ['Chubby'],
    'Forehead_visible': ['Obstructed_Forehead','Fully_Visible_Forehead'],
    'Brown_Eyes':['Brown_Eyes'],
    'Bags_Under_Eyes':['Bags_Under_Eyes'],
    'Bushy_Eyebrows':['Bushy_Eyebrows'],
    'Arched_Eyebrows':['Arched_Eyebrows'],
    'Mouth_Closed':['Mouth_Closed'],
    'Smiling':['Smiling'],
    'Big_Lips':['Big_Lips'],
    'NoseType': ['Big_Nose','Pointy_Nose'],
    'Heavy_Makeup': ['Heavy_Makeup'],
    'Eyeglasses': ['Eyeglasses'],
    'Wearing_Lipstick': ['Wearing_Lipstick'],
    'Attractive': ['Attractive']}
    
    if args.dataset_type=='celeba':
    
        target_attr=['Male','Young','Middle_Aged','Senior','Asian','White','Black','Rosy_Cheeks',
                    'Bangs','Sideburns','Black_Hair','Blond_Hair','Brown_Hair','Gray_Hair','No_Beard',
                    'Mustache','5_o_Clock_Shadow','Goatee','Oval_Face','Square_Face','Round_Face','Double_Chin',
                    'High_Cheekbones','Chubby','Obstructed_Forehead','Fully_Visible_Forehead','Brown_Eyes','Bags_Under_Eyes',
                    'Bushy_Eyebrows','Arched_Eyebrows','Mouth_Closed','Smiling','Big_Lips','Big_Nose','Pointy_Nose',
                    'Heavy_Makeup','Wearing_Lipstick','Eyeglasses','Attractive','Mouth_Slightly_Open','Narrow_Eyes','Pale_Skin']
        feature_groups = {'Gender': ['Male'],
        'Age':['Young', 'Middle_Aged', 'Senior'],
        'Race':['Asian','White','Black'],
        'Rosy_Cheeks': ['Rosy_Cheeks'],
        'Pale_Skin': ['Pale_Skin'],
        'Bangs':['Bangs'],
        'Sideburns':['Sideburns'],
        'HairColor': ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair'],
        'Beard': ['No_Beard','Mustache','5_o_Clock_Shadow','Goatee'],
        'FaceShape': ['Oval_Face', 'Square_Face', 'Round_Face'],
        'Double_Chin': ['Double_Chin'],
        'High_Cheekbones': ['High_Cheekbones'],
        'Chubby': ['Chubby'],
        'Forehead_visible': ['Obstructed_Forehead','Fully_Visible_Forehead'],
        'Brown_Eyes':['Brown_Eyes'],
        'Narrow_Eyes':['Narrow_Eyes'],
        'Bags_Under_Eyes':['Bags_Under_Eyes'],
        'Bushy_Eyebrows':['Bushy_Eyebrows'],
        'Arched_Eyebrows':['Arched_Eyebrows'],
        'Mouth_Closed_Or_Open':['Mouth_Closed','Mouth_Slightly_Open'],
        'Smiling':['Smiling'],
        'Big_Lips':['Big_Lips'],
        'NoseType': ['Big_Nose','Pointy_Nose'],
        'Heavy_Makeup': ['Heavy_Makeup'],
        'Eyeglasses': ['Eyeglasses'],
        'Wearing_Lipstick': ['Wearing_Lipstick'],
        'Attractive': ['Attractive']}
    backbone_name="vggface2-caam-resnet"

    # ba & dbana
    metric=BiasAm(args.maad_path,args.train_csv,args.test_csv,target_attribution=target_attr)
    metric.update_group(feature_groups)
    result1=metric.multi_ba(args.test_pre_csv)
    print(result1)
    np.save(backbone_name+"_ba.npy",result1)
    metric.plot_bar_result(result1,backbone_name)
    result2=metric.multi_dba(args.test_pre_csv)
    print(result2)
    metric.plot_bar_result(result2,backbone_name,False)
    np.save(backbone_name+"_dba.npy",result2)


    # groupbias
    metric_gb=gb(args.maad_path,args.test_csv,target_attribution=target_attr)
    metric_gb.update_group(feature_groups)
    result=metric_gb.GroupMetric(args.test_pre_csv)
    print(result)
    metric_gb.plot_bar_result(result,backbone_name)
    np.save(backbone_name+"_gf.npy",result)


    # individual eo
    metric_in=ind(args.maad_path,args.test_csv,target_attribution=target_attr)
    metric_in.update_group(feature_groups)
    result=metric_in.EqualOppotunity(args.test_pre_csv)
    print(result)
    metric_in.plot_bar_result(result,backbone_name)
    np.save(backbone_name+"_eo.npy",result)

    # metric_rf=rf(args.maad_path,args.train_csv,args.test_csv,target_attribution=target_attr)
    # metric_rf.update_group(feature_groups)
    # print(metric_rf.RF(args.test_pre_csv))
    
main()