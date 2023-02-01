from metrics.BiasAmplification import BiasAm
from metrics.groupbias import groupFairness as gb
import numpy as np
import pandas as pd
import argparse


def makeargs():
    parse=argparse.ArgumentParser()
    parse.add_argument('--maad_path',type=str,default='/exdata/data/vggface2/maad_id.csv')
    parse.add_argument('--train_csv',type=str,default='/exdata/data/vggface2/train_id_sample.csv')
    parse.add_argument('--test_csv',type=str,default='/exdata/data/vggface2/test_id_sample.csv')
    parse.add_argument('--test_pre_csv',type=str,default='data/test_pre_id.csv')
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
    
    # ba & dbana
    # metric=BiasAm(args.maad_path,args.train_csv,args.test_csv,target_attribution=target_attr)
    # metric.update_group(feature_groups)
    # result1=metric.multi_ba(args.test_pre_csv)
    # metric.plot_result(result1,type="ba")
    # result2=metric.multi_dba(args.test_pre_csv)
    # metric.plot_result(result2,type="dba")

    # eo & eodds
    metric_gb=gb(args.maad_path,args.test_csv,target_attribution=target_attr)
    metric_gb.update_group(feature_groups)
    result=metric_gb.GroupMetric(args.test_pre_csv)


main()