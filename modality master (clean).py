from dfply import *
from plotnine import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import rsatoolbox

possibility_data = pd.read_csv("study1possibilityData.csv")
event_data = pd.read_csv("study1eventData.csv")
could_data = pd.read_csv("study2couldData.csv")
may_data = pd.read_csv("study2mayData.csv")
might_data = pd.read_csv("study2mightData.csv")
ought_data = pd.read_csv("study2oughtData.csv")
should_data = pd.read_csv("study2shouldData.csv")
ej_could_data = pd.read_csv("upd_ej_could_data.csv")
ej_may_data = pd.read_csv("upd_ej_may_data.csv")
ej_might_data = pd.read_csv("upd_ej_might_data.csv")
ej_ought_data = pd.read_csv("upd_ej_ought_data.csv")
ej_should_data = pd.read_csv("upd_ej_should_data.csv")

# ---------------------------------------------------------------------------------------
# [CLEANING THE DF]
# NOTES

def cleanfunc(dataframe):
    dataframe['responses'] = dataframe['responses'].replace(['f'], 1) # changes value of 'f' under column 'responses' to 1
    dataframe['responses'] = dataframe['responses'].replace(['j'], 0)
    dataframe['responses'] = dataframe['responses'].replace(['timeout'], 99)
    dataframe['condition3'] = dataframe['condition3'].replace(['fast'], 'speeded')
    dataframe['condition3'] = dataframe['condition3'].replace(['slow'], 'reflective')
    dataframe['responses'] = dataframe['responses'].astype(float) # converts responses to float
    result = dataframe >> select(X.condition1, X.condition2, X.condition3, X.target, X.RTs, X.responses, X.trialNo,
                             X.turkID) \
             >> filter_by(X.condition1 != 'na') \
             >> filter_by(X.responses != 99) \
             >> mutate(trialNo=X.trialNo)
    return result

ej_could_table = cleanfunc(ej_could_data)
# print(ej_could_table)
#
# def cleanfunc(dataframe):
#     dataframe['responses'] = dataframe['responses'].replace(['f'], 1) # changes value of 'f' under column 'responses' to 1
#     dataframe['responses'] = dataframe['responses'].replace(['j'], 0)
#     # dataframe['responses'] = dataframe['responses'].astype(float) # converts responses to float
#     result = dataframe >> select(X.condition1, X.condition2, X.condition3, X.target, X.RTs, X.responses, X.trialNo,
#                              X.turkID) \
#              >> filter_by(X.condition1 != 'na') \
#              >> filter_by(X.responses != '99') \
#              >> filter_by(X.responses != 'timeout') \
#              >> mutate(trialNo=X.trialNo)
#     return result

# test
# main_data = cleanfunc(possibility_data)
# print(main_data)

# print(ej_could_table)

def cleanfunc_should(dataframe):
    dataframe['responses'] = dataframe['responses'].replace(['f'], 1) # changes value of 'f' under column 'responses' to 1
    dataframe['responses'] = dataframe['responses'].replace(['j'], 0)
    dataframe['responses'] = dataframe['responses'].astype(float) # converts responses to float
    result = dataframe[~dataframe.turkID.isin(np.r_[21, 70, 90, 93, 227])]  >> select(X.condition1, X.condition2, X.condition3, X.target, X.RTs, X.responses, X.trialNo,
                             X.turkID) \
             >> filter_by(X.condition1 != 'na') \
             >> filter_by(X.responses != 99) \
             >> mutate(trialNo=X.trialNo)
    return result

# ---------------------------------------------------------------------------------------
# [FILTERING THE DF]
# NOTES

def filterfunc(dataframe):
    dataframe['trialNo'] = dataframe['trialNo'].astype(int)
    speeded = dataframe >> filter_by(X.condition3 != 'na') \
              >> filter_by(X.RTs <= 6000) \
              >> filter_by(X.condition3 == 'speeded') \
              >> select(X.trialNo, X.responses, X.turkID, X.condition3, X.RTs) \
              >> group_by(X.turkID) \
              >> summarize(mean_RT_s=mean(X.RTs)) \
              >> filter_by(X.mean_RT_s < 800)

    reflective = dataframe >> filter_by(X.condition3 != 'na') \
                 >> filter_by(X.RTs <= 6000) \
                 >> filter_by(X.condition3 == 'reflective') \
                 >> select(X.trialNo, X.responses, X.turkID, X.condition3, X.RTs) \
                 >> group_by(X.turkID) \
                 >> summarize(mean_RT_r=mean(X.RTs)) \
                 >> filter_by(X.mean_RT_r < 1000)

    final_speeded = dataframe[~dataframe.turkID.isin(speeded.turkID)] \
                    >> filter_by(X.condition3 == 'speeded') \
                    >> filter_by(X.RTs > 500) \
                    >> select(X.turkID, X.trialNo, X.condition3, X.responses, X.RTs) \
                    >> group_by(X.trialNo) >> summarize(mean_response_sp=mean(X.responses), mean_RT_s=mean(X.RTs))

    final_reflective = dataframe[~dataframe.turkID.isin(reflective.turkID)] \
                       >> filter_by(X.condition3 == 'reflective') \
                       >> filter_by(X.RTs > 1500) \
                       >> select(X.trialNo, X.responses, X.turkID, X.condition3, X.RTs) \
                       >> group_by(X.trialNo) >> summarize(mean_response_rf=mean(X.responses), mean_RT_r=mean(X.RTs))

    final_merged = pd.merge(final_speeded, final_reflective, on=['trialNo'], how='left') >> arrange(X.trialNo, ascending=True)
    return final_merged

# test
# poss_table = cleanfunc(possibility_data)
# main_table = filterfunc(poss_table)
# print(main_table)

# ---------------------------------------------------------------------------------------
# [JOIN FUNC]
# NOTES
def joinfunc(main, data):
    result = pd.merge(main, data, on='trialNo', how='outer')
    return result

# ---------------------------------------------------------------------------------------
# [JOIN FUNC EJ]
# NOTES
def ej_joinfunc(main, data):
    result = pd.concat([main, data])
    return result

# ---------------------------------------------------------------------------------------
# [MAINTABLE DF]
# NOTES
# main_data.target.apply(lambda x: 1 if 'moral' in x else 2 if 'rational' in x else 3 if 'likely' in x else x))
# -changes the value of column 'target' containing 'moral' to 1, 'rational' to 2, 'likely' to 3
def maintable(dataframe):
    result = dataframe >> select(X.condition1, X.condition2, X.trialNo, X.target, X.responses, X.RTs) \
             >> mutate(judgment=dataframe.target.apply(
        lambda x: 1 if 'moral' in x else 2 if 'rational' in x else 3 if 'likely' in x else x)) \
             >> group_by(X.trialNo, X.condition1, X.condition2, X.judgment) \
             >> summarize(meanresponse=mean(X.responses)) \
             >> select(X.condition1, X.condition2, X.trialNo, X.judgment, X.meanresponse) \
             >> spread(X.judgment, X.meanresponse)
    result.columns = ['condition1', 'condition2', 'trialNo', 'response_m', 'response_r', 'response_l']
    return result


# test
# print(maintable(event_data))

# ---------------------------------------------------------------------------------------
# [EJ MAINTABLE DF]
def ej_maintable(dataframe):
    dataframe['trialNo'] = dataframe['trialNo'].astype(int)
    speeded = dataframe >> filter_by(X.condition3 != 'na') \
              >> filter_by(X.RTs <= 6000) \
              >> filter_by(X.condition3 == 'speeded') \
              >> select(X.trialNo, X.responses, X.turkID, X.condition3, X.RTs) \
              >> group_by(X.turkID) \
              >> summarize(mean_RT_s=mean(X.RTs)) \
              >> filter_by(X.mean_RT_s < 800)

    reflective = dataframe >> filter_by(X.condition3 != 'na') \
                 >> filter_by(X.RTs <= 6000) \
                 >> filter_by(X.condition3 == 'reflective') \
                 >> select(X.trialNo, X.responses, X.turkID, X.condition3, X.RTs) \
                 >> group_by(X.turkID) \
                 >> summarize(mean_RT_r=mean(X.RTs)) \
                 >> filter_by(X.mean_RT_r < 1000)

    final_speeded = dataframe[~dataframe.turkID.isin(speeded.turkID)] \
                    >> filter_by(X.condition3 == 'speeded') \
                    >> filter_by(X.RTs > 500) \
                    >> select(X.turkID, X.trialNo, X.condition3, X.responses, X.RTs, X.condition1, X.condition2) \
                    >> group_by(X.trialNo, X.condition1, X.condition2) >> summarize(mean_response_sp=mean(X.responses), mean_RT_s=mean(X.RTs))

    final_reflective = dataframe[~dataframe.turkID.isin(reflective.turkID)] \
                       >> filter_by(X.condition3 == 'reflective') \
                       >> filter_by(X.RTs > 1500) \
                       >> select(X.trialNo, X.responses, X.turkID, X.condition3, X.RTs) \
                       >> group_by(X.trialNo) >> summarize(mean_response_rf=mean(X.responses), mean_RT_r=mean(X.RTs))

    final_merged = pd.merge(final_speeded, final_reflective, on=['trialNo'], how='left') >> arrange(X.trialNo, ascending=True)
    return final_merged


# test
# print(maintable(event_data))

# ---------------------------------------------------------------------------------------
main = maintable(event_data)
# main.to_csv('main.csv', index=False)
# print(main)

# POSSIBILITIES
poss_table = cleanfunc(possibility_data)
poss_filter = filterfunc(poss_table) >> rename(p_resp_s=X.mean_response_sp, p_meanRT_s=X.mean_RT_s, p_resp_rf=X.mean_response_rf, p_meanRT_rf=X.mean_RT_r)
main = joinfunc(main, poss_filter)

# COULD
could_table = cleanfunc(could_data)
could_filter = filterfunc(could_table) >> rename(c_resp_s=X.mean_response_sp, c_meanRT_s=X.mean_RT_s, c_resp_rf=X.mean_response_rf, c_meanRT_rf=X.mean_RT_r)
main = joinfunc(main, could_filter)

# MAY
may_table = cleanfunc(may_data)
may_filter = filterfunc(may_table) >> rename(may_resp_s=X.mean_response_sp, may_meanRT_s=X.mean_RT_s, may_resp_rf=X.mean_response_rf, may_meanRT_rf=X.mean_RT_r)
main = joinfunc(main, may_filter)

# MIGHT
might_table = cleanfunc(might_data)
might_filter = filterfunc(might_table) >> rename(might_resp_s=X.mean_response_sp, might_meanRT_s=X.mean_RT_s, might_resp_rf=X.mean_response_rf, might_meanRT_rf=X.mean_RT_r)
main = joinfunc(main, might_filter)

# OUGHT
ought_table = cleanfunc(ought_data)
ought_filter = filterfunc(ought_table) >> rename(o_resp_s=X.mean_response_sp, o_meanRT_s=X.mean_RT_s, o_resp_rf=X.mean_response_rf, o_meanRT_rf=X.mean_RT_r)
main = joinfunc(main, ought_filter)

# SHOULD
should_table = cleanfunc_should(should_data)
should_filter = filterfunc(should_table) >> rename(s_resp_s=X.mean_response_sp, s_meanRT_s=X.mean_RT_s, s_resp_rf=X.mean_response_rf, s_meanRT_rf=X.mean_RT_r)
main = joinfunc(main, should_filter)

# EJ COULD
ej_could_table = cleanfunc(ej_could_data)
ej_could_filter = ej_maintable(ej_could_table) >> rename(c_resp_s=X.mean_response_sp, c_meanRT_s=X.mean_RT_s, c_resp_rf=X.mean_response_rf, c_meanRT_rf=X.mean_RT_r)

# EJ MAY
ej_may_table = cleanfunc(ej_may_data)
ej_may_filter = filterfunc(ej_may_table) >> rename(may_resp_s=X.mean_response_sp, may_meanRT_s=X.mean_RT_s, may_resp_rf=X.mean_response_rf, may_meanRT_rf=X.mean_RT_r)
ej_main = joinfunc(ej_could_filter, ej_may_filter)

# EJ MIGHT
ej_might_table = cleanfunc(ej_might_data)
ej_might_filter = filterfunc(ej_might_table) >> rename(might_resp_s=X.mean_response_sp, might_meanRT_s=X.mean_RT_s, might_resp_rf=X.mean_response_rf, might_meanRT_rf=X.mean_RT_r)
ej_main = joinfunc(ej_main, ej_might_filter)

# EJ OUGHT
ej_ought_table = cleanfunc(ej_ought_data)
ej_ought_filter = filterfunc(ej_ought_table) >> rename(o_resp_s=X.mean_response_sp, o_meanRT_s=X.mean_RT_s, o_resp_rf=X.mean_response_rf, o_meanRT_rf=X.mean_RT_r)
ej_main = joinfunc(ej_main, ej_ought_filter)

# EJ SHOULD
ej_should_table = cleanfunc(ej_should_data)
ej_should_filter = filterfunc(ej_should_table) >> rename(s_resp_s=X.mean_response_sp, s_meanRT_s=X.mean_RT_s, s_resp_rf=X.mean_response_rf, s_meanRT_rf=X.mean_RT_r)
ej_main = joinfunc(ej_main, ej_should_filter)
#
main = ej_joinfunc(main, ej_main)
# print(main)


main.to_csv('main_ej.csv', index=False)

# ---------------------------------------------------------------------------------------
# [CORRELATIONS]

def corrfunc(dataframe):
    data = {'reflective':[(dataframe.p_resp_rf.corr(dataframe.c_resp_rf)),
                      (dataframe.may_resp_rf.corr(dataframe.c_resp_rf)),
                      (dataframe.might_resp_rf.corr(dataframe.c_resp_rf)),
                      (dataframe.o_resp_rf.corr(dataframe.c_resp_rf)),
                      (dataframe.s_resp_rf.corr(dataframe.c_resp_rf)),
                      (dataframe.p_resp_rf.corr(dataframe.may_resp_rf)),
                      (dataframe.p_resp_rf.corr(dataframe.might_resp_rf)),
                      (dataframe.p_resp_rf.corr(dataframe.o_resp_rf)),
                      (dataframe.p_resp_rf.corr(dataframe.c_resp_rf)),
                      (dataframe.may_resp_rf.corr(dataframe.might_resp_rf)),
                      (dataframe.may_resp_rf.corr(dataframe.o_resp_rf)),
                      (dataframe.may_resp_rf.corr(dataframe.s_resp_rf)),
                      (dataframe.might_resp_rf.corr(dataframe.o_resp_rf)),
                      (dataframe.might_resp_rf).corr(dataframe.s_resp_rf),
                      (dataframe.o_resp_rf.corr(dataframe.s_resp_rf))],
            'relation':['possible-could', 'may-could', 'might-could', 'ought-could',
                        'should-could', 'possible-may', 'possible-might', 'possible-ought',
                        'possible-could', 'may-might', 'may-ought', 'may-should',
                        'might-ought', 'might-should', 'ought-should'],
            'speeded':[(dataframe.p_resp_s.corr(dataframe.c_resp_s)),
                          (dataframe.may_resp_s.corr(dataframe.c_resp_s)),
                          (dataframe.might_resp_s.corr(dataframe.c_resp_s)),
                          (dataframe.o_resp_s.corr(dataframe.c_resp_s)),
                          (dataframe.s_resp_s.corr(dataframe.c_resp_s)),
                          (dataframe.p_resp_s.corr(dataframe.may_resp_s)),
                          (dataframe.p_resp_s.corr(dataframe.might_resp_s)),
                          (dataframe.p_resp_s.corr(dataframe.o_resp_s)),
                          (dataframe.p_resp_s.corr(dataframe.c_resp_s)),
                          (dataframe.may_resp_s.corr(dataframe.might_resp_s)),
                          (dataframe.may_resp_s.corr(dataframe.o_resp_s)),
                          (dataframe.may_resp_s.corr(dataframe.s_resp_s)),
                          (dataframe.might_resp_s.corr(dataframe.o_resp_s)),
                          (dataframe.might_resp_s.corr(dataframe.s_resp_s)),
                          (dataframe.o_resp_s.corr(dataframe.s_resp_s))]}

    result = pd.DataFrame(data) >> gather('condition3', 'response', ['speeded', 'reflective'])

    return result

correlation1 = corrfunc(main)
# print(correlation)

main_corrplot = ggplot(correlation1, aes(x='condition3', y='response', color='relation')) \
    + geom_point()\
    + geom_line(aes(group='relation')) \
    + theme_classic()

# DESCRIPTIVE DATA
impossible_filter = main >> filter_by(X.condition1 == 'impossible')
improbable_filter = main >> filter_by(X.condition1 == 'improbable')
main2 = pd.concat([impossible_filter, improbable_filter])
# print(main)

correlation2 = corrfunc(main2) >> mutate(plot='descriptive')
# print(correlation2)

# PRESCRIPTIVE DATA
immoral_filter = main >> filter_by(X.condition1 == 'immoral')
irrational_filter = main >> filter_by(X.condition1 == 'irrational')
main3 = pd.concat([immoral_filter, irrational_filter])

correlation3 = corrfunc(main3) >> mutate(plot='prescriptive')

corrbind = pd.concat([correlation2, correlation3])

corrbind_corrplot = ggplot(corrbind, aes(x='condition3', y='response', color='relation')) \
                    + geom_point() \
                    + geom_line(aes(group='relation')) \
                    + facet_grid('.~plot') \
                    + theme_classic()


# ---------------------------------------------------------------------------------------

# ---------------------------------------------------------------------------------------
# [CORRELATIONS w/ EJ]

# def corrfunc(dataframe):
#     data = {'reflective':[(dataframe.p_resp_rf.corr(dataframe.c_resp_rf)),
#                       (dataframe.may_resp_rf.corr(dataframe.c_resp_rf)),
#                       (dataframe.might_resp_rf.corr(dataframe.c_resp_rf)),
#                       (dataframe.o_resp_rf.corr(dataframe.c_resp_rf)),
#                       (dataframe.s_resp_rf.corr(dataframe.c_resp_rf)),
#                       (dataframe.p_resp_rf.corr(dataframe.may_resp_rf)),
#                       (dataframe.p_resp_rf.corr(dataframe.might_resp_rf)),
#                       (dataframe.p_resp_rf.corr(dataframe.o_resp_rf)),
#                       (dataframe.p_resp_rf.corr(dataframe.c_resp_rf)),
#                       (dataframe.may_resp_rf.corr(dataframe.might_resp_rf)),
#                       (dataframe.may_resp_rf.corr(dataframe.o_resp_rf)),
#                       (dataframe.may_resp_rf.corr(dataframe.s_resp_rf)),
#                       (dataframe.might_resp_rf.corr(dataframe.o_resp_rf)),
#                       (dataframe.might_resp_rf).corr(dataframe.s_resp_rf),
#                       (dataframe.o_resp_rf.corr(dataframe.s_resp_rf))],
#             'relation':['possible-could', 'may-could', 'might-could', 'ought-could',
#                         'should-could', 'possible-may', 'possible-might', 'possible-ought',
#                         'possible-could', 'may-might', 'may-ought', 'may-should',
#                         'might-ought', 'might-should', 'ought-should'],
#             'speeded':[(dataframe.p_resp_s.corr(dataframe.c_resp_s)),
#                           (dataframe.may_resp_s.corr(dataframe.c_resp_s)),
#                           (dataframe.might_resp_s.corr(dataframe.c_resp_s)),
#                           (dataframe.o_resp_s.corr(dataframe.c_resp_s)),
#                           (dataframe.s_resp_s.corr(dataframe.c_resp_s)),
#                           (dataframe.p_resp_s.corr(dataframe.may_resp_s)),
#                           (dataframe.p_resp_s.corr(dataframe.might_resp_s)),
#                           (dataframe.p_resp_s.corr(dataframe.o_resp_s)),
#                           (dataframe.p_resp_s.corr(dataframe.c_resp_s)),
#                           (dataframe.may_resp_s.corr(dataframe.might_resp_s)),
#                           (dataframe.may_resp_s.corr(dataframe.o_resp_s)),
#                           (dataframe.may_resp_s.corr(dataframe.s_resp_s)),
#                           (dataframe.might_resp_s.corr(dataframe.o_resp_s)),
#                           (dataframe.might_resp_s.corr(dataframe.s_resp_s)),
#                           (dataframe.o_resp_s.corr(dataframe.s_resp_s))]}
#
#     result = pd.DataFrame(data) >> gather('condition3', 'response', ['speeded', 'reflective'])
#
#     return result
#
# correlation1 = corrfunc(main)
# # print(correlation)
#
# main_corrplot = ggplot(correlation1, aes(x='condition3', y='response', color='relation')) \
#     + geom_point()\
#     + geom_line(aes(group='relation')) \
#     + theme_classic()
#
# # print(main_corrplot)
#
# # DESCRIPTIVE DATA
# impossible_filter = main >> filter_by(X.condition1 == 'impossible')
# improbable_filter = main >> filter_by(X.condition1 == 'improbable')
# main2 = pd.concat([impossible_filter, improbable_filter])
#
# correlation2 = corrfunc(main2) >> mutate(plot='descriptive')
# # print(correlation2)
#
# # PRESCRIPTIVE DATA
# immoral_filter = main >> filter_by(X.condition1 == 'immoral')
# irrational_filter = main >> filter_by(X.condition1 == 'irrational')
# main3 = pd.concat([immoral_filter, irrational_filter])
#
# correlation3 = corrfunc(main3) >> mutate(plot='prescriptive')
#
# corrbind = pd.concat([correlation2, correlation3])
#
# corrbind_corrplot = ggplot(corrbind, aes(x='condition3', y='response', color='relation')) \
#                     + geom_point() \
#                     + geom_line(aes(group='relation')) \
#                     + facet_grid('.~plot') \
#                     + theme_classic()
#
# # print(corrbind_corrplot)

# ---------------------------------------------------------------------------------------
# [PCA]

# pca_data = main >> select(X.may_resp_s,
#                           X.c_resp_s, X.might_resp_s, X.o_resp_s,
#                           X.s_resp_s, X.may_resp_rf,
#                           X.c_resp_rf, X.might_resp_rf, X.o_resp_rf,
#                           X.s_resp_rf)
#
# PCA_features = ['may_resp_s', 'c_resp_s', 'might_resp_s',
#                 'o_resp_s', 's_resp_s', 'may_resp_rf',
#                 'c_resp_rf', 'might_resp_rf', 'o_resp_rf', 's_resp_rf']


pca_data = main >> select(X.p_resp_s, X.may_resp_s,
                          X.c_resp_s, X.might_resp_s, X.o_resp_s,
                          X.s_resp_s, X.p_resp_rf, X.may_resp_rf,
                          X.c_resp_rf, X.might_resp_rf, X.o_resp_rf,
                          X.s_resp_rf)

PCA_features = ['p_resp_s', 'may_resp_s', 'c_resp_s', 'might_resp_s',
                'o_resp_s', 's_resp_s', 'p_resp_rf', 'may_resp_rf',
                'c_resp_rf', 'might_resp_rf', 'o_resp_rf', 's_resp_rf']

pca_matrix = np.asmatrix(pca_data)
pca_df = pd.DataFrame(pca_matrix, columns=PCA_features)
# print(pca_df)

# Standardizing the Data
x = pca_df.loc[:, PCA_features].values
# y = pca_df.loc[:, ['trialNo']].values
x = StandardScaler().fit_transform(x)
# print(x)


# PCA Projection

pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data=principalComponents, columns=['principal component 1', 'principal component 2'])
# print(principalDf)
finalDf = pd.concat([principalDf, main[['trialNo']]], axis=1)
# print(finalDf)
finalDf['principal component 2'] = finalDf['principal component 2']*-1
result = pd.merge(main, finalDf, on='trialNo', how='left')
# print(result)
print(result.explained_variance_ratio_)

PCA_plot = ggplot(result, aes(x='principal component 1', y='principal component 2', color='condition1')) \
            + geom_text(aes(label='trialNo'), position=position_dodge(width=0.05))
#
# PCA_plot.save('PCA_withoutEJ.png')
# print(PCA_plot)

# ---------------------------------------------------------------------------------------
# # [PCA]
# # print(main)
#
# pca_data = main >> select(X.may_resp_s,
#                           X.c_resp_s, X.might_resp_s, X.o_resp_s,
#                           X.s_resp_s, X.may_resp_rf,
#                           X.c_resp_rf, X.might_resp_rf, X.o_resp_rf,
#                           X.s_resp_rf)
#
# PCA_features = ['may_resp_s', 'c_resp_s', 'might_resp_s',
#                 'o_resp_s', 's_resp_s', 'may_resp_rf',
#                 'c_resp_rf', 'might_resp_rf', 'o_resp_rf', 's_resp_rf']
#
# pca_matrix = np.asmatrix(pca_data)
# pca_df = pd.DataFrame(pca_matrix, columns=PCA_features)
# # print(pca_df)
#
# # Standardizing the Data
# x = pca_df.loc[:, PCA_features].values
# # y = pca_df.loc[:, ['trialNo']].values
# x = StandardScaler().fit_transform(x)
#
#
# # PCA Projection
#
# pca = PCA(n_components=2)
# principalComponents = pca.fit_transform(x)
# principalDf = pd.DataFrame(data=principalComponents, columns=['principal component 1', 'principal component 2'])
# # print(principalDf)
# finalDf = pd.concat([principalDf, main[['trialNo']]], axis=1)
# finalDf['principal component 2'] = finalDf['principal component 2']*-1
# result = pd.merge(main, finalDf, on='trialNo', how='left')
# # print(result)
#
# PCA_plot = ggplot(result, aes(x='principal component 1', y='principal component 2', color='condition1')) \
#             + geom_text(aes(label='trialNo'), position=position_dodge(width=0.05))
#
# # print(PCA_plot)