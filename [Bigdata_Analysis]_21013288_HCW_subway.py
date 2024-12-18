import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


file_path_2line = 'C:\\Users\\USER\\Desktop\\changwoo\\BigData_Analysis\\termproject_v1\\2023 2호선 지하철 수송인원.xlsx'
trans_p_2line = pd.read_excel(file_path_2line)
file_path_5line = 'C:\\Users\\USER\\Desktop\\changwoo\\BigData_Analysis\\termproject_v1\\2023 5호선 지하철 수송인원.xlsx'
trans_p_5line = pd.read_excel(file_path_5line)
file_path_7line = 'C:\\Users\\USER\\Desktop\\changwoo\\BigData_Analysis\\termproject_v1\\2023 7호선 지하철 수송인원.xlsx'
trans_p_7line = pd.read_excel(file_path_7line)
file_path_body = 'C:\\Users\\USER\\Desktop\\changwoo\\BigData_Analysis\\termproject_v1\\신체 길이 직접 측정.xlsx'
body_2020 = pd.read_excel(file_path_body)
file_path_congestion = 'C:\\Users\\USER\\Desktop\\changwoo\\BigData_Analysis\\termproject_v1\\서울교통공사_지하철혼잡도정보_20231231.csv'
congestion_2023 = pd.read_csv(file_path_congestion,encoding='cp949')
# print(trans_p_2line.shape)
# print(trans_p_7line.shape)
# print(body_2021.shape)
# print(congestion_2023.shape)
# print(trans_p_2line.head())
# print(trans_p_7line.head())
# print(body_2021.head())
# print(congestion_2023.head())
# print(trans_p_2line.isnull().sum())
# print(trans_p_7line.isnull().sum())
# print(body_2021.isnull().sum())
# print(congestion_2023.isnull().sum())
# print(congestion_2023.head())
# print(congestion_2023.describe())

# print(trans_p.describe())
## data check

# print(trans_p.head())
# print(trans_p.describe())
# print(trans_p.head())

####################################################################  지하철 수송량 #########################################################################
## data preprocessing
# trans_p_2line = trans_p_2line.sort_values('일평균',ascending=False)
# columns_to_drop = [col for col in trans_p_2line.columns if '월' in col]
# trans_p_2line_dropped = trans_p_2line.drop(columns = columns_to_drop)
# trans_p_2line_dropped = trans_p_2line_dropped.drop(['계'],axis=1)
# # print(trans_p_2line_dropped.head())

# trans_p_7line = trans_p_7line.sort_values('일평균',ascending=False)
# columns_to_drop = [col for col in trans_p_7line.columns if '월' in col]
# trans_p_7line_dropped = trans_p_7line.drop(columns = columns_to_drop)
# trans_p_7line_dropped = trans_p_7line_dropped.drop(['계'],axis=1)
# print(trans_p_7line_dropped.head())


# # '지하철역' 열의 데이터를 섞기
# shuffled_indices = np.random.permutation(trans_p_2line_dropped.index)
# trans_p_2line_dropped = trans_p_2line_dropped.loc[shuffled_indices].reset_index(drop=True)

# shuffled_indices_2 = np.random.permutation(trans_p_7line_dropped.index)
# trans_p_7line_dropped = trans_p_7line_dropped.loc[shuffled_indices_2].reset_index(drop=True)

# ## visualization
# plt.rcParams['font.family'] ='Malgun Gothic'
# plt.rcParams['axes.unicode_minus'] =False

# # # Figure 생성
# fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))

# ## 첫 번째 서브플롯 (2호선)
# station_2_x = trans_p_2line_dropped['지하철역']
# station_2_y = trans_p_2line_dropped['일평균']
# max_value_2 = station_2_y.max() # 최댓값 구하기
# max_index_2 = station_2_y.idxmax()
# ax1.bar(station_2_x, station_2_y, color='darkkhaki', alpha=0.7) # 바그래프 
# ax1.bar(station_2_x[max_index_2], max_value_2, color='greenyellow') # 최댓값 표시
# ax1.plot(station_2_x, station_2_y, marker='o', linestyle='-', color='olive') # 라인그래프
# ax1.set_title('2호선 일평균 수송량', fontweight='bold',fontsize=18)  # 서브플롯 제목
# ax1.set_xlabel('지하철역',fontsize=18)  # x축 제목
# ax1.set_ylabel('일평균 수송량',fontsize=18)  # y축 제목
# ax1.set_xticklabels(station_2_x, rotation=90, ha='right')  # x축 눈금 라벨 회전

# # 두 번째 서브플롯 (7호선)
# station_7_x = trans_p_7line_dropped['지하철역']
# station_7_y = trans_p_7line_dropped['일평균']
# max_value_7 = station_7_y.max() # 최댓값 구하기
# max_index_7 = station_7_y.idxmax()
# ax2.bar(station_7_x, station_7_y, color='forestgreen', alpha=0.7)# 바그래프
# ax2.bar(station_7_x[max_index_7], max_value_7, color='aquamarine') # 최댓값 표시
# ax2.plot(station_7_x, station_7_y, marker='o', linestyle='-', color='limegreen')# 라인그래프
# ax2.set_title('7호선 일평균 수송량', fontweight='bold',fontsize=18)  # 서브플롯 제목
# ax2.set_xlabel('지하철역',fontsize=18)  # x축 제목
# ax2.set_ylabel('일평균 수송량',fontsize=18)  # y축 제목
# ax2.set_xticklabels(station_7_x, rotation=90, ha='right')  # x축 눈금 라벨 회전

# # 전체 제목 설정
# fig.suptitle('2호선 및 7호선의 일평균 수송량', fontweight='bold', fontsize=16)

# # 레이아웃 조정 및 그래프 출력
# plt.tight_layout(rect=[0, 0, 1, 0.95])  # 전체 제목을 위한 여백 추가
# plt.show()

######################################################################### 수송량 산출 #####################################################################
# 최댓값과 평균값 비교

# print(station_2_y.max())
# print(station_7_y.max())
# print(station_2_y.mean())
# print(station_7_y.mean())

# 2호선 일평균 수송량 중 1위부터 3위까지의 일평균 수송량 구하기
# top_3_2 = trans_p_2line_dropped.nlargest(3, '일평균')
# print("2호선 일평균 수송량 1위부터 3위까지:")
# print(top_3_2[['지하철역', '일평균']])

######################################################################## 혼잡도 ########################################################################
## pre-processing
# congestion_2023 = congestion_2023.drop(['00시30분'],axis=1)
# congestion_2023 = congestion_2023.drop(['연번'],axis=1)
# congestion_2023 = congestion_2023.drop(['요일구분'],axis=1)
# congestion_2023 = congestion_2023.drop(['역번호'],axis=1)
# hongdae_data = congestion_2023[congestion_2023['출발역']=='홍대입구']
# guro_data = congestion_2023[congestion_2023['출발역']=='구로디지털단지']

# '홍대입구역' 데이터에서 각 열의 가장 높은 값으로 구성된 데이터프레임 생성
# hongdae_max = hongdae_data.max(axis=0).to_frame().T
# hongdae_max['지하철역'] = '홍대입구'
# hongdae_max = hongdae_max.drop(['지하철역'],axis=1)
# print(hongdae_max)

# '구로디지털단지' 데이터에서 각 열의 가장 높은 값으로 구성된 데이터프레임 생성
# guro_max = guro_data.max(axis=0).to_frame().T
# guro_max['지하철역'] = '구로디지털단지'
# guro_max = guro_max.drop(['지하철역'],axis=1)
# print(guro_max)

## Figure 생성
# fig1, (ax4, ax3) = plt.subplots(2, 1, figsize=(16, 8))

# # 시간대별 데이터를 추출하여 선 그래프 그리기
# time_columns_guro = guro_max.columns[3:]  # 시간대 열 자동 추출 (3번째 열부터 끝까지)

# # 첫 번째 서브플롯 (7호선)
# ax3.plot(time_columns_guro, guro_max.iloc[0][time_columns_guro], marker='o', linestyle='-', color='olive')  # 라인그래프
# ax3.set_title('구로디지털단지역 시간대별 혼잡도', fontweight='bold')  # 서브플롯 제목
# ax3.set_xlabel('시간')  # x축 제목
# ax3.set_ylabel('혼잡도')  # y축 제목
# ax3.grid(True)

# # x축의 눈금과 레이블 설정
# ax3.set_xticks(range(0, len(time_columns_guro), 2))  # 간격을 두어 x축 눈금 설정
# ax3.set_xticklabels(time_columns_guro[::2], fontsize=8)  # 레이블 설정

# ## 홍대입구역
# # 시간대별 데이터를 추출하여 선 그래프 그리기
# time_columns_hongdae = hongdae_max.columns[3:]  # 시간대 열 자동 추출 (3번째 열부터 끝까지)

# # 두 번째 서브플롯 (2호선)
# ax4.plot(time_columns_hongdae, hongdae_max.iloc[0][time_columns_hongdae], marker='o', linestyle='-', color='limegreen')  # 라인그래프
# ax4.set_title('홍대입구역 시간대별 혼잡도', fontweight='bold')  # 서브플롯 제목
# ax4.set_xlabel('시간')  # x축 제목
# ax4.set_ylabel('혼잡도')  # y축 제목
# ax4.grid(True)

# # x축의 눈금과 레이블 설정
# ax4.set_xticks(range(0, len(time_columns_hongdae), 2))  # 간격을 두어 x축 눈금 설정
# ax4.set_xticklabels(time_columns_hongdae[::2], fontsize=8)  # 레이블 설정

# plt.tight_layout()
# plt.show()

######################################################################## 혼잡도 산출 #######################################################################
# # 최댓값 산출
# print(hongdae_max.iloc[0][time_columns_hongdae].max())
# print(guro_max.iloc[0][time_columns_guro].max())

###############################################################3#### 어깨가쪽사이길이 평균값 ####################################################################
## pre-processing
## 열 이름에서 숫자, 점, 공백 제거
# body_2020.columns = body_2020.columns.str.replace(r'\d+\.\s*', '', regex=True)
# # print(body_2020.head())
# body_2020 = body_2020.drop(['조사년도'],axis=1)
# body_2020 = body_2020.drop(['조사일'],axis=1)
# body_2020 = body_2020.drop(['생년월일'],axis=1)
# body_2020 = body_2020.drop(['현주거지(시/도)'],axis=1)
# print(body_2020.head())

# 산점도 그래프 그리기 (성별 구분)
# plt.figure(figsize=(10, 6))
# sns.scatterplot(x='나이', y='팔꿈치주먹수평길이(팔굽힌) ', hue='성별',palette={'M': 'cornflowerblue', 'F': 'tomato'}, data=body_2020.sample(frac=0.2, random_state=42), alpha=0.3)
# sns.lineplot(x='나이', y='팔꿈치주먹수평길이(팔굽힌) ', hue='성별',palette={'M': 'mediumblue', 'F': 'red'}, data=body_2020,alpha=0.7, estimator='mean', ci=None, lw=1.5, legend=False)
# plt.title('나이에 따른 팔꿈치주먹수평길이(팔굽힌)의 변화',fontweight = 'bold',fontsize=24)
# plt.xticks(fontsize=18)
# plt.yticks(fontsize=18)
# plt.xlabel('나이',fontsize=24)
# plt.ylabel('팔꿈치주먹수평길이(팔굽힌) ',fontsize=24)
# plt.legend(title='성별',labels=['남성', '여성'],handles=[
#     plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='cornflowerblue', markersize=10, label='Male'),
#     plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='tomato', markersize=10, label='Female')
# ])  # 레전드 핸들링
# plt.show()

# 바이올린 플롯 그리기 (성별에 따른 색상 지정 및 평균값 라인 추가, 스웜플롯 추가)
# plt.figure(figsize=(12, 8))
# sns.violinplot(x='성별', y='팔꿈치주먹수평길이(팔굽힌) ', data=body_2020.sample(frac=0.1, random_state=42), palette={'M': 'cornflowerblue', 'F': 'tomato'}, inner=None, alpha=0.7)
# sns.pointplot(x='성별', y='팔꿈치주먹수평길이(팔굽힌) ', data=body_2020, join=True, estimator='mean', color='red', markers='o', scale=0.5)
# sns.swarmplot(x='성별', y='팔꿈치주먹수평길이(팔굽힌) ', data=body_2020.sample(frac=0.05, random_state=42),palette={'M': 'mediumblue', 'F': 'red'}, color='k', alpha=0.6, size=3)
# plt.title('나이에 따른 팔꿈치주먹수평길이(팔굽힌)의 변화',fontweight = 'bold',fontsize=24)
# plt.xticks(ticks=[0, 1], labels=['남성', '여성'],fontsize=18)
# plt.yticks(fontsize=18)
# plt.xlabel('성별',fontsize=24)
# plt.ylabel('팔꿈치주먹수평길이(팔굽힌) ',fontsize=24)
# plt.legend(title='성별',labels=['남성', '여성'],handles=[
#     plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='cornflowerblue', markersize=10, label='Male'),
#     plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='tomato', markersize=10, label='Female')
# ])  # 레전드 핸들링
# plt.show()

#########################################################################   평균값 산출   ###########################################################################

# 성별에 따른 '어깨가쪽사이길이'의 평균값 계산
# mean_value_shoulder = body_2020.groupby('성별')['어깨가쪽사이길이 '].mean()
# print(mean_value_shoulder)

# # 성별에 따른 모든 열의 평균값 계산
# mean_values_all = body_2020.groupby('성별').mean()
# print(mean_values_all)

# # 남/여 평균값의 차이
# mean_value_rock = body_2020.groupby('성별')['팔꿈치주먹수평길이(팔굽힌) '].mean()

# print(f"남성의 평균 팔꿈치주먹수평길이(팔굽힌): {mean_value_rock}")

################################################################## 부 록(5호선의 혼잡도) ##################################################################################################################

# trans_p_5line = trans_p_5line.sort_values('일평균',ascending=False)
# columns_to_drop = [col for col in trans_p_5line.columns if '월' in col]
# trans_p_5line_dropped = trans_p_5line.drop(columns = columns_to_drop)
# trans_p_5line_dropped = trans_p_5line_dropped.drop(['계'],axis=1)
# print(trans_p_5line_dropped.head())

# shuffled_indices_2 = np.random.permutation(trans_p_5line_dropped.index)
# trans_p_5line_dropped = trans_p_5line_dropped.loc[shuffled_indices_2].reset_index(drop=True)

# ## visualization
# plt.rcParams['font.family'] ='Malgun Gothic'
# plt.rcParams['axes.unicode_minus'] =False

# # 서브플롯 (5호선)
# station_5_x = trans_p_5line_dropped['지하철역']
# station_5_y = trans_p_5line_dropped['일평균']
# max_value_5 = station_5_y.max() # 최댓값 구하기
# max_index_5 = station_5_y.idxmax()

# # 그래프 생성
# plt.figure(figsize=(12, 6))
# plt.bar(station_5_x, station_5_y, color='forestgreen', alpha=0.7)# 바그래프
# plt.bar(station_5_x[max_index_5], max_value_5, color='aquamarine') # 최댓값 표시
# plt.plot(station_5_x, station_5_y, marker='o', linestyle='-', color='limegreen')# 라인그래프
# plt.title('5호선 일평균 수송량', fontweight='bold',fontsize=18)  # 서브플롯 제목
# plt.xlabel('지하철역',fontsize=18)  # x축 제목
# plt.ylabel('일평균 수송량',fontsize=18)  # y축 제목
# plt.xticks(station_5_x, rotation=90, ha='right')  # x축 눈금 라벨 회전

# # 범례 추가
# plt.legend()

# # 그래프 표시
# plt.tight_layout()  # 레이아웃 조정
# plt.show()

# gwangwha_data = congestion_2023[congestion_2023['출발역']=='광화문']

# # '광화문' 데이터에서 숫자형 열만 선택하고 최댓값으로 구성된 데이터프레임 생성
# gwangwha_max = gwangwha_data.select_dtypes(include='number').max(axis=0).to_frame().T

# # 시간대 열 자동 추출
# time_columns_gwangwha = [col for col in gwangwha_max.columns if '시' in col or '시간' in col]

# # 그래프 생성
# plt.figure(figsize=(12, 6))
# plt.plot(
#     time_columns_gwangwha, 
#     gwangwha_max.iloc[0][time_columns_gwangwha], 
#     marker='o', 
#     linestyle='-', 
#     color='olive', 
#     label='혼잡도 추이'
# )
# plt.title('광화문역 시간대별 혼잡도', fontweight='bold')
# plt.xlabel('시간')
# plt.ylabel('혼잡도')
# plt.grid(True)

# # x축의 눈금과 레이블 설정
# plt.xticks(range(0, len(time_columns_gwangwha), 2), time_columns_gwangwha[::2], fontsize=8)

# # 범례 추가
# plt.legend()

# # 레이아웃 조정 및 그래프 표시
# plt.tight_layout()
# plt.show()
