import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
from datetime import datetime
from zoneinfo import ZoneInfo

from configs.config import get_args
from dataloaders.data_loader import PAMAP2, get_data
from statsmodels.distributions.empirical_distribution import ECDF

def run_eda_grouped_kde_plot():
    """
    하나의 그림 파일 안에 9개의 센서 그룹별 KDE 플롯을 생성합니다.
    - 행: 3개의 신체 부위 (Hand, Chest, Ankle)
    - 열: 3개의 랜덤 윈도우
    - 각 서브플롯에는 x, y, z축 데이터가 함께 그려집니다.
    - 파일은 스크립트 실행 위치에 저장됩니다.
    """
    print("--- EDA (Grouped KDE Plot) 시작 ---")

    # 1. 설정 및 데이터 로더 준비
    args = get_args()
    args.datanorm_type = None
    dataset = PAMAP2(args)
    
    # 2. 분석할 센서 그룹 정의
    sensor_groups = {
        'Hand': ['acc_x_hand', 'acc_y_hand', 'acc_z_hand'],
        'Chest': ['acc_x_chest', 'acc_y_chest', 'acc_z_chest'],
        'Ankle': ['acc_x_ankle', 'acc_y_ankle', 'acc_z_ankle']
    }
    axis_colors = {'x': 'royalblue', 'y': 'darkorange', 'z': 'forestgreen'}

    # 3. 모든 그룹에 동일하게 적용할 3개의 윈도우를 랜덤으로 선택
    all_windows = dataset.train_slidingwindows + dataset.test_slidingwindows
    if len(all_windows) < 3:
        print("오류: 윈도우 개수가 3개 미만이므로 실행할 수 없습니다.")
        return
    random_indices = random.sample(range(len(all_windows)), 3)
    print(f"분석에 사용할 랜덤 윈도우 인덱스: {random_indices}\n")

    # 4. 단일 Figure와 3x3 서브플롯 그리드 생성
    fig, axes = plt.subplots(3, 3, figsize=(18, 15), sharex=False, sharey=False)
    fig.suptitle('KDE Plots for Sensor Groups across 3 Random Windows', fontsize=16)

    # 5. 각 신체 부위(행)와 랜덤 윈도우(열)에 대해 반복
    for row_idx, (body_part, sensor_list) in enumerate(sensor_groups.items()):
        print(f"--- '{body_part}' 센서 그룹 처리 중 ---")
        for col_idx, window_idx in enumerate(random_indices):
            ax = axes[row_idx, col_idx]
            window = all_windows[window_idx]
            _, start, end = window
            
            #activity_id =dataset.data_y.iloc[start:end].mode()[0] 가장 많은걸로 activity_id 선택

            #현재 window의 활동 비율 계산
            activity_series = dataset.data_y.iloc[start:end].value_counts(normalize=True)
            title_str= f"Window #{window_idx}\n"

            for i, (activity_id, ratio) in enumerate(activity_series.head(3).items()):
                activity_name = "Unknown"
                for act_id_map, act_name_map in dataset.label_map:
        
                    if act_id_map == activity_id:
                        activity_name = act_name_map
                        break
                title_str += f'[{i+1}] {activity_name} ({ratio:.0%})'
            #sub plot 제목
            ax.set_title(title_str.strip())

            if row_idx == 0:
                pass
            
            if col_idx == 0:
                ax.set_ylabel(f'{body_part}\n\nDensity', fontweight='bold')

            for sensor_name in sensor_list:
                try:
                    window_data = dataset.data_x.iloc[start:end][sensor_name].values
                except KeyError:
                    print(f"경고: '{sensor_name}' 컬럼을 찾을 수 없습니다. 건너뜁니다.")
                    continue

                if len(window_data) == 0:
                    continue
                
                axis = sensor_name.split('_')[1]
                color = axis_colors.get(axis, 'black')
                label = f'Axis {axis.upper()}'

                sns.kdeplot(x=window_data, fill=True, alpha=0.3, color=color, label=label, ax=ax)
                
                data_mean = np.mean(window_data)
                ax.axvline(data_mean, color=color, linestyle='--', linewidth=1.5)
        
            ax.legend()
            ax.set_xlabel('Sensor Value')

    # 6. 전체 레이아웃 정리 및 파일 저장
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    script_dir = os.path.dirname(os.path.abspath(__file__))
    korea_time = datetime.now(ZoneInfo("Asia/Seoul"))
    timestamp = korea_time.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"eda_grouped_kde_{timestamp}.png"
    output_path = os.path.join(script_dir, output_filename)
    
    plt.savefig(output_path)
    plt.close(fig) 
    print(f"\n'{output_filename}' 파일에 9개 KDE 플롯이 모두 저장되었습니다.")
    print(f"저장 위치: {output_path}")
    print("\n--- 모든 EDA 종료 ---")

def run_eda_activity_comparison_plot():
    """
    신체 부위와 센서 축을 기준으로, 여러 활동의 데이터 분포를 비교합니다.
    - 행: 3개의 신체 부위 (Hand, Chest, Ankle)
    - 열: 3개의 센서 축 (X, Y, Z)
    - 각 서브플롯에는 'lying', 'walking', 'running' 활동의 분포가 함께 그려집니다.
    """
    print("--- EDA (Activity Comparison KDE Plot) 시작 ---")

    # 1. 설정 및 데이터 로더 준비
    args = get_args()
    args.datanorm_type = None # 정규화 비활성화
    dataset = PAMAP2(args)
    
    # 2. 분석 대상 정의
    body_parts = ['Hand', 'Chest', 'Ankle']
    axes = ['x', 'y', 'z']
    activities_to_compare = ['lying', 'walking', 'running']
    activity_colors = {'lying': '#1f77b4', 'walking': '#ff7f0e', 'running': '#2ca02c'}

    # 필요한 데이터만 담을 데이터프레임 생성
    label_to_id = {v: k for k, v in dataset.label_map}
    id_to_transformed_id = dataset.labelToId
    
    # 모든 데이터를 담을 리스트
    all_plot_data = [] 

    # 모든 윈도우에서 데이터 추출
    for window in (dataset.train_slidingwindows + dataset.test_slidingwindows):
        sub_id, start, end = window
        activity_id = dataset.data_y.iloc[start:end].mode()[0]
        
        # 윈도우의 데이터를 가져옴
        window_df = dataset.data_x.iloc[start:end]
        
        # 각 센서 데이터에 활동 ID 추가
        temp_df = window_df.melt(var_name='sensor', value_name='value')
        temp_df['activity_id'] = activity_id
        all_plot_data.append(temp_df)

    if not all_plot_data:
        print("오류: 분석할 데이터를 찾을 수 없습니다.")
        return

    full_df = pd.concat(all_plot_data, ignore_index=True)
    full_df['activity_name'] = full_df['activity_id'].map({k: v for k, v in id_to_transformed_id.items()}).map({k: v for k, v in dataset.label_map})

    # 3. 3x3 서브플롯 그리드 생성
    fig, axes_grid = plt.subplots(3, 3, figsize=(18, 15), sharex=True, sharey=True)
    fig.suptitle('KDE Plots of Sensor Data Across Different Activities', fontsize=16)

    # 4. 각 서브플롯에 데이터 그리기
    for row, part in enumerate(body_parts):
        for col, axis in enumerate(axes):
            ax = axes_grid[row, col]
            sensor_name_part = f'acc_{axis}_{part.lower()}'
            
            # 현재 서브플롯에 해당하는 데이터 필터링
            plot_data = full_df[(full_df['sensor'] == sensor_name_part) & (full_df['activity_name'].isin(activities_to_compare))]
            
            if plot_data.empty:
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center')
                continue

            # 활동별로 KDE 플롯 그리기
            sns.kdeplot(data=plot_data, x='value', hue='activity_name', 
                        hue_order=activities_to_compare, palette=activity_colors,
                        fill=True, alpha=0.4, ax=ax, common_norm=False)
            
            ax.legend_.set_title('Activity')
            ax.set_title(f'{part} - Axis {axis.upper()}')
            ax.set_xlabel('Sensor Value')
            ax.set_ylabel('Density' if col == 0 else '')

    # 5. 전체 레이아웃 정리 및 파일 저장
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    script_dir = os.path.dirname(os.path.abspath(__file__))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"eda_activity_comparison_{timestamp}.png"
    output_path = os.path.join(script_dir, output_filename)
    
    plt.savefig(output_path)
    plt.close(fig)
    print(f"\n'{output_filename}' 파일에 활동 비교 플롯이 모두 저장되었습니다.")
    print(f"저장 위치: {output_path}")
    print("\n--- 모든 EDA 종료 ---")

if __name__ == '__main__':
    # run_eda_grouped_kde_plot() 
    run_eda_grouped_kde_plot()
    #run_eda_activity_comparison_plot()