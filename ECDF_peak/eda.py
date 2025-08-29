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
from scipy.stats import gaussian_kde, kurtosis, skew
from matplotlib.lines import Line2D

from configs.config import get_args
from dataloaders.data_loader import PAMAP2, get_data
from statsmodels.distributions.empirical_distribution import ECDF

def run_eda_grouped_kde_plot():
    """
    하나의 그림 파일 안에 36개의 센서 그룹별 KDE 플롯을 생성합니다. (클래스별 윈도우 1개씩)
    - 행: 3개의 신체 부위 (Hand, Chest, Ankle)
    - 열: 12개의 랜덤 윈도우 (각각 다른 활동 클래스)
    - 각 서브플롯에는 x, y, z축 데이터가 함께 그려집니다.
    - 파일은 스크립트 실행 위치에 저장됩니다.
    """
    print("--- EDA (Grouped KDE Plot for 12 Activities) 시작 ---")

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

    # 2. 특정 Subject가 지정된 경우, 해당 데이터만 필터링
    if args.specific_subject is not None and args.specific_subject in dataset.all_keys:
        print(f"--- 특정 Subject '{args.specific_subject}'의 훈련 데이터만 사용 ---")
        
        # 해당 subject의 윈도우와 활동만 필터링
        train_windows = [w for w in dataset.train_slidingwindows if w[0] == args.specific_subject]
        train_activities = [act for w, act in zip(dataset.train_slidingwindows, dataset.activity_per_windows) if w[0] == args.specific_subject]
        
        # 만약 해당 subject의 데이터가 없다면 경고 후 종료
        if not train_windows:
            print(f"오류: Subject '{args.specific_subject}'에 대한 훈련 데이터가 없습니다.")
            return
    else:
        print("--- 모든 Subject의 훈련 데이터 사용 ---")
        train_windows = dataset.train_slidingwindows
        train_activities = dataset.activity_per_windows

    # 3. 활동 클래스별로 윈도우 1개씩 샘플링
    print("--- 활동 클래스별 윈도우 샘플링 시작 ---")
    
    activity_id_to_name = {id: name for id, name in dataset.label_map}

    # 필터링된 훈련 데이터에서 윈도우가 할당된 활동만 추출
    activity_to_windows = {activity_id: [] for activity_id in activity_id_to_name.keys()}
    for window, activity_id in zip(train_windows, train_activities):
        if activity_id in activity_to_windows:
            activity_to_windows[activity_id].append(window)

    # 윈도우가 하나 이상 있는 활동 클래스만 필터링
    activities_with_windows = [act_id for act_id, win_list in activity_to_windows.items() if win_list]
    
    # 분석할 데이터가 없으면 종료
    if not activities_with_windows:
        print("오류: 분석할 훈련 데이터가 없습니다.")
        return

    # 훈련 데이터에 존재하는 모든 유니크한 활동을 사용 (정렬하여 일관성 유지)
    selected_activity_ids = sorted(activities_with_windows)
    num_activities_to_sample = len(selected_activity_ids)

    selected_windows = []
    for act_id in selected_activity_ids:
        window = random.choice(activity_to_windows[act_id])
        selected_windows.append(window)

    print(f"선택된 활동 클래스: {[activity_id_to_name.get(i, 'Unknown') for i in selected_activity_ids]}")
    print(f"분석에 사용할 윈도우 (start, end): {[(w[1], w[2]) for w in selected_windows]}\n")

    # 4. 단일 Figure와 3x12 서브플롯 그리드 생성
    fig, axes = plt.subplots(3, num_activities_to_sample, figsize=(4 * num_activities_to_sample, 15), sharex=False, sharey=False)

    # 제목에 Subject 정보 추가
    title_text = f'KDE Plots for Sensor Groups across {num_activities_to_sample} Different Activities'
    if args.specific_subject is not None and args.specific_subject in dataset.all_keys:
        title_text += f' (Subject {args.specific_subject})'
    fig.suptitle(title_text, fontsize=16)

    # 5. 각 신체 부위(행)와 랜덤 윈도우(열)에 대해 반복
    for row_idx, (body_part, sensor_list) in enumerate(sensor_groups.items()):
        print(f"--- '{body_part}' 센서 그룹 처리 중 ---")
        for col_idx, window in enumerate(selected_windows):
            # axes가 1차원 배열일 경우를 대비하여 인덱싱 수정
            ax = axes[row_idx, col_idx] if num_activities_to_sample > 1 else axes[row_idx]
            
            sub_id, start, end = window
            
            dominant_activity_id = selected_activity_ids[col_idx]
            activity_name = activity_id_to_name.get(dominant_activity_id, "Unknown")
            title_str = f"Activity: {activity_name}\n(Window: {start}-{end})"
            ax.set_title(title_str)

            if col_idx == 0:
                ax.set_ylabel(f'{body_part}\n\nDensity', fontweight='bold')

            for sensor_name in sensor_list:
                try:
                    window_data = dataset.data_x.iloc[start:end][sensor_name].values
                except KeyError:
                    print(f"경고: '{sensor_name}' 컬럼을 찾을 수 없습니다. 건너뜁니다.")
                    continue

                if len(window_data) < 2:
                    continue
                
                axis = sensor_name.split('_')[1]
                color = axis_colors.get(axis, 'black')
                
                data_mean = np.mean(window_data)
                peak_x = data_mean
                try:
                    kde = gaussian_kde(window_data)
                    x_range = np.linspace(window_data.min(), window_data.max(), 500)
                    kde_values = kde(x_range)
                    peak_x = x_range[np.argmax(kde_values)]
                except (np.linalg.LinAlgError, ValueError):
                    print(f"경고: {sensor_name} 데이터의 KDE peak 계산 중 오류 발생. Peak을 Mean으로 대체합니다.")
                
                distance = abs(peak_x - data_mean)
                label = f'Axis {axis.upper()} | P-M: {distance:.3f}'

                sns.kdeplot(x=window_data, fill=True, alpha=0.3, color=color, label=label, ax=ax)
                
                ax.axvline(data_mean, color=color, linestyle='--', linewidth=1.5)
                ax.axvline(peak_x, color=color, linestyle=':', linewidth=1.5)

            handles, labels = ax.get_legend_handles_labels()
            if handles:
                handles.append(Line2D([0], [0], color='black', linestyle='--', linewidth=1.5))
                labels.append('Mean')
                handles.append(Line2D([0], [0], color='black', linestyle=':', linewidth=1.5))
                labels.append('Peak (Mode)')
                ax.legend(handles=handles, labels=labels, fontsize='small')

            ax.set_xlabel('Sensor Value')

    # 6. 전체 레이아웃 정리 및 파일 저장
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    script_dir = os.path.dirname(os.path.abspath(__file__))
    korea_time = datetime.now(ZoneInfo("Asia/Seoul"))
    timestamp = korea_time.strftime("%Y%m%d_%H%M%S")
    output_filename = f"eda_grouped_kde_activities_{timestamp}.png"
    output_path = os.path.join(script_dir, output_filename)
    
    plt.savefig(output_path)
    plt.close(fig) 
    print(f"\n'{output_filename}' 파일에 {3 * num_activities_to_sample}개 KDE 플롯이 모두 저장되었습니다.")
    print(f"저장 위치: {output_path}")
    print("\n--- 모든 EDA 종료 ---")

def run_eda_statistical_analysis(statistic='kurtosis'):
    """
    모든 subject와 activity에 대해 주어진 통계치(kurtosis, std, skew)를 계산하고 CSV와 그래프로 저장합니다.
    """
    print(f"\n{'='*60}")
    print(f"--- EDA ({statistic.capitalize()} Analysis for All Subjects and Activities) 시작 ---")
    print(f"{'='*60}")

    # 1. 설정 및 데이터 로더 준비
    args = get_args()
    args.datanorm_type = None
    dataset = PAMAP2(args)
    
    seq_id_to_name = {i: name for i, (orig_id, name) in enumerate(dataset.label_map)}
    
    sensors_to_analyze = [
        'acc_x_hand', 'acc_y_hand', 'acc_z_hand',
        'acc_x_chest', 'acc_y_chest', 'acc_z_chest',
        'acc_x_ankle', 'acc_y_ankle', 'acc_z_ankle'
    ]
    
    results = []

    # 2. 모든 Subject에 대해 반복
    print(f"분석 대상 Subjects: {sorted(dataset.all_keys)}")
    for subject_id in sorted(dataset.all_keys):
        print(f"--- Subject {subject_id} 처리 중 ---")

        train_windows = [w for w in dataset.train_slidingwindows if w[0] == subject_id]
        train_activities = [act for w, act in zip(dataset.train_slidingwindows, dataset.activity_per_windows) if w[0] == subject_id]

        if not train_windows:
            print(f"  - 데이터 없음, 건너뜁니다.")
            continue

        activity_to_windows_subject = {act_id: [] for act_id in dataset.no_drop_activites}
        for window, activity_id in zip(train_windows, train_activities):
            if activity_id in activity_to_windows_subject:
                activity_to_windows_subject[activity_id].append(window)

        for activity_id, windows in activity_to_windows_subject.items():
            if not windows:
                continue
            
            activity_name = seq_id_to_name.get(activity_id, "Unknown")

            for sensor_name in sensors_to_analyze:
                all_sensor_data = []
                for window in windows:
                    sub_id, start, end = window
                    try:
                        window_data = dataset.data_x.iloc[start:end][sensor_name].values
                        all_sensor_data.extend(window_data)
                    except KeyError:
                        continue
                
                if len(all_sensor_data) > 20: # 충분한 데이터가 있을 때만 계산
                    value = 0
                    if statistic == 'kurtosis':
                        value = kurtosis(all_sensor_data, fisher=True)
                    elif statistic == 'std':
                        value = np.std(all_sensor_data)
                    elif statistic == 'skew':
                        value = skew(all_sensor_data)
                    else:
                        continue
                    
                    results.append({
                        'subject_id': subject_id,
                        'activity_name': activity_name,
                        'sensor': sensor_name,
                        statistic: value
                    })

    if not results:
        print(f"오류: {statistic.capitalize()}를 계산할 데이터가 없습니다.")
        return

    results_df = pd.DataFrame(results)
    timestamp = datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y%m%d_%H%M%S")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 5. Raw 데이터 결과 출력 및 저장
    print(f"\n--- {statistic.capitalize()} 분석 Raw 데이터 ---")
    print(results_df.to_string())
    output_filename = f"eda_{statistic}_raw_data_{timestamp}.csv"
    output_path = os.path.join(script_dir, output_filename)
    results_df.to_csv(output_path, index=False)
    print(f"\n'{output_filename}' 파일에 Raw 데이터가 저장되었습니다.")
    print(f"저장 위치: {output_path}")

    # 6. 활동 클래스별 집계 및 시각화
    print(f"\n--- 활동 클래스별 {statistic.capitalize()} 평균 ---")
    
    results_df['sensor'] = pd.Categorical(results_df['sensor'], categories=sensors_to_analyze, ordered=True)
    agg_df = results_df.groupby(['activity_name', 'sensor'], observed=False)[statistic].mean().unstack()
    
    print(agg_df.to_string())

    # --- 시각화 ---
    acc_sensors = [s for s in sensors_to_analyze if 'acc' in s]

    if acc_sensors:
        plot_df = agg_df[[s for s in acc_sensors if s in agg_df.columns]]
        if not plot_df.empty:
            ax = plot_df.plot(kind='bar', figsize=(18, 8), width=0.8, 
                              title=f'Average {statistic.capitalize()} per Activity (ACC Sensors)',
                              grid=True)
            ax.set_xlabel("Activity", fontsize=12)
            ax.set_ylabel(f"Average {statistic.capitalize()}", fontsize=12)
            ax.tick_params(axis='x', labelrotation=45)
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()

            plot_filename = f"eda_{statistic}_barchart_acc_{timestamp}.png"
            plot_output_path = os.path.join(script_dir, plot_filename)
            plt.savefig(plot_output_path)
            plt.close()
            print(f"\n'{plot_filename}' 파일에 ACC 센서 그래프가 저장되었습니다.")
            print(f"저장 위치: {plot_output_path}")

    print(f"\n--- {statistic.capitalize()} 분석 종료 ---")

def run_eda_feature_scaling_analysis():
    """
    모델이 예측할 타겟 통계치들의 분포(scale)를 분석합니다.
    """
    print(f"\n{'='*60}")
    print(f"--- EDA (Target Feature Scaling Analysis) 시작 ---")
    print(f"{ '='*60}")

    # 1. 설정 및 데이터 로더 준비
    args = get_args()
    args.datanorm_type = None
    dataset = PAMAP2(args)
    
    sensors_to_analyze = [
        'acc_x_hand', 'acc_y_hand', 'acc_z_hand',
        'acc_x_chest', 'acc_y_chest', 'acc_z_chest',
        'acc_x_ankle', 'acc_y_ankle', 'acc_z_ankle'
    ]
    
    stats_to_calculate = ['mean', 'std', 'skew', 'kurtosis']
    feature_values = {stat: [] for stat in stats_to_calculate}

    print("전체 훈련 데이터의 모든 윈도우에 대해 통계치 계산 중...")
    # 2. 모든 윈도우에 대해 반복하며 통계치 계산
    for window in dataset.train_slidingwindows:
        sub_id, start, end = window
        for sensor_name in sensors_to_analyze:
            window_data = dataset.data_x.iloc[start:end][sensor_name].values
            if len(window_data) < 20:
                continue
            
            feature_values['mean'].append(np.mean(window_data))
            feature_values['std'].append(np.std(window_data))
            feature_values['skew'].append(skew(window_data))
            feature_values['kurtosis'].append(kurtosis(window_data, fisher=True))

    # 3. 분포 요약
    summary = []
    for stat_name, values in feature_values.items():
        if not values:
            continue
        series = pd.Series(values)
        summary.append({
            'statistic': stat_name,
            'mean': series.mean(),
            'std': series.std(),
            'min': series.min(),
            '25%': series.quantile(0.25),
            '50% (median)': series.median(),
            '75%': series.quantile(0.75),
            'max': series.max()
        })
    
    if not summary:
        print("오류: 분석할 통계 데이터가 없습니다.")
        return

    summary_df = pd.DataFrame(summary).set_index('statistic')
    
    print("\n--- 타겟 통계치 분포 분석 결과 ---")
    print("ECDF 값은 0과 1 사이의 값을 가지므로 비교에서 제외했습니다.")
    print(summary_df.to_string())
    print(f"\n--- 분석 종료 ---")


if __name__ == '__main__':
    # run_eda_grouped_kde_plot()
    # run_eda_statistical_analysis(statistic='kurtosis')
    # run_eda_statistical_analysis(statistic='std')
    # run_eda_statistical_analysis(statistic='skew')
    run_eda_feature_scaling_analysis()
