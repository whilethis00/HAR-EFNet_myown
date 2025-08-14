import os
import pickle
import numpy as np
import pandas as pd
import random
from collections import Counter
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

# 내부 유틸리티 모듈 임포트
from .data_utils import Normalizer,components_selection_one_signal

# 데이터 로더를 생성하는 메인 함수
def get_data(dataset, batch_size, flag="train", collossl_mode=False):
    """
    주어진 데이터셋 객체(PAMAP2)를 사용하여 PyTorch의 DataLoader를 생성합니다.

    Args:
        dataset (PAMAP2): 데이터셋의 모든 정보와 전처리 로직을 담고 있는 객체입니다.
        batch_size (int): DataLoader가 한 번에 반환할 데이터 샘플의 수입니다.
        flag (str): 데이터셋의 종류를 지정합니다 ('train', 'valid', 'test').
        collossl_mode (bool): 특정 훈련 모드(ColloSSL) 사용 여부입니다. 여기서는 shuffle 여부에만 영향을 줍니다.

    Returns:
        DataLoader: PyTorch 모델에 데이터를 공급할 수 있는 데이터 로더 객체입니다.
    """
    # 훈련(train) 모드일 경우, 데이터를 섞어주는 것이 일반적이므로 shuffle_flag를 True로 설정합니다.
    # 모델이 데이터 순서에 과적합되는 것을 방지합니다.
    if flag == 'train' and not collossl_mode:
        shuffle_flag = True
    # 검증(valid)이나 테스트(test) 모드에서는 평가의 일관성을 위해 데이터를 섞지 않습니다.
    else:
        shuffle_flag = False
    
    # 주어진 플래그('train', 'valid', 'test')에 따라 PyTorch의 Dataset 객체(아래 정의된 Dataset 클래스)를 생성합니다.
    # 이 객체는 DataLoader에 의해 사용되어 실제 데이터를 배치 단위로 공급하는 역할을 합니다.
    # 이 시점의 `dataset`은 PAMAP2 객체이며, `Dataset` 클래스는 이 객체로부터 필요한 데이터를 받아옵니다.
    # 예: shape (168, 9)
    dataset = Dataset(dataset, flag)

    # PyTorch의 DataLoader를 생성합니다.
    # DataLoader는 Dataset으로부터 데이터를 받아와 지정된 배치 크기(batch_size)로 묶어주고,
    # 훈련 시 데이터를 섞는 등의 기능을 제공합니다.
    # 예: shape (128, 168, 9)
    data_loader = DataLoader(dataset,
                             batch_size=batch_size,     # 배치 크기
                             shuffle=shuffle_flag,      # 데이터 섞음 여부
                             num_workers=0,             # 데이터 로딩에 사용할 서브프로세스 수 (0은 메인 프로세스 사용)
                             drop_last=False)           # 마지막 배치가 배치 크기보다 작을 경우 버릴지 여부
    return data_loader


class PAMAP2(object):
    """
    PAMAP2 데이터셋의 로딩, 전처리, 분할 등 모든 과정을 관리하는 클래스입니다.
    """
    def __init__(self, args):
        # --- 1. 기본 설정 및 파라미터 초기화 ---
        self.args = args                               # main.py에서 전달받은 인자(arguments)
        self.data_name = args.data_name                # 데이터셋 이름 ('pamap2')
        self.data_path = args.data_path                # 원본 데이터 파일 경로
        self.windowsize = args.window_size             # 슬라이딩 윈도우 크기 (예: 168)
        self.freq1 = args.freq1                        # 필터링 주파수 1
        self.freq2 = args.freq2                        # 필터링 주파수 2
        self.sampling_freq = args.sampling_freq        # 샘플링 주파수
        self.pkl_save_path = args.pkl_save_path        # 전처리된 데이터를 저장할 경로
        self.filtering = args.filtering                # 노이즈/중력 필터링 적용 여부

        self.datanorm_type = args.datanorm_type        # 데이터 정규화 방법 ('znorm' 등)
        self.train_vali_quote = args.train_vali_quote  # 훈련 데이터 중 검증 데이터로 사용할 비율
        
        # LOCV(Leave-One-Subject-Out Cross-Validation)에 사용할 피실험자 목록입니다.
        # 각 리스트는 테스트에 사용될 피실험자 ID를 의미합니다. (피실험자 9는 데이터 부족 등의 이유로 제외됨)
        self.LOCV_keys = [[1], [2], [3], [4], [5], [6], [7], [8]] 
        self.all_keys = [1, 2, 3, 4, 5, 6, 7, 8]       # 전체 피실험자 ID 목록

        self.index_of_cv = 0                           # 현재 진행 중인 교차 검증 폴드의 인덱스

        # 원본 데이터 파일에서 사용할 열(column)의 인덱스 목록
        self.used_cols = [1,
                            # 3개의 IMU 센서(손, 가슴, 발목)에서 가속도계와 자이로스코프 데이터를 사용
                            # 각 센서는 3축(x, y, z) 데이터를 가짐
                            # 4, 5, 6: 손 가속도계(x,y,z), 10, 11, 12: 손 자이로스코프(x,y,z)
                            4, 5, 6, 10, 11, 12,      # IMU Hand
                            # 21, 22, 23: 가슴 가속도계, 27, 28, 29: 가슴 자이로스코프
                            21, 22, 23, 27, 28, 29,   # IMU Chest
                            # 38, 39, 40: 발목 가속도계, 44, 45, 46: 발목 자이로스코프
                            38, 39, 40, 44, 45, 46    # IMU ankle
                            ]
        
        # 선택된 열에 부여할 이름
        self.col_names = ['activity_id',
                            'acc_x_hand', 'acc_y_hand', 'acc_z_hand',
                            'gyro_x_hand', 'gyro_y_hand', 'gyro_z_hand',
                            'acc_x_chest', 'acc_y_chest', 'acc_z_chest',
                            'gyro_x_chest', 'gyro_y_chest', 'gyro_z_chest',
                            'acc_x_ankle', 'acc_y_ankle', 'acc_z_ankle',
                            'gyro_x_ankle', 'gyro_y_ankle', 'gyro_z_ankle'
                            ]

        # 활동 ID와 활동 이름 매핑
        self.label_map = [ 
            (0, 'other'), (1, 'lying'), (2, 'sitting'), (3, 'standing'), (4, 'walking'),
            (5, 'running'), (6, 'cycling'), (7, 'nordic walking'), (12, 'ascending stairs'),
            (13, 'descending stairs'), (16, 'vacuum cleaning'), (17, 'ironing'), (24, 'rope jumping')
        ]

        self.sensor_filter = ["acc", "gyro"]   # 필터링할 센서 종류
        self.pos_filter = ["hand", "chest", "ankle"] # 필터링할 센서 위치

        # 사용자가 선택한 센서 위치/종류에 따라 실제 사용할 열을 필터링
        self.selected_cols = self.Sensor_filter_acoording_to_pos_and_type(args.pos_select, self.pos_filter, self.col_names[1:], "position")
        if self.selected_cols is None:
            self.selected_cols = self.Sensor_filter_acoording_to_pos_and_type(args.sensor_select, self.sensor_filter, self.col_names[1:], "Sensor Type")
        else:
            self.selected_cols = self.Sensor_filter_acoording_to_pos_and_type(args.sensor_select, self.sensor_filter, self.selected_cols, "Sensor Type")
        
        # 활동 ID를 0부터 시작하는 정수 인덱스로 변환하는 딕셔너리 생성
        self.labelToId = {int(x[0]): i for i, x in enumerate(self.label_map)}

        self.all_labels = list(range(len(self.label_map)))

        # 데이터에서 제외할 활동 (예: 'other')
        self.drop_activities = [0]
        self.drop_activities = [self.labelToId[i] for i in self.drop_activities]
        self.no_drop_activites = [item for item in self.all_labels if item not in self.drop_activities]

        # 데이터 파일 이름과 피실험자 ID 매핑
        self.file_encoding = {'subject101.dat':1, 'subject102.dat':2, 'subject103.dat':3, 
                            'subject104.dat':4, 'subject105.dat':5, 'subject106.dat':6,
                            'subject107.dat':7, 'subject108.dat':8 }

        self.sub_ids_of_each_sub = {}

        # --- 2. 데이터 로딩 및 전처리 실행 ---
        # 모든 원본 데이터를 로드하고 기본적인 전처리를 수행
        self.data_x, self.data_y = self.load_all_the_data(self.data_path)

        # 슬라이딩 윈도우 인덱스를 생성
        self.train_slidingwindows, self.activity_per_windows = self.get_the_sliding_index(self.data_x.copy(), self.data_y.copy(), "train")
        self.test_slidingwindows, _  = self.get_the_sliding_index(self.data_x.copy(), self.data_y.copy(), "test")

    def load_all_the_data(self, data_path):
        """
        모든 피실험자의 원본 데이터를 로드하고 전처리합니다.
        이미 전처리된 pickle 파일이 있다면, 해당 파일을 로드하여 시간을 절약합니다.
        """
        # 전처리된 데이터가 저장될 파일 경로
        saved_data_path = os.path.join(self.pkl_save_path, f"preprocessed_{self.data_name}.pickle")

        # 이미 전처리된 파일이 존재하는 경우
        if os.path.exists(saved_data_path):
            print(f"전처리된 파일이 존재합니다. 경로: {saved_data_path}. \n파일을 로드합니다...")
            with open(saved_data_path, 'rb') as f:
                data = pickle.load(f)
            
            # 사용자가 선택한 열('selected_cols')만 필터링하여 사용
            data_x = data['data_x'][["sub_id"] + self.selected_cols + ["sub"]]
            data_y = data['data_y']
        
        # 전처리된 파일이 없는 경우 (처음 실행 시)
        else:
            print(f"전처리된 pickle 파일이 없습니다: {saved_data_path} \n{self.data_name} 데이터 전처리를 시작합니다...")
            
            if not os.path.exists(self.pkl_save_path):
                os.makedirs(self.pkl_save_path)

            file_list = os.listdir(data_path)
            
            df_dict = {}
            for file in file_list:
                if file == 'subject109.dat': continue # 피실험자 9는 데이터가 부족하여 제외
                
                # 원본 데이터 파일을 공백 기준으로 읽어옴
                sub_data = pd.read_table(os.path.join(data_path, file), header=None, sep=r'\s+')
                sub_data = sub_data.iloc[:, self.used_cols] # 정의된 열만 선택
                sub_data.columns = self.col_names           # 열 이름 부여

                # 결측치가 있을 경우, 선형 보간법으로 채움
                sub_data = sub_data.interpolate(method='linear', limit_direction='both')
                sub = int(self.file_encoding[file])
                sub_data['sub_id'] = sub  # 피실험자 ID 열 추가
                sub_data["sub"] = sub

                if sub not in self.sub_ids_of_each_sub.keys():
                    self.sub_ids_of_each_sub[sub] = []
                self.sub_ids_of_each_sub[sub].append(sub)
                df_dict[self.file_encoding[file]] = sub_data   

            # 모든 피실험자의 데이터를 하나의 DataFrame으로 합침
            df_all = pd.concat(df_dict)

            # 다운샘플링: 원본 데이터는 약 100Hz인데, 3개 행 중 1개만 선택하여 약 33Hz로 주파수를 낮춤
            # 이는 계산량을 줄이고 데이터의 핵심 특징에 집중하기 위함입니다.
            df_all.reset_index(drop=True, inplace=True)
            index_list = list(np.arange(0, df_all.shape[0], 3))
            df_all = df_all.iloc[index_list]

            df_all = df_all.set_index('sub_id')

            # 활동 ID를 0부터 시작하는 인덱스로 변환
            df_all["activity_id"] = df_all["activity_id"].map(self.labelToId)

            # 열 순서 재정렬
            if self.selected_cols:
                df_all = df_all[self.selected_cols + ["sub"] + ["activity_id"]]
            else:
                df_all = df_all[self.col_names[1:] + ["sub"] + ["activity_id"]]

            # 데이터(X)와 레이블(Y)로 분리
            data_y = df_all.iloc[:, -1]
            data_x = df_all.iloc[:, :-1]

            data_x = data_x.reset_index()

            # 신호 필터링 (노이즈, 중력 가속도 제거) 옵션이 켜져 있으면 실행
            if self.filtering:
                data_x = self.Sensor_data_noise_grav_filtering(data_x.set_index('sub_id').copy())

            # 전처리된 데이터를 나중을 위해 pickle 파일로 저장
            data = {'data_x': data_x, 'data_y': data_y}
            with open(saved_data_path, 'wb') as f:
                pickle.dump(data, f)

        return data_x, data_y
    
    def update_train_val_test_keys(self):
        """
        LOCV의 각 폴드(iteration)가 시작될 때마다 호출되어 다음을 수행합니다:
        1. 현재 폴드에 맞는 훈련/검증/테스트 피실험자 ID를 설정
        2. 데이터를 정규화 (훈련 데이터 기준으로)
        3. 훈련/검증/테스트에 사용할 윈도우 인덱스를 분리
        """
        # --- 1. 훈련/테스트 피실험자 ID 업데이트 ---
        self.test_keys = self.LOCV_keys[self.index_of_cv]
        self.train_keys = [key for key in self.all_keys if key not in self.test_keys]
        # 다음 이터레이션을 위해 CV 인덱스를 1 증가
        self.index_of_cv = self.index_of_cv + 1

        # --- 2. 데이터 정규화 ---
        if self.datanorm_type is not None:
            # 훈련/검증에 사용할 피실험자들의 데이터만 모음
            train_vali_x = pd.DataFrame()
            for sub in self.train_keys:
                temp = self.data_x[self.data_x["sub"] == sub]
                train_vali_x = pd.concat([train_vali_x, temp])
            
            # 테스트에 사용할 피실험자의 데이터만 모음
            test_x = pd.DataFrame()
            for sub in self.test_keys:
                temp = self.data_x[self.data_x["sub"] == sub]
                test_x = pd.concat([test_x, temp])

            # 훈련 데이터 기준으로 정규화를 수행. 테스트 데이터 정보가 훈련 과정에 누수되는 것을 방지.
            train_vali_x, test_x = self.normalization(train_vali_x, test_x)

            # 정규화된 데이터를 다시 합침
            self.normalized_data_x = pd.concat([train_vali_x, test_x])
            self.normalized_data_x.sort_index(inplace=True)
        else:
            self.normalized_data_x = self.data_x.copy()

        # --- 3. 윈도우 인덱스 분할 ---
        all_test_keys = self.test_keys.copy()

        # 테스트 윈도우 인덱스 로드 또는 생성
        test_file_name = os.path.join(self.pkl_save_path,
                                      f"{self.data_name}_test_windowsize_{self.windowsize}_subject_{self.index_of_cv}_filtered_{self.filtering}.pickle")
        if os.path.exists(test_file_name):
            with open(test_file_name, 'rb') as handle:
                self.test_window_index = pickle.load(handle)
        else:
            self.test_window_index = []
            for index, window in enumerate(self.test_slidingwindows):
                sub_id = window[0] # 윈도우의 피실험자 ID
                if sub_id in all_test_keys: # 해당 윈도우가 테스트 피실험자의 것이면 추가
                    self.test_window_index.append(index)
            with open(test_file_name, 'wb') as handle:
                pickle.dump(self.test_window_index, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # 훈련+검증 윈도우 인덱스 로드 또는 생성
        train_file_name = os.path.join(self.pkl_save_path,
                                       f"{self.data_name}_train_windowsize_{self.windowsize}_subject_{self.index_of_cv}_filtered_{self.filtering}.pickle")
        if os.path.exists(train_file_name):
            with open(train_file_name, 'rb') as handle:
                train_vali_window_index = pickle.load(handle)
        else:
            train_vali_window_index = []
            for index, window in enumerate(self.train_slidingwindows):
                sub_id = window[0]
                if sub_id not in all_test_keys: # 윈도우가 훈련 피실험자의 것이면 추가
                    train_vali_window_index.append(index)
            with open(train_file_name, 'wb') as handle:
                pickle.dump(train_vali_window_index, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        # 훈련+검증 윈도우 인덱스를 훈련용과 검증용으로 분할 (계층적 샘플링)
        # 활동별 비율을 유지하면서 분할하여 데이터 불균형 문제를 완화
        self.train_window_index, self.vali_window_index, _, _ = self.stratified_train_valid_split(
            windows=train_vali_window_index, 
            activities=[self.activity_per_windows[i] for i in train_vali_window_index],
            valid_ratio=1.0 - self.train_vali_quote
        )

    def normalization(self, train_vali, test):
        """
        데이터를 정규화합니다. 훈련 데이터의 통계량(평균, 표준편차)을 사용하여 테스트 데이터도 정규화합니다.
        """
        train_vali_sensors = train_vali.iloc[:, 1:-1] # 센서 데이터만 선택 (sub_id, sub 열 제외)
        self.normalizer = Normalizer(self.datanorm_type)
        self.normalizer.fit(train_vali_sensors) # 훈련 데이터로 정규화기(normalizer)를 학습
        train_vali_sensors = self.normalizer.normalize(train_vali_sensors) # 훈련 데이터 정규화
        train_vali_sensors = pd.concat([train_vali.iloc[:, 0], train_vali_sensors, train_vali.iloc[:, -1]], axis=1)
        
        test_sensors = test.iloc[:, 1:-1]
        test_sensors = self.normalizer.normalize(test_sensors) # 학습된 정규화기로 테스트 데이터 정규화
        test_sensors = pd.concat([test.iloc[:, 0], test_sensors, test.iloc[:, -1]], axis=1)
        return train_vali_sensors, test_sensors

    def Sensor_data_noise_grav_filtering(self, df):
        """
        가속도계 신호에서 중력 성분을 제거하고 신체 움직임 성분만 남기는 필터링을 수행합니다.
        """
        all_columns = list(df.columns)[:-1]
        filtered_data = []
        for sub_id in df.index.unique():
            temp = df.loc[sub_id, all_columns]
            filtered_temp = pd.DataFrame()

            for col in temp.columns:
                t_signal = np.array(temp[col])
                if 'acc' in col: # 가속도계 신호인 경우
                    # 신호를 필터링하여 중력 성분(grav_acc)과 신체 활동 성분(body_acc)으로 분리
                    grav_acc, body_acc = components_selection_one_signal(
                        t_signal, self.freq1, self.freq2, self.sampling_freq)
                    filtered_temp[col] = body_acc # 신체 활동 성분만 저장
                else: # 자이로스코프 신호는 그대로 사용
                    filtered_temp[col] = t_signal
            
            filtered_temp.index = temp.index
            filtered_data.append(filtered_temp)

        filtered_data = pd.concat(filtered_data)
        filtered_data = pd.concat([filtered_data, df.iloc[:, -1]], axis=1)
        return filtered_data.reset_index()

    def get_the_sliding_index(self, data_x, data_y, flag="train"):
        """
        연속적인 센서 데이터를 슬라이딩 윈도우 방식으로 잘라 인덱스를 생성합니다.
        메모리 효율을 위해 실제 데이터가 아닌 [피실험자 ID, 시작 인덱스, 끝 인덱스]만 저장합니다.
        """
        data_y = data_y.reset_index()
        data_x["activity_id"] = data_y["activity_id"]

        # 피실험자별로 데이터 블록을 구분
        data_x['act_block'] = (data_x['sub_id'].shift(1) != data_x['sub_id']).astype(int).cumsum()
        
        # 윈도우가 이동하는 간격(displacement) 설정
        # 훈련 시에는 50% 겹치게(overlap) 하여 데이터 양을 늘리고,
        if flag == "train":
            displacement = int(0.5 * self.windowsize)
        # 테스트 시에는 90% 겹치게 하여 더 촘촘하게 데이터를 검사함으로써 성능 평가의 정확도를 높입니다.
        elif flag == "test":
            displacement = int(0.1 * self.windowsize)

        window_index = []
        activity_per_window = []
        for index in data_x.act_block.unique():
            temp_df = data_x[data_x["act_block"] == index]
            assert len(temp_df["sub_id"].unique()) == 1
            sub_id = temp_df["sub_id"].unique()[0]
            start = temp_df.index[0]
            end = start + self.windowsize

            while end <= temp_df.index[-1] + 1:
                # 윈도우 내에서 가장 빈번하게 나타나는 활동을 해당 윈도우의 레이블로 결정 (majority voting)
                curr_activity = temp_df.loc[start:end-1, "activity_id"].mode().loc[0]
                if curr_activity not in self.drop_activities:
                    window_index.append([sub_id, start, end])
                    activity_per_window.append(curr_activity)
                
                start = start + displacement
                end = start + self.windowsize

        return window_index, activity_per_window
    
    def stratified_train_valid_split(self, windows, activities, valid_ratio=0.1):
        """
        훈련 데이터를 다시 훈련셋과 검증셋으로 분할합니다.
        'stratify' 옵션을 사용하여 각 활동(클래스)의 비율이 분할 후에도 동일하게 유지되도록 합니다.
        """
        counts = Counter(activities)
        for cls, count in counts.items():
            if count < 2:
                print(f"클래스 {self.label_map[cls]}는 샘플이 {count}개밖에 없습니다.")

        # scikit-learn의 train_test_split 함수를 사용하여 계층적 분할 수행
        X_train, X_valid, y_train, y_valid = train_test_split(
            windows, activities, 
            test_size=valid_ratio, 
            stratify=activities, 
            random_state=self.args.seed
        )
        return X_train, X_valid, y_train, y_valid
    
    def Sensor_filter_acoording_to_pos_and_type(self, select, filter_options, all_col_names, filtertype):
        """
        사용자 인자에 따라 특정 위치(손, 가슴 등)나 특정 종류(가속도, 자이로 등)의 센서 데이터만 선택하는 유틸리티 함수입니다.
        """
        if select is not None:
            if filter_options is None:
                raise Exception(f'이 데이터셋은 센서 {filtertype}로 선택할 수 없습니다!')
            else:
                col_names = []
                for col in all_col_names:
                    selected = False
                    for one_select in select:
                        assert one_select in filter_options
                        if one_select in col:
                            selected = True
                    if selected:
                        col_names.append(col)
                return col_names
        else:
            return None

class Dataset(object):
    """
    PyTorch의 `torch.utils.data.Dataset` 클래스를 감싸는 래퍼 클래스입니다.
    `DataLoader`가 데이터를 배치 단위로 가져올 수 있도록 `__getitem__`과 `__len__` 메소드를 구현합니다.
    """
    def __init__(self, dataset, flag):
        self.flag = flag
        # 플래그에 따라 훈련/검증/테스트에 사용할 윈도우 인덱스 목록을 설정
        if self.flag == "train":
            self.slidingwindows = dataset.train_slidingwindows
            self.window_index = dataset.train_window_index
        elif self.flag == "valid":
            self.slidingwindows = dataset.train_slidingwindows
            self.window_index = dataset.vali_window_index
        else: # test
            self.slidingwindows = dataset.test_slidingwindows
            self.window_index = dataset.test_window_index

        # 최종적으로 사용할 클래스 목록 (예: 'other' 제외)
        classes = dataset.no_drop_activites
        # 클래스 레이블을 0부터 시작하는 연속적인 정수로 다시 매핑합니다. (예: {1:0, 2:1, ...})
        # PyTorch의 CrossEntropyLoss는 0부터 시작하는 클래스 인덱스를 요구하기 때문입니다.
        self.class_transform = {x: i for i, x in enumerate(classes)}

        # 정규화된 전체 데이터(X)와 레이블(Y)을 참조
        self.data_x = dataset.normalized_data_x
        self.data_y = dataset.data_y

        # 한 윈도우의 길이(타임스텝 수)와 채널 수(센서 피처 수)를 계산
        self.input_length = self.slidingwindows[0][2] - self.slidingwindows[0][1] # 예: 168
        self.channel_in = self.data_x.shape[1] - 2 # sub_id, sub 열 제외. 예: 9

    def __getitem__(self, index):
        """
        DataLoader에 의해 호출되며, 주어진 index에 해당하는 데이터 샘플(x, y) 하나를 반환합니다.
        """
        # 현재 분할(train/valid/test)에 맞는 윈도우 목록에서 실제 윈도우의 인덱스를 가져옴
        index = self.window_index[index]
        # 해당 윈도우의 시작과 끝 인덱스를 가져옴
        start_index = self.slidingwindows[index][1]
        end_index = self.slidingwindows[index][2]
        
        # 전체 데이터프레임(`data_x`)에서 해당 윈도우만큼의 데이터를 슬라이싱하여 가져옴
        # .iloc[start:end, 1:-1]은 윈도우 길이만큼의 행과, 'sub_id'와 'sub' 열을 제외한 센서 데이터 열을 선택
        # .values를 통해 NumPy 배열로 변환
        # sample_x shape: (168, 9)
        sample_x = self.data_x.iloc[start_index:end_index, 1:-1].values
        
        # 해당 윈도우 내의 레이블들 중 가장 빈번한 값(majority vote)을 찾고,
        # self.class_transform을 이용해 0부터 시작하는 최종 레이블로 변환
        sample_y = self.class_transform[self.data_y.iloc[start_index:end_index].mode().loc[0]]

        return sample_x, sample_y

    def __len__(self):
        """
        DataLoader가 전체 데이터셋의 크기를 알 수 있도록, 현재 분할(train/valid/test)에 포함된 총 윈도우(샘플) 수를 반환합니다.
        """
        return len(self.window_index)