import os
import time
import torch
import numpy as np
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
import yaml
from typing import Tuple, Dict, List, Optional, Any, Union

# --- 수정된 부분 1: 새로운 특징 추출 함수와 인코더 클래스를 임포트합니다. ---
from encoders.base.deepconvlstm_encoder import DeepConvLSTMEncoder
from encoders.base.deepconvlstm_attn_encoder import DeepConvLSTMAttnEncoder
from encoders.base.sa_har_encoder import SAHAREncoder
from dataloaders.data_utils import compute_batch_ecdf_features, compute_batch_extended_features
# --------------------------------------------------------------------

from utils.training_utils import EarlyStopping, adjust_learning_rate, set_seed
from utils.logger import Logger

# Initialize global logger
Logger.initialize(log_dir='logs')

class EncoderTrainer:
    """
    ECDF feature prediction encoder training class
    """"
    def __init__(self, args: Any, model: nn.Module, save_path: str):
        """"
        Initialize the encoder trainer
        
        Args:
            args: configuration parameters
            model: encoder model to train
            save_path: model save path
        """"
        self.model = model
        self.args = args # args를 저장하여 나중에 사용
        self.device = args.device
        self.model.to(self.device)
        
        self.logger = Logger(f"encoder_{args.encoder_type}")
        self.logger.info(f"Using device: {self.device}")
        
        self.criterion = nn.MSELoss()
        
        if args.optimizer == "Adam":
            self.optimizer = optim.Adam(self.model.parameters(), lr=args.learning_rate)
        else:
            self.optimizer = optim.SGD(self.model.parameters(), lr=args.learning_rate)
        
        self.save_path = save_path
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        
        self.epochs = args.train_epochs
        
        self.early_stopping = EarlyStopping(patience=args.early_stop_patience, verbose=True, 
                                          logger_name=f"es_encoder_{args.encoder_type}")

    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        self.model.train()
        train_loss = []
        epoch_time = time.time()
        batch_count = 0
        for batch_x, _ in train_loader:
            batch_count += 1
            self.logger.debug(f"Processing batch #{batch_count} in train epoch")
            
            batch_x = batch_x.float().to(self.device)
            
            # --- 수정된 부분 2: encoder_type에 따라 다른 특징 추출 함수를 호출합니다. ---
            if self.args.encoder_type == 'deepconvlstm_attn_extended':
                batch_features = torch.tensor(compute_batch_extended_features(batch_x),
                                            dtype=torch.float32).to(self.device)
            else:
                batch_features = torch.tensor(compute_batch_ecdf_features(batch_x), 
                                            dtype=torch.float32).to(self.device)
            # --------------------------------------------------------------------

            predicted_features = self.model(batch_x)
            
            if hasattr(self.model, 'calculate_loss'):
                loss, _ = self.model.calculate_loss(predicted_features, batch_features)
            else:
                loss = self.criterion(predicted_features, batch_features)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            train_loss.append(loss.item())
        
        epoch_time = time.time() - epoch_time
        train_loss = np.average(train_loss)
        self.logger.info(f"Completed epoch with {batch_count} batches")
        
        return train_loss, epoch_time
    
    def validate(self, valid_loader: DataLoader) -> float:
        self.model.eval()
        valid_loss = []
        batch_count = 0
        
        with torch.no_grad():
            for batch_x, _ in valid_loader:  
                batch_count += 1
                self.logger.debug(f"Processing batch #{batch_count} in validation")
                
                batch_x = batch_x.float().to(self.device)
                
                # --- 수정된 부분 3: train_epoch와 동일하게 수정합니다. ---
                if self.args.encoder_type == 'deepconvlstm_attn_extended':
                    batch_features = torch.tensor(compute_batch_extended_features(batch_x),
                                                dtype=torch.float32).to(self.device)
                else:
                    batch_features = torch.tensor(compute_batch_ecdf_features(batch_x), 
                                                dtype=torch.float32).to(self.device)
                # --------------------------------------------------------------------

                predicted_features = self.model(batch_x)
                
                if hasattr(self.model, 'calculate_loss'):
                    loss, _ = self.model.calculate_loss(predicted_features, batch_features)
                else:
                    loss = self.criterion(predicted_features, batch_features)
                
                valid_loss.append(loss.item())
        
        valid_loss = np.average(valid_loss)
        self.logger.info(f"Completed validation with {batch_count} batches")
        
        return valid_loss
    
    def train(self, train_loader: DataLoader, valid_loader: DataLoader) -> nn.Module:
        self.logger.info(f"Starting encoder training, saving to: {self.save_path}")
        
        for epoch in range(self.epochs):
            train_loss, epoch_time = self.train_epoch(train_loader)
            self.logger.info(f"Epoch: {epoch+1}, train_loss: {train_loss:.7f}, time: {epoch_time:.2f}s")
            
            valid_loss = self.validate(valid_loader)
            self.logger.info(f"Validation: Epoch: {epoch+1}, Train Loss: {train_loss:.7f}, Valid Loss: {valid_loss:.7f}")
            
            self.early_stopping(valid_loss, self.model, self.save_path, None)
            if self.early_stopping.early_stop:
                self.logger.info("Early stopping triggered")
                break
        
        self.logger.info("Encoder training completed")
        return self.model


def create_encoder(args: Any) -> nn.Module:
    logger = Logger("encoder_creator")
    
    encoder_args = {
        'input_channels': args.input_channels,
        'window_size': args.window_size,
        'output_size': args.output_size,
        'device': args.device
    }
    
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'configs', 'model.yaml')
    with open(config_path, mode='r') as config_file:
        model_config = yaml.load(config_file, Loader=yaml.FullLoader)
    
    encoder_config = model_config['efnet_encoder']
    
    # --- 수정된 부분 4: 새로운 인코더 타입을 위한 elif 블록을 추가합니다. ---
    if args.encoder_type == 'deepconvlstm':
        encoder_args.update(encoder_config.get('deepconvlstm', {}))
        model_class = DeepConvLSTMEncoder
        logger.info(f"Using DeepConvLSTM encoder configuration")

    elif args.encoder_type == 'deepconvlstm_attn':
        encoder_args.update(encoder_config.get('deepconvlstm_attn', {}))
        model_class = DeepConvLSTMAttnEncoder
        logger.info(f"Using DeepConvLSTM with Attention encoder configuration")

    elif args.encoder_type == 'deepconvlstm_attn_extended':
        from encoders.base.deepconvlstm_attn_extended_encoder import DeepConvLSTMAttnExtendedEncoder
        encoder_args.update(encoder_config.get('deepconvlstm_attn', {})) # 기본 설정은 기존 attn 모델과 공유
        model_class = DeepConvLSTMAttnExtendedEncoder
        logger.info(f"Using DeepConvLSTM with Extended Attention encoder configuration")

    elif args.encoder_type == 'sa_har':
        encoder_args.update(encoder_config.get('sa_har', {}))
        model_class = SAHAREncoder
        logger.info(f"Using SA-HAR encoder configuration")
        
    else:
        logger.error(f"Unsupported encoder type: {args.encoder_type}")
        raise ValueError(f"Unsupported encoder type: {args.encoder_type}")
    # --------------------------------------------------------------------
    
    encoder = model_class(encoder_args)
    
    logger.info(f"Created {args.encoder_type} encoder")
    return encoder

def load_pretrained_encoder(encoder: nn.Module, path: str) -> nn.Module:
    logger = Logger("encoder_loader")
    logger.info(f"Loading pretrained encoder from: {path}")
    
    if not os.path.exists(path):
        logger.error(f"Checkpoint file not found: {path}")
        raise FileNotFoundError(f"Checkpoint file not found: {path}")
    
    try:
        checkpoint = torch.load(path, map_location=encoder.device, weights_only=False)
        encoder.load_state_dict(checkpoint['model_state_dict'])
        val_loss = checkpoint.get('val_loss', 'N/A')
        logger.info(f"Successfully loaded model with validation loss: {val_loss}")
    except Exception as e:
        logger.error(f"Error loading checkpoint: {str(e)}")
        raise RuntimeError(f"Failed to load checkpoint: {str(e)}")
    
    return encoder