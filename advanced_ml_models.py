import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import ta
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime, timedelta
import joblib

logger = logging.getLogger(__name__)

# ============================================
# LSTM MODEL
# ============================================

class CryptoLSTM(nn.Module):
    """Kripto fiyat tahmini için LSTM modeli"""
    
    def __init__(self, input_size: int, hidden_size: int = 128, 
                 num_layers: int = 3, output_size: int = 1, 
                 dropout: float = 0.2):
        super(CryptoLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size * 2,  # Bidirectional
            num_heads=8,
            dropout=dropout
        )
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, output_size)
        
        # Activation and regularization
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.batch_norm1 = nn.BatchNorm1d(hidden_size)
        self.batch_norm2 = nn.BatchNorm1d(hidden_size // 2)
        
    def forward(self, x):
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Attention mechanism
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Use last time step output with attention
        out = attn_out[:, -1, :]
        
        # Fully connected layers
        out = self.fc1(out)
        out = self.batch_norm1(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.fc2(out)
        out = self.batch_norm2(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.fc3(out)
        
        return out

# ============================================
# TRANSFORMER MODEL
# ============================================

class CryptoTransformer(nn.Module):
    """Transformer tabanlı kripto tahmin modeli"""
    
    def __init__(self, input_dim: int, d_model: int = 512, 
                 n_heads: int = 8, n_layers: int = 6, 
                 d_ff: int = 2048, dropout: float = 0.1):
        super(CryptoTransformer, self).__init__()
        
        self.d_model = d_model
        
        # Input embedding
        self.input_embedding = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation='relu'
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=n_layers
        )
        
        # Output layers
        self.fc1 = nn.Linear(d_model, d_model // 2)
        self.fc2 = nn.Linear(d_model // 2, d_model // 4)
        self.fc3 = nn.Linear(d_model // 4, 3)  # 3 outputs: [price_change, direction, volatility]
        
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        
    def forward(self, src):
        # Input embedding
        src = self.input_embedding(src)
        src = self.pos_encoding(src)
        
        # Transformer encoding
        output = self.transformer_encoder(src)
        
        # Global average pooling
        output = torch.mean(output, dim=1)
        
        # Output layers
        output = self.fc1(output)
        output = self.relu(output)
        output = self.dropout(output)
        
        output = self.fc2(output)
        output = self.relu(output)
        output = self.dropout(output)
        
        output = self.fc3(output)
        
        return output

class PositionalEncoding(nn.Module):
    """Pozisyonel encoding for Transformer"""
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

# ============================================
# TEMPORAL FUSION TRANSFORMER (TFT)
# ============================================

class TemporalFusionTransformer(nn.Module):
    """Google'ın TFT modeli - Zaman serisi tahmini için"""
    
    def __init__(self, 
                 input_size: int,
                 hidden_size: int = 256,
                 num_heads: int = 4,
                 num_encoder_layers: int = 2,
                 num_decoder_layers: int = 2,
                 dropout: float = 0.1):
        super(TemporalFusionTransformer, self).__init__()
        
        # Variable Selection Networks
        self.vsn_historical = VariableSelectionNetwork(input_size, hidden_size)
        self.vsn_future = VariableSelectionNetwork(input_size, hidden_size)
        
        # LSTM Encoder-Decoder
        self.encoder_lstm = nn.LSTM(
            hidden_size, hidden_size, 
            batch_first=True, bidirectional=True
        )
        self.decoder_lstm = nn.LSTM(
            hidden_size * 2, hidden_size,
            batch_first=True
        )
        
        # Multi-Head Attention
        self.attention = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=dropout
        )
        
        # Gated Residual Networks
        self.grn_historical = GatedResidualNetwork(hidden_size * 2, hidden_size)
        self.grn_future = GatedResidualNetwork(hidden_size, hidden_size)
        
        # Output
        self.output_layer = nn.Linear(hidden_size, 3)  # [price, volume, volatility]
        
    def forward(self, historical_inputs, future_inputs):
        # Variable selection
        selected_historical = self.vsn_historical(historical_inputs)
        selected_future = self.vsn_future(future_inputs)
        
        # Encode historical
        encoded, (hidden, cell) = self.encoder_lstm(selected_historical)
        
        # Decode with future inputs
        decoded, _ = self.decoder_lstm(selected_future, (hidden, cell))
        
        # Apply attention
        attn_output, _ = self.attention(decoded, encoded, encoded)
        
        # Gated residual connections
        grn_output = self.grn_historical(torch.cat([encoded, attn_output], dim=-1))
        
        # Final output
        output = self.output_layer(grn_output[:, -1, :])
        
        return output

class VariableSelectionNetwork(nn.Module):
    """Önemli değişkenleri seçen network"""
    
    def __init__(self, input_size: int, hidden_size: int):
        super(VariableSelectionNetwork, self).__init__()
        
        self.flattened_size = input_size
        self.hidden_size = hidden_size
        
        # Gating layers
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, input_size)
        
        # Variable weights
        self.softmax = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # Calculate variable weights
        weights = self.fc1(x)
        weights = torch.relu(weights)
        weights = self.fc2(weights)
        weights = self.softmax(weights)
        
        # Apply weights
        selected = x * weights
        
        return selected

class GatedResidualNetwork(nn.Module):
    """Gated Residual Network component"""
    
    def __init__(self, input_size: int, hidden_size: int):
        super(GatedResidualNetwork, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        
        self.gate = nn.Linear(hidden_size, hidden_size)
        self.sigmoid = nn.Sigmoid()
        
        self.layer_norm = nn.LayerNorm(hidden_size)
        
    def forward(self, x):
        # ELU activation
        a = torch.nn.functional.elu(self.fc1(x))
        
        # Gate
        gate = self.sigmoid(self.gate(a))
        
        # Gated connection
        a = self.fc2(a)
        
        # Residual connection if dimensions match
        if x.shape[-1] == a.shape[-1]:
            output = gate * a + (1 - gate) * x
        else:
            output = gate * a
        
        # Layer normalization
        output = self.layer_norm(output)
        
        return output

# ============================================
# DATASET VE FEATURE ENGINEERING
# ============================================

class CryptoDataset(Dataset):
    """Kripto veri seti sınıfı"""
    
    def __init__(self, data: pd.DataFrame, sequence_length: int = 60, 
                 prediction_horizon: int = 24):
        self.data = data
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        
        # Feature engineering
        self.features = self._create_features()
        
        # Scaling
        self.scaler = MinMaxScaler()
        self.scaled_features = self.scaler.fit_transform(self.features)
        
    def _create_features(self) -> pd.DataFrame:
        """Özellik mühendisliği"""
        df = self.data.copy()
        
        # Teknik göstergeler
        df['RSI'] = ta.momentum.RSIIndicator(df['close']).rsi()
        df['MACD'] = ta.trend.MACD(df['close']).macd()
        df['BB_upper'] = ta.volatility.BollingerBands(df['close']).bollinger_hband()
        df['BB_lower'] = ta.volatility.BollingerBands(df['close']).bollinger_lband()
        df['ATR'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()
        
        # Volume göstergeleri
        df['OBV'] = ta.volume.OnBalanceVolumeIndicator(df['close'], df['volume']).on_balance_volume()
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        
        # Price features
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        df['volatility'] = df['returns'].rolling(window=20).std()
        
        # Time features
        df['hour'] = pd.to_datetime(df.index).hour
        df['day_of_week'] = pd.to_datetime(df.index).dayofweek
        df['month'] = pd.to_datetime(df.index).month
        
        # Lag features
        for i in [1, 3, 6, 12, 24]:
            df[f'close_lag_{i}'] = df['close'].shift(i)
            df[f'volume_lag_{i}'] = df['volume'].shift(i)
        
        # Rolling statistics
        for window in [12, 24, 48]:
            df[f'rolling_mean_{window}'] = df['close'].rolling(window=window).mean()
            df[f'rolling_std_{window}'] = df['close'].rolling(window=window).std()
            df[f'rolling_max_{window}'] = df['close'].rolling(window=window).max()
            df[f'rolling_min_{window}'] = df['close'].rolling(window=window).min()
        
        # Drop NaN values
        df = df.dropna()
        
        return df
    
    def __len__(self):
        return len(self.scaled_features) - self.sequence_length - self.prediction_horizon
    
    def __getitem__(self, idx):
        # Input sequence
        x = self.scaled_features[idx:idx + self.sequence_length]
        
        # Target (next price movement)
        y = self.scaled_features[idx + self.sequence_length + self.prediction_horizon - 1]
        
        return torch.FloatTensor(x), torch.FloatTensor(y)

# ============================================
# ENSEMBLE MODEL
# ============================================

class EnsembleModel:
    """Multiple model ensemble for better predictions"""
    
    def __init__(self):
        self.models = {
            'lstm': CryptoLSTM(input_size=50, hidden_size=128, num_layers=3),
            'transformer': CryptoTransformer(input_dim=50, d_model=256),
            'tft': TemporalFusionTransformer(input_size=50)
        }
        
        self.weights = {
            'lstm': 0.35,
            'transformer': 0.35,
            'tft': 0.30
        }
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def train_ensemble(self, train_data: DataLoader, val_data: DataLoader, 
                       epochs: int = 100):
        """Ensemble modellerini eğit"""
        
        for model_name, model in self.models.items():
            logger.info(f"Training {model_name}...")
            
            model = model.to(self.device)
            optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
            criterion = nn.MSELoss()
            
            best_val_loss = float('inf')
            patience_counter = 0
            patience = 10
            
            for epoch in range(epochs):
                # Training
                model.train()
                train_loss = 0
                
                for batch_x, batch_y in train_data:
                    batch_x = batch_x.to(self.device)
                    batch_y = batch_y.to(self.device)
                    
                    optimizer.zero_grad()
                    
                    if model_name == 'tft':
                        # TFT requires special input handling
                        historical = batch_x[:, :-24, :]
                        future = batch_x[:, -24:, :]
                        outputs = model(historical, future)
                    else:
                        outputs = model(batch_x)
                    
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    
                    optimizer.step()
                    train_loss += loss.item()
                
                # Validation
                model.eval()
                val_loss = 0
                
                with torch.no_grad():
                    for batch_x, batch_y in val_data:
                        batch_x = batch_x.to(self.device)
                        batch_y = batch_y.to(self.device)
                        
                        if model_name == 'tft':
                            historical = batch_x[:, :-24, :]
                            future = batch_x[:, -24:, :]
                            outputs = model(historical, future)
                        else:
                            outputs = model(batch_x)
                        
                        loss = criterion(outputs, batch_y)
                        val_loss += loss.item()
                
                avg_train_loss = train_loss / len(train_data)
                avg_val_loss = val_loss / len(val_data)
                
                logger.info(f"Epoch {epoch+1}/{epochs} - "
                          f"Train Loss: {avg_train_loss:.4f}, "
                          f"Val Loss: {avg_val_loss:.4f}")
                
                # Early stopping
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                    # Save best model
                    torch.save(model.state_dict(), f'models/{model_name}_best.pt')
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        logger.info(f"Early stopping triggered for {model_name}")
                        break
                
                scheduler.step()
            
            # Load best model
            model.load_state_dict(torch.load(f'models/{model_name}_best.pt'))
    
    def predict(self, x: torch.Tensor) -> Dict:
        """Ensemble tahmin"""
        predictions = {}
        
        x = x.to(self.device)
        
        for model_name, model in self.models.items():
            model = model.to(self.device)
            model.eval()
            
            with torch.no_grad():
                if model_name == 'tft':
                    historical = x[:, :-24, :]
                    future = x[:, -24:, :]
                    pred = model(historical, future)
                else:
                    pred = model(x)
                
                predictions[model_name] = pred.cpu().numpy()
        
        # Weighted ensemble
        ensemble_pred = sum(
            self.weights[name] * predictions[name] 
            for name in predictions
        )
        
        # Interpret predictions
        price_change = ensemble_pred[0] if len(ensemble_pred.shape) == 1 else ensemble_pred[:, 0]
        direction = 1 if price_change > 0 else -1
        confidence = abs(price_change)
        
        return {
            'price_change': float(price_change),
            'direction': direction,
            'confidence': float(confidence),
            'individual_predictions': predictions,
            'ensemble_weights': self.weights
        }

# ============================================
# MODEL TRAINER VE EVALUATOR
# ============================================

class ModelTrainer:
    """Model eğitimi ve değerlendirme"""
    
    def __init__(self):
        self.ensemble = EnsembleModel()
        self.metrics = {}
        
    def prepare_data(self, df: pd.DataFrame, test_size: float = 0.2):
        """Veriyi hazırla"""
        dataset = CryptoDataset(df)
        
        # Train/test split
        train_size = int((1 - test_size) * len(dataset))
        val_size = len(dataset) - train_size
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )
        
        # Data loaders
        train_loader = DataLoader(
            train_dataset, batch_size=32, shuffle=True, num_workers=4
        )
        val_loader = DataLoader(
            val_dataset, batch_size=32, shuffle=False, num_workers=4
        )
        
        return train_loader, val_loader
    
    def train(self, df: pd.DataFrame):
        """Modeli eğit"""
        logger.info("Preparing data...")
        train_loader, val_loader = self.prepare_data(df)
        
        logger.info("Training ensemble models...")
        self.ensemble.train_ensemble(train_loader, val_loader)
        
        logger.info("Training completed!")
    
    def evaluate(self, test_data: pd.DataFrame) -> Dict:
        """Model performansını değerlendir"""
        dataset = CryptoDataset(test_data)
        test_loader = DataLoader(dataset, batch_size=1, shuffle=False)
        
        predictions = []
        actuals = []
        
        for x, y in test_loader:
            pred = self.ensemble.predict(x)
            predictions.append(pred['price_change'])
            actuals.append(y.numpy()[0])
        
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        
        # Metrics
        mae = np.mean(np.abs(predictions - actuals))
        mse = np.mean((predictions - actuals) ** 2)
        rmse = np.sqrt(mse)
        
        # Directional accuracy
        pred_direction = np.sign(predictions)
        actual_direction = np.sign(actuals)
        directional_accuracy = np.mean(pred_direction == actual_direction)
        
        # Profit factor (for trading)
        winning_trades = predictions[(predictions > 0) & (actuals > 0)]
        losing_trades = predictions[(predictions < 0) & (actuals < 0)]
        
        if len(losing_trades) > 0:
            profit_factor = np.sum(np.abs(winning_trades)) / np.sum(np.abs(losing_trades))
        else:
            profit_factor = float('inf')
        
        self.metrics = {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'directional_accuracy': directional_accuracy,
            'profit_factor': profit_factor,
            'total_predictions': len(predictions)
        }
        
        return self.metrics
    
    def save_models(self, path: str = 'models/'):
        """Modelleri kaydet"""
        for name, model in self.ensemble.models.items():
            torch.save({
                'model_state_dict': model.state_dict(),
                'model_config': model.__dict__
            }, f"{path}{name}_model.pt")
        
        # Save ensemble weights
        joblib.dump(self.ensemble.weights, f"{path}ensemble_weights.pkl")
        
        logger.info(f"Models saved to {path}")
    
    def load_models(self, path: str = 'models/'):
        """Modelleri yükle"""
        for name, model in self.ensemble.models.items():
            checkpoint = torch.load(f"{path}{name}_model.pt")
            model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load ensemble weights
        self.ensemble.weights = joblib.load(f"{path}ensemble_weights.pkl")
        
        logger.info(f"Models loaded from {path}")