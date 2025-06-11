import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf

# Handle TensorFlow 2.19+ imports
try:
    from keras.models import Sequential
    from keras.layers import LSTM, Dense, Dropout, BatchNormalization, Bidirectional
    from keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from keras.optimizers import Adam
    print("✓ Using new Keras imports (TF 2.19)")
except ImportError:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Bidirectional
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.optimizers import Adam
    print("✓ Using legacy Keras imports")

import warnings
warnings.filterwarnings('ignore')

class ImprovedForexLSTM:
    """
    Enhanced LSTM with technical indicators and better architecture
    """
    def __init__(self, sequence_length=60, lstm_units=100, dropout_rate=0.3):
        self.sequence_length = sequence_length
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.price_scaler = MinMaxScaler(feature_range=(0, 1))
        self.feature_scaler = StandardScaler()
        self.model = None
        
    def add_technical_indicators(self, df, price_col='DEXUSEU'):
        """
        Add technical indicators to improve predictions
        """
        print("🔧 Adding technical indicators...")
        
        # Simple Moving Averages
        df['SMA_5'] = df[price_col].rolling(window=5).mean()
        df['SMA_10'] = df[price_col].rolling(window=10).mean()
        df['SMA_20'] = df[price_col].rolling(window=20).mean()
        
        # Exponential Moving Averages
        df['EMA_5'] = df[price_col].ewm(span=5).mean()
        df['EMA_10'] = df[price_col].ewm(span=10).mean()
        
        # Price momentum
        df['momentum_5'] = df[price_col].pct_change(5)
        df['momentum_10'] = df[price_col].pct_change(10)
        
        # Volatility (rolling standard deviation)
        df['volatility_5'] = df[price_col].rolling(window=5).std()
        df['volatility_10'] = df[price_col].rolling(window=10).std()
        
        # RSI (Relative Strength Index)
        delta = df[price_col].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['BB_middle'] = df[price_col].rolling(window=20).mean()
        bb_std = df[price_col].rolling(window=20).std()
        df['BB_upper'] = df['BB_middle'] + (bb_std * 2)
        df['BB_lower'] = df['BB_middle'] - (bb_std * 2)
        df['BB_position'] = (df[price_col] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])
        
        # Price position relative to moving averages
        df['price_vs_sma5'] = df[price_col] / df['SMA_5'] - 1
        df['price_vs_sma20'] = df[price_col] / df['SMA_20'] - 1
        
        # Day of week effect (if available)
        if 'observation_date' in df.columns:
            df['day_of_week'] = pd.to_datetime(df['observation_date']).dt.dayofweek
            df['month'] = pd.to_datetime(df['observation_date']).dt.month
        
        # Returns
        df['returns_1d'] = df[price_col].pct_change(1)
        df['returns_5d'] = df[price_col].pct_change(5)
        
        # Drop rows with NaN values
        df = df.dropna()
        
        print(f"✓ Added {len(df.columns)} features including technical indicators")
        return df
    
    def load_and_prepare_data(self, file_path):
        """
        Load data and add technical indicators
        """
        try:
            df = pd.read_csv(file_path)
            
            # Handle FRED format
            if 'DEXUSEU' in df.columns and 'observation_date' in df.columns:
                print("✓ Detected FRED USD per EUR format")
                
                # Clean data
                df = df.dropna(subset=['DEXUSEU'])
                if df['DEXUSEU'].dtype == 'object':
                    df = df[df['DEXUSEU'] != '.']
                    df['DEXUSEU'] = pd.to_numeric(df['DEXUSEU'])
                
                # Add technical indicators
                df = self.add_technical_indicators(df)
                
                # Select features for training
                feature_cols = [
                    'DEXUSEU', 'SMA_5', 'SMA_10', 'SMA_20', 'EMA_5', 'EMA_10',
                    'momentum_5', 'momentum_10', 'volatility_5', 'volatility_10',
                    'RSI', 'BB_position', 'price_vs_sma5', 'price_vs_sma20',
                    'returns_1d', 'returns_5d'
                ]
                
                # Add day features if available
                if 'day_of_week' in df.columns:
                    feature_cols.extend(['day_of_week', 'month'])
                
                # Keep only available columns
                available_cols = [col for col in feature_cols if col in df.columns]
                self.feature_data = df[available_cols].values
                self.target_data = df['DEXUSEU'].values.reshape(-1, 1)
                self.dates = pd.to_datetime(df['observation_date']).values
                
                print(f"✓ Loaded {len(self.feature_data)} samples with {len(available_cols)} features")
                print(f"✓ Features: {available_cols}")
                
                return df
                
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return self.create_sample_data_with_features()
    
    def create_sample_data_with_features(self):
        """
        Create sample data with technical indicators for testing
        """
        print("🔄 Creating enhanced sample data...")
        
        # Create base price data
        dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
        np.random.seed(42)
        
        # More realistic USD/EUR simulation
        base_rate = 1.0800
        returns = np.random.normal(0, 0.008, len(dates))
        
        # Add some persistence and volatility clustering
        for i in range(1, len(returns)):
            returns[i] += 0.1 * returns[i-1]  # More momentum
        
        rates = [base_rate]
        for ret in returns[1:]:
            new_rate = rates[-1] * (1 + ret)
            new_rate = max(1.00, min(1.20, new_rate))
            rates.append(new_rate)
        
        df = pd.DataFrame({
            'observation_date': dates,
            'DEXUSEU': rates
        })
        
        # Add technical indicators
        df = self.add_technical_indicators(df)
        
        # Prepare features
        feature_cols = [
            'DEXUSEU', 'SMA_5', 'SMA_10', 'SMA_20', 'EMA_5', 'EMA_10',
            'momentum_5', 'momentum_10', 'volatility_5', 'volatility_10',
            'RSI', 'BB_position', 'price_vs_sma5', 'price_vs_sma20',
            'returns_1d', 'returns_5d', 'day_of_week', 'month'
        ]
        
        self.feature_data = df[feature_cols].values
        self.target_data = df['DEXUSEU'].values.reshape(-1, 1)
        self.dates = dates.values
        
        print(f"✓ Created sample data with {len(feature_cols)} features")
        return df
    
    def prepare_sequences(self):
        """
        Create sequences with multiple features
        """
        print("🔄 Preparing sequences with technical indicators...")
        
        # Scale features and target separately
        scaled_features = self.feature_scaler.fit_transform(self.feature_data)
        scaled_target = self.price_scaler.fit_transform(self.target_data)
        
        # Create sequences
        X, y = [], []
        for i in range(self.sequence_length, len(scaled_features)):
            # Use all features for input
            X.append(scaled_features[i-self.sequence_length:i])
            # Predict only the price
            y.append(scaled_target[i, 0])
        
        X, y = np.array(X), np.array(y)
        
        # Time-based splits
        total_samples = len(X)
        train_size = int(total_samples * 0.7)
        val_size = int(total_samples * 0.2)
        
        self.X_train = X[:train_size]
        self.y_train = y[:train_size]
        self.X_val = X[train_size:train_size + val_size]
        self.y_val = y[train_size:train_size + val_size]
        self.X_test = X[train_size + val_size:]
        self.y_test = y[train_size + val_size:]
        
        print(f"✓ Created sequences:")
        print(f"  • Training: {len(self.X_train)} samples")
        print(f"  • Validation: {len(self.X_val)} samples") 
        print(f"  • Test: {len(self.X_test)} samples")
        print(f"  • Input shape: {self.X_train.shape}")
        
        return X, y
    
    def build_advanced_model(self):
        """
        Build a more sophisticated LSTM architecture
        """
        print("🏗️ Building advanced LSTM model...")
        
        input_shape = (self.sequence_length, self.feature_data.shape[1])
        
        model = Sequential([
            # First bidirectional LSTM layer
            Bidirectional(LSTM(self.lstm_units, 
                              return_sequences=True, 
                              recurrent_dropout=0.1), 
                         input_shape=input_shape),
            BatchNormalization(),
            Dropout(self.dropout_rate),
            
            # Second LSTM layer
            LSTM(self.lstm_units // 2, 
                 return_sequences=True,
                 recurrent_dropout=0.1),
            BatchNormalization(),
            Dropout(self.dropout_rate),
            
            # Third LSTM layer
            LSTM(self.lstm_units // 4, 
                 return_sequences=False,
                 recurrent_dropout=0.1),
            BatchNormalization(),
            Dropout(self.dropout_rate),
            
            # Dense layers with residual connection concept
            Dense(50, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            
            Dense(25, activation='relu'),
            Dropout(0.1),
            
            Dense(1, activation='linear')
        ])
        
        # Use advanced optimizer settings
        optimizer = Adam(
            learning_rate=0.001,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7
        )
        
        model.compile(
            optimizer=optimizer,
            loss='huber',  # More robust than MSE
            metrics=['mae', 'mse']
        )
        
        self.model = model
        print("✓ Advanced model built successfully!")
        print(f"✓ Total parameters: {model.count_params():,}")
        
        return model
    
    def train_model(self, epochs=200, batch_size=32, verbose=1):
        """
        Train with advanced callbacks and techniques
        """
        if self.model is None:
            self.build_advanced_model()
        
        print(f"🚀 Training advanced model for up to {epochs} epochs...")
        
        # Advanced callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=25,  # More patience for complex model
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=10,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        # Train with class weights if needed
        history = self.model.fit(
            self.X_train, self.y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(self.X_val, self.y_val),
            callbacks=callbacks,
            verbose=verbose,
            shuffle=False  # Important for time series
        )
        
        print("✅ Training completed!")
        return history
    
    def evaluate_comprehensive(self):
        """
        Comprehensive evaluation with multiple metrics
        """
        print("📊 Comprehensive model evaluation...")
        
        # Predictions
        train_pred_scaled = self.model.predict(self.X_train, verbose=0)
        val_pred_scaled = self.model.predict(self.X_val, verbose=0)
        test_pred_scaled = self.model.predict(self.X_test, verbose=0)
        
        # Inverse transform predictions
        train_pred = self.price_scaler.inverse_transform(train_pred_scaled.reshape(-1, 1))
        val_pred = self.price_scaler.inverse_transform(val_pred_scaled.reshape(-1, 1))
        test_pred = self.price_scaler.inverse_transform(test_pred_scaled.reshape(-1, 1))
        
        # Inverse transform actual values
        train_actual = self.price_scaler.inverse_transform(self.y_train.reshape(-1, 1))
        val_actual = self.price_scaler.inverse_transform(self.y_val.reshape(-1, 1))
        test_actual = self.price_scaler.inverse_transform(self.y_test.reshape(-1, 1))
        
        # Calculate comprehensive metrics
        def calculate_all_metrics(actual, pred, name):
            rmse = np.sqrt(mean_squared_error(actual, pred))
            mae = mean_absolute_error(actual, pred)
            mape = np.mean(np.abs((actual - pred) / actual)) * 100
            
            # Direction accuracy (very important for trading)
            actual_direction = np.diff(actual.flatten()) > 0
            pred_direction = np.diff(pred.flatten()) > 0
            direction_accuracy = np.mean(actual_direction == pred_direction) * 100
            
            return rmse, mae, mape, direction_accuracy
        
        train_rmse, train_mae, train_mape, train_dir = calculate_all_metrics(train_actual, train_pred, "Train")
        val_rmse, val_mae, val_mape, val_dir = calculate_all_metrics(val_actual, val_pred, "Val")
        test_rmse, test_mae, test_mape, test_dir = calculate_all_metrics(test_actual, test_pred, "Test")
        
        print("\n" + "="*70)
        print("📊 COMPREHENSIVE MODEL PERFORMANCE")
        print("="*70)
        print(f"{'Metric':<12} {'Train':<12} {'Validation':<12} {'Test':<12}")
        print("-" * 70)
        print(f"{'RMSE':<12} {train_rmse:<12.6f} {val_rmse:<12.6f} {test_rmse:<12.6f}")
        print(f"{'MAE':<12} {train_mae:<12.6f} {val_mae:<12.6f} {test_mae:<12.6f}")
        print(f"{'MAPE (%)':<12} {train_mape:<12.2f} {val_mape:<12.2f} {test_mape:<12.2f}")
        print(f"{'Direction':<12} {train_dir:<12.1f}% {val_dir:<12.1f}% {test_dir:<12.1f}%")
        print("="*70)
        
        # Trading insights
        print(f"\n💡 Trading Performance Insights:")
        if test_dir > 55:
            print(f"   ✅ Excellent direction accuracy: {test_dir:.1f}%")
        elif test_dir > 50:
            print(f"   ⚠️  Moderate direction accuracy: {test_dir:.1f}%")
        else:
            print(f"   ❌ Poor direction accuracy: {test_dir:.1f}% (worse than random)")
            
        if test_rmse < 0.005:
            print(f"   ✅ Excellent price accuracy: {test_rmse:.6f}")
        elif test_rmse < 0.01:
            print(f"   ⚠️  Good price accuracy: {test_rmse:.6f}")
        else:
            print(f"   ❌ Poor price accuracy: {test_rmse:.6f}")
        
        return {
            'metrics': {
                'test_rmse': test_rmse, 'test_mae': test_mae, 
                'test_mape': test_mape, 'test_direction': test_dir
            },
            'predictions': {
                'test_pred': test_pred, 'test_actual': test_actual,
                'val_pred': val_pred, 'val_actual': val_actual
            }
        }
    
    def plot_advanced_results(self, results, history):
        """
        Advanced plotting with multiple insights
        """
        fig, axes = plt.subplots(3, 2, figsize=(16, 12))
        
        # Plot 1: Training history
        axes[0, 0].plot(history.history['loss'], label='Training Loss', alpha=0.8)
        axes[0, 0].plot(history.history['val_loss'], label='Validation Loss', alpha=0.8)
        axes[0, 0].set_title('Training Progress')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Plot 2: Test predictions vs actual
        test_actual = results['predictions']['test_actual'].flatten()
        test_pred = results['predictions']['test_pred'].flatten()
        
        axes[0, 1].plot(test_actual, label='Actual', alpha=0.8)
        axes[0, 1].plot(test_pred, label='Predicted', alpha=0.8)
        axes[0, 1].set_title('Test Set: Actual vs Predicted')
        axes[0, 1].set_xlabel('Time')
        axes[0, 1].set_ylabel('USD per EUR')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Plot 3: Scatter plot
        axes[1, 0].scatter(test_actual, test_pred, alpha=0.6)
        axes[1, 0].plot([test_actual.min(), test_actual.max()], 
                       [test_actual.min(), test_actual.max()], 'r--', lw=2)
        axes[1, 0].set_xlabel('Actual')
        axes[1, 0].set_ylabel('Predicted')
        axes[1, 0].set_title('Prediction Accuracy Scatter')
        axes[1, 0].grid(True)
        
        # Plot 4: Prediction errors
        errors = test_actual - test_pred
        axes[1, 1].hist(errors, bins=20, alpha=0.7, edgecolor='black')
        axes[1, 1].axvline(x=0, color='red', linestyle='--')
        axes[1, 1].set_xlabel('Prediction Error')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Error Distribution')
        axes[1, 1].grid(True)
        
        # Plot 5: Direction accuracy over time
        actual_direction = np.diff(test_actual) > 0
        pred_direction = np.diff(test_pred) > 0
        correct_direction = actual_direction == pred_direction
        
        # Rolling direction accuracy
        window = 10
        rolling_accuracy = pd.Series(correct_direction.astype(int)).rolling(window).mean() * 100
        
        axes[2, 0].plot(rolling_accuracy, label=f'{window}-day Rolling Direction Accuracy')
        axes[2, 0].axhline(y=50, color='red', linestyle='--', alpha=0.7, label='Random (50%)')
        axes[2, 0].set_xlabel('Time')
        axes[2, 0].set_ylabel('Direction Accuracy (%)')
        axes[2, 0].set_title('Direction Prediction Accuracy Over Time')
        axes[2, 0].legend()
        axes[2, 0].grid(True)
        
        # Plot 6: Feature importance proxy (gradients)
        # This is a simplified version - in practice you'd use more sophisticated methods
        sample_input = self.X_test[:1]
        with tf.GradientTape() as tape:
            tape.watch(sample_input)
            prediction = self.model(sample_input)
        
        gradients = tape.gradient(prediction, sample_input)
        feature_importance = np.mean(np.abs(gradients.numpy()), axis=(0, 1))
        
        # Feature names (simplified)
        feature_names = ['Price', 'SMA5', 'SMA10', 'SMA20', 'EMA5', 'EMA10', 
                        'Mom5', 'Mom10', 'Vol5', 'Vol10', 'RSI', 'BB', 
                        'vs_SMA5', 'vs_SMA20', 'Ret1d', 'Ret5d'][:len(feature_importance)]
        
        axes[2, 1].bar(range(len(feature_importance)), feature_importance)
        axes[2, 1].set_xlabel('Features')
        axes[2, 1].set_ylabel('Importance')
        axes[2, 1].set_title('Feature Importance (Gradient-based)')
        axes[2, 1].set_xticks(range(len(feature_names)))
        axes[2, 1].set_xticklabels(feature_names, rotation=45)
        axes[2, 1].grid(True)
        
        plt.tight_layout()
        plt.show()

def main():
    """
    Run the improved LSTM model
    """
    print("🎯 IMPROVED USD per EUR LSTM Prediction Model")
    print("   (With Technical Indicators & Advanced Architecture)")
    print("="*70)
    
    # Initialize improved model
    model = ImprovedForexLSTM(
        sequence_length=60,  # Longer sequences for better patterns
        lstm_units=100,      # More complex model
        dropout_rate=0.3     # Higher dropout to prevent overfitting
    )
    
    # Load and prepare data
    file_path = 'usd_eur_data.csv'  # Change to your file
    try:
        df = model.load_and_prepare_data(file_path)
    except:
        print("Using sample data for demonstration...")
        df = model.create_sample_data_with_features()
    
    # Prepare sequences
    model.prepare_sequences()
    
    # Train the improved model
    history = model.train_model(
        epochs=200,
        batch_size=16,  # Smaller batch for more stable training
        verbose=1
    )
    
    # Comprehensive evaluation
    results = model.evaluate_comprehensive()
    
    # Advanced plotting
    model.plot_advanced_results(results, history)
    
    print("\n✅ Advanced LSTM analysis complete!")
    print("💡 If results are still poor, consider:")
    print("   • More external data (economic indicators, news sentiment)")
    print("   • Ensemble methods (combining multiple models)")
    print("   • Different prediction targets (direction instead of price)")
    print("   • Shorter prediction horizons (next day vs next week)")

if __name__ == "__main__":
    main()