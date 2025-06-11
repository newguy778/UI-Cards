import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
# Updated imports for TensorFlow 2.19+
try:
    # Try new import style for TF 2.16+
    from keras.models import Sequential
    from keras.layers import LSTM, Dense, Dropout
    from keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from keras.optimizers import Adam
    print("✓ Using new Keras imports (TF 2.19)")
except ImportError:
    # Fallback to old import style
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.optimizers import Adam
    print("✓ Using legacy Keras imports")
import warnings
warnings.filterwarnings('ignore')

# Set CPU optimization
tf.config.threading.set_intra_op_parallelism_threads(0)
tf.config.threading.set_inter_op_parallelism_threads(0)

class ForexLSTMPredictor:
    """
    LSTM model for predicting USD per EUR exchange rates
    Predicts how many US Dollars are needed to buy 1 Euro (DEXUSEU)
    """
    def __init__(self, sequence_length=30, lstm_units=50, dropout_rate=0.2):
        self.sequence_length = sequence_length
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None
        
    def load_data(self, file_path):
        """
        Load USD/EUR forex data from CSV file
        Expected format: observation_date, DEXUSEU
        """
        try:
            df = pd.read_csv(file_path)
            
            # Handle FRED USD per EUR data format
            if 'DEXUSEU' in df.columns and 'observation_date' in df.columns:
                print("✓ Detected FRED USD per EUR exchange rate format (Dollars to One Euro)")
                rate_column = 'DEXUSEU'
                date_column = 'observation_date'
            else:
                # Fallback to auto-detection
                rate_columns = ['DEXUSEU', 'Close', 'Rate', 'Price', 'Exchange_Rate', 'Value']
                rate_column = None
                
                for col in rate_columns:
                    if col in df.columns:
                        rate_column = col
                        break
                
                if rate_column is None:
                    rate_column = df.columns[1]
                    print(f"Using column '{rate_column}' as exchange rate")
                
                # Process date column
                date_columns = ['observation_date', 'Date', 'date', 'DATE', 'Time', 'Timestamp']
                date_column = None
                
                for col in date_columns:
                    if col in df.columns:
                        date_column = col
                        break
                
                if date_column is None:
                    date_column = df.columns[0]
                    print(f"Using column '{date_column}' as date")
            
            # Process the data
            df[date_column] = pd.to_datetime(df[date_column])
            df = df.sort_values(date_column)
            
            # Handle missing values (common in FRED data)
            df = df.dropna(subset=[rate_column])
            
            # Handle '.' values that might be in FRED data
            if df[rate_column].dtype == 'object':
                df = df[df[rate_column] != '.']
                df[rate_column] = pd.to_numeric(df[rate_column])
            
            self.data = df[rate_column].values.reshape(-1, 1)
            self.dates = df[date_column].values
            
            print(f"✓ Loaded {len(self.data)} data points")
            print(f"✓ Date range: {df[date_column].min()} to {df[date_column].max()}")
            print(f"✓ USD per EUR rate range: {df[rate_column].min():.4f} to {df[rate_column].max():.4f}")
            print(f"    (${df[rate_column].min():.4f} - ${df[rate_column].max():.4f} per 1 Euro)")
            
            # Data size warning for LSTM
            if len(self.data) < 200:
                print("⚠️  Warning: Limited data size. Consider using simpler models for small datasets.")
            
            return df
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            print("Please check your CSV format. Expected: observation_date, DEXUSEU")
            print("(DEXUSEU = US Dollars to One Euro)")
            print("Creating sample USD per EUR data instead...")
            return self.create_sample_data()
    
    def create_sample_data(self):
        """
        Create sample USD per EUR data for demonstration (2024 data)
        DEXUSEU = US Dollars to One Euro Spot Exchange Rate
        """
        np.random.seed(42)
        dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
        
        # Simulate realistic USD per EUR movements (typical range 1.05-1.15 in 2024)
        base_rate = 1.0800  # Starting around $1.08 per Euro
        returns = np.random.normal(0, 0.008, len(dates))  # Daily volatility ~0.8%
        
        # Add some trend and volatility clustering typical of USD per EUR
        for i in range(1, len(returns)):
            returns[i] += 0.05 * returns[i-1]  # Slight momentum
            
        # Add seasonal patterns (USD often stronger in Q4, so rate might decrease)
        seasonal = -0.01 * np.sin(2 * np.pi * np.arange(len(dates)) / 365)
        
        rates = [base_rate]
        for i, ret in enumerate(returns[1:]):
            new_rate = rates[-1] * (1 + ret + seasonal[i+1])
            # Keep within realistic bounds for 2024
            new_rate = max(1.00, min(1.20, new_rate))
            rates.append(new_rate)
        
        df = pd.DataFrame({
            'observation_date': dates,
            'DEXUSEU': rates
        })
        
        self.data = np.array(rates).reshape(-1, 1)
        self.dates = dates.values
        
        print(f"✓ Created sample USD per EUR data with {len(self.data)} points")
        print("  (Sample shows dollars needed to buy 1 Euro)")
        return df
    
    def prepare_data(self):
        """
        Prepare data for LSTM training with time-based splits
        Optimized for 2024 USD/EUR data
        """
        print("🔄 Preparing data...")
        
        # Adjust sequence length based on data size
        data_size = len(self.data)
        if data_size < 300:
            self.sequence_length = min(self.sequence_length, 20)
            print(f"🔧 Adjusted sequence length to {self.sequence_length} due to limited data")
        
        # Scale the data
        scaled_data = self.scaler.fit_transform(self.data)
        
        # Create sequences
        X, y = [], []
        for i in range(self.sequence_length, len(scaled_data)):
            X.append(scaled_data[i-self.sequence_length:i, 0])
            y.append(scaled_data[i, 0])
        
        X, y = np.array(X), np.array(y)
        X = X.reshape((X.shape[0], X.shape[1], 1))
        
        # Time-based split adjusted for 2024 data
        total_samples = len(X)
        
        if total_samples < 100:
            # For very small datasets, adjust splits
            train_size = int(total_samples * 0.7)
            val_size = int(total_samples * 0.2)
            print("⚠️  Small dataset detected - adjusted splits to 70/20/10")
        else:
            # Standard splits for reasonable data size
            train_size = int(total_samples * 0.75)  # 75% for training
            val_size = int(total_samples * 0.15)    # 15% for validation
            # 10% for testing
        
        self.X_train = X[:train_size]
        self.y_train = y[:train_size]
        self.X_val = X[train_size:train_size + val_size]
        self.y_val = y[train_size:train_size + val_size]
        self.X_test = X[train_size + val_size:]
        self.y_test = y[train_size + val_size:]
        
        # Store split indices for plotting
        self.train_end_idx = train_size + self.sequence_length
        self.val_end_idx = train_size + val_size + self.sequence_length
        
        print(f"✓ Training samples: {len(self.X_train)} (~{len(self.X_train)/total_samples*100:.1f}%)")
        print(f"✓ Validation samples: {len(self.X_val)} (~{len(self.X_val)/total_samples*100:.1f}%)")
        print(f"✓ Test samples: {len(self.X_test)} (~{len(self.X_test)/total_samples*100:.1f}%)")
        
        # Check if we have enough data
        if len(self.X_train) < 50:
            print("⚠️  Warning: Very limited training data. Consider collecting more data for better results.")
        
        return X, y
    
    def build_model(self):
        """
        Build optimized LSTM model for CPU training
        """
        print("🏗️ Building model...")
        
        model = Sequential([
            # First LSTM layer
            LSTM(self.lstm_units, 
                 return_sequences=True, 
                 input_shape=(self.sequence_length, 1),
                 recurrent_dropout=0.1),
            Dropout(self.dropout_rate),
            
            # Second LSTM layer
            LSTM(self.lstm_units // 2, 
                 return_sequences=False,
                 recurrent_dropout=0.1),
            Dropout(self.dropout_rate),
            
            # Dense layers
            Dense(25, activation='relu'),
            Dropout(0.1),
            Dense(1)
        ])
        
        # Use Adam with lower learning rate for stability
        try:
            optimizer = Adam(learning_rate=0.001)
        except NameError:
            optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        
        model.compile(
            optimizer=optimizer,
            loss='huber',  # More robust than MSE
            metrics=['mae']
        )
        
        self.model = model
        print("✓ Model built successfully!")
        
        return model
    
    def train_model(self, epochs=100, batch_size=32, verbose=1):
        """
        Train the LSTM model with callbacks
        """
        if self.model is None:
            self.build_model()
        
        print(f"🚀 Starting training for up to {epochs} epochs...")
        print(f"📊 Batch size: {batch_size}")
        
        # Callbacks for efficient training
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=8,
                min_lr=1e-6,
                verbose=1
            )
        ]
        
        # Train the model
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
    
    def predict(self, X):
        """Make predictions and inverse transform"""
        predictions = self.model.predict(X, verbose=0)
        return self.scaler.inverse_transform(predictions)
    
    def evaluate_model(self):
        """
        Comprehensive model evaluation
        """
        print("📈 Evaluating model...")
        
        # Make predictions
        train_pred = self.predict(self.X_train)
        val_pred = self.predict(self.X_val)
        test_pred = self.predict(self.X_test)
        
        # Inverse transform actual values
        y_train_actual = self.scaler.inverse_transform(self.y_train.reshape(-1, 1))
        y_val_actual = self.scaler.inverse_transform(self.y_val.reshape(-1, 1))
        y_test_actual = self.scaler.inverse_transform(self.y_test.reshape(-1, 1))
        
        # Calculate metrics
        def calculate_metrics(actual, pred, name):
            rmse = np.sqrt(mean_squared_error(actual, pred))
            mae = mean_absolute_error(actual, pred)
            mape = np.mean(np.abs((actual - pred) / actual)) * 100
            return rmse, mae, mape
        
        train_rmse, train_mae, train_mape = calculate_metrics(y_train_actual, train_pred, "Train")
        val_rmse, val_mae, val_mape = calculate_metrics(y_val_actual, val_pred, "Validation")
        test_rmse, test_mae, test_mape = calculate_metrics(y_test_actual, test_pred, "Test")
        
        print("\n" + "="*50)
        print("📊 MODEL PERFORMANCE RESULTS")
        print("="*50)
        print(f"Training   - RMSE: {train_rmse:.6f} | MAE: {train_mae:.6f} | MAPE: {train_mape:.2f}%")
        print(f"Validation - RMSE: {val_rmse:.6f} | MAE: {val_mae:.6f} | MAPE: {val_mape:.2f}%")
        print(f"Test       - RMSE: {test_rmse:.6f} | MAE: {test_mae:.6f} | MAPE: {test_mape:.2f}%")
        print("="*50)
        
        # Check for overfitting
        if val_rmse > train_rmse * 1.5:
            print("⚠️  Warning: Possible overfitting detected!")
        else:
            print("✅ Model shows good generalization")
        
        return {
            'metrics': {
                'train_rmse': train_rmse, 'val_rmse': val_rmse, 'test_rmse': test_rmse,
                'train_mae': train_mae, 'val_mae': val_mae, 'test_mae': test_mae,
                'train_mape': train_mape, 'val_mape': val_mape, 'test_mape': test_mape
            },
            'predictions': {
                'train_pred': train_pred, 'val_pred': val_pred, 'test_pred': test_pred,
                'train_actual': y_train_actual, 'val_actual': y_val_actual, 'test_actual': y_test_actual
            }
        }
    
    def plot_results(self, results, history=None):
        """
        Plot training history and predictions
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Training History
        if history:
            axes[0, 0].plot(history.history['loss'], label='Training Loss')
            axes[0, 0].plot(history.history['val_loss'], label='Validation Loss')
            axes[0, 0].set_title('Model Loss During Training')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True)
        
        # Plot 2: Full Time Series with Predictions
        full_actual = np.concatenate([
            results['predictions']['train_actual'],
            results['predictions']['val_actual'],
            results['predictions']['test_actual']
        ])
        
        full_pred = np.concatenate([
            results['predictions']['train_pred'],
            results['predictions']['val_pred'],
            results['predictions']['test_pred']
        ])
        
        # Create time indices
        start_idx = self.sequence_length
        time_indices = np.arange(start_idx, start_idx + len(full_actual))
        
        axes[0, 1].plot(time_indices, full_actual, label='Actual', alpha=0.7)
        axes[0, 1].plot(time_indices, full_pred, label='Predicted', alpha=0.7)
        
        # Add vertical lines for splits
        axes[0, 1].axvline(x=self.train_end_idx, color='red', linestyle='--', alpha=0.5, label='Train/Val Split')
        axes[0, 1].axvline(x=self.val_end_idx, color='orange', linestyle='--', alpha=0.5, label='Val/Test Split')
        
        axes[0, 1].set_title('USD per EUR Rate Prediction - Full Timeline')
        axes[0, 1].set_xlabel('Time Index')
        axes[0, 1].set_ylabel('USD per EUR Rate')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Plot 3: Test Set Zoom
        test_actual = results['predictions']['test_actual']
        test_pred = results['predictions']['test_pred']
        test_indices = np.arange(len(test_actual))
        
        axes[1, 0].plot(test_indices, test_actual, label='Actual', marker='o', markersize=3)
        axes[1, 0].plot(test_indices, test_pred, label='Predicted', marker='s', markersize=3)
        axes[1, 0].set_title('Test Set Predictions (Zoom)')
        axes[1, 0].set_xlabel('Test Sample')
        axes[1, 0].set_ylabel('USD per EUR Rate')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Plot 4: Prediction Error Distribution
        test_errors = test_actual.flatten() - test_pred.flatten()
        axes[1, 1].hist(test_errors, bins=20, alpha=0.7, edgecolor='black')
        axes[1, 1].set_title('Test Prediction Error Distribution')
        axes[1, 1].set_xlabel('Prediction Error')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].axvline(x=0, color='red', linestyle='--', alpha=0.7)
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def predict_future(self, days=7):
        """
        Predict future values
        """
        print(f"🔮 Predicting next {days} days...")
        
        # Use last sequence_length points for prediction
        last_sequence = self.data[-self.sequence_length:]
        last_sequence_scaled = self.scaler.transform(last_sequence)
        
        predictions = []
        current_sequence = last_sequence_scaled.reshape(1, self.sequence_length, 1)
        
        for _ in range(days):
            pred = self.model.predict(current_sequence, verbose=0)
            predictions.append(pred[0, 0])
            
            # Update sequence for next prediction
            current_sequence = np.roll(current_sequence, -1, axis=1)
            current_sequence[0, -1, 0] = pred[0, 0]
        
        # Inverse transform predictions
        predictions = np.array(predictions).reshape(-1, 1)
        future_predictions = self.scaler.inverse_transform(predictions)
        
        print(f"✓ Future {days}-day predictions completed")
        return future_predictions.flatten()

def main():
    """
    Main execution function for USD per EUR prediction
    """
    print("🎯 USD per EUR Exchange Rate LSTM Prediction Model")
    print("   (US Dollars to One Euro - DEXUSEU)")
    print("="*60)
    
    # Initialize model with parameters optimized for USD per EUR data
    predictor = ForexLSTMPredictor(
        sequence_length=25,  # 25 days lookback (about 1 month)
        lstm_units=40,       # Slightly smaller for 2024 data
        dropout_rate=0.2
    )
    
    # Load your USD per EUR data
    print("📁 Loading USD per EUR exchange rate data...")
    print("Expected format: observation_date, DEXUSEU")
    print("DEXUSEU = US Dollars needed to buy 1 Euro")
    
    # Replace 'usd_eur_data.csv' with your actual file path
    file_path = 'usd_eur_data.csv'  # Change this to your file path
    
    try:
        df = predictor.load_data(file_path)
    except FileNotFoundError:
        print(f"❌ File '{file_path}' not found!")
        print("📝 Please update the file_path variable with your CSV file location")
        print("🔄 Using sample USD per EUR data for demonstration...")
        df = predictor.create_sample_data()
    
    # Show data summary
    print(f"\n📊 Data Summary:")
    print(f"   • Total observations: {len(predictor.data)}")
    print(f"   • Date range: {predictor.dates[0]} to {predictor.dates[-1]}")
    print(f"   • USD per EUR range: ${predictor.data.min():.4f} - ${predictor.data.max():.4f}")
    print(f"   • Interpretation: Each unit = Dollars needed to buy 1 Euro")
    
    # Prepare data with adaptive splits
    predictor.prepare_data()
    
    # Train model with settings optimized for USD per EUR
    print(f"\n🚀 Training LSTM model for USD per EUR prediction...")
    history = predictor.train_model(
        epochs=80,       # More epochs for 2024 data
        batch_size=16,   # Smaller batch for limited data
        verbose=1
    )
    
    # Evaluate model performance
    results = predictor.evaluate_model()
    
    # Specific insights for USD per EUR
    test_rmse = results['metrics']['test_rmse']
    print(f"\n💡 USD per EUR Prediction Insights:")
    print(f"   • Test RMSE: ${test_rmse:.4f}")
    if test_rmse < 0.01:
        print("   • ✅ Excellent accuracy (< 1 cent per Euro)")
    elif test_rmse < 0.02:
        print("   • ⚠️  Good accuracy (1-2 cents error per Euro)")
    else:
        print("   • ❌ Poor accuracy (> 2 cents error per Euro) - consider more data")
    
    # Plot comprehensive results
    predictor.plot_results(results, history)
    
    # Future predictions for next week
    future_pred = predictor.predict_future(days=7)
    print(f"\n🔮 Next 7 days USD per EUR predictions:")
    for i, pred in enumerate(future_pred, 1):
        print(f"   Day {i}: ${pred:.4f} per 1 Euro")
    
    # Trading insights
    current_rate = predictor.data[-1][0]
    avg_future = np.mean(future_pred)
    
    if avg_future > current_rate:
        trend = "📈 USD Weakening (Euro getting more expensive)"
        interpretation = "Euro strengthening against Dollar"
    else:
        trend = "📉 USD Strengthening (Euro getting cheaper)"
        interpretation = "Dollar strengthening against Euro"
    
    change_pct = ((avg_future - current_rate) / current_rate) * 100
    
    print(f"\n📈 Trading Insights:")
    print(f"   • Current rate: ${current_rate:.4f} per 1 Euro")
    print(f"   • 7-day avg prediction: ${avg_future:.4f} per 1 Euro")
    print(f"   • Trend: {trend}")
    print(f"   • Change: {change_pct:+.2f}% ({interpretation})")
    print(f"   • Rate difference: ${avg_future - current_rate:+.4f} per Euro")
    
    print("\n✅ USD per EUR analysis complete!")
    print("💾 Save your model with: predictor.model.save('usd_per_eur_model.h5')")

if __name__ == "__main__":
    main()