import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from data_preprocessing import load_and_clean_data, prepare_lstm_data


def build_and_train_lstm(X_train, y_train, epochs=50, batch_size=16):
    print("\n--- LSTM Modeli İnşa Ediliyor ---")
    model = Sequential()

    # 1. LSTM Katmani (64 unit - daha fazla ogrenim kapasitesi)
    model.add(LSTM(units=64, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(Dropout(0.2))

    # 2. LSTM Katmani (64 unit)
    model.add(LSTM(units=64, return_sequences=False))
    model.add(Dropout(0.2))

    # Cikis Katmani
    model.add(Dense(units=1))

    # Modeli Derle (mae: gercek yuzde hatasini takip etmek icin)
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

    # EarlyStopping: val_loss iyilesmedigi zaman en iyi agirliklara don
    # ReduceLROnPlateau: ogrenme hizi takilinca yavas yavas dusur
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=0),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=0),
    ]

    print("Model eğitimi başlıyor. Lütfen bekleyin...")
    # Egitimi Baslat (verinin %20'si dogrulama icin ayrildi)
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        callbacks=callbacks,
        verbose=0
    )

    best_epoch = len(history.history['loss'])
    best_val_loss = min(history.history['val_loss'])
    best_val_mae = min(history.history['val_mae'])
    print(f"Eğitim tamamlandı! {best_epoch} epoch çalıştı.")
    print(f"  En düşük val_loss (MSE) : {best_val_loss:.6f}")
    print(f"  En düşük val_mae       : {best_val_mae:.4f} (ölçekli birim)")

    return model


if __name__ == "__main__":
    veri_yolu = os.path.join(os.path.dirname(__file__), '..', 'data', 'bosna_hersek_cop_verisi_gercekci.xlsx')
    temiz_veri = load_and_clean_data(veri_yolu)
    X, y, scaler, islenmis_veri = prepare_lstm_data(temiz_veri, window_size=5)

    # Modeli Egit (EarlyStopping ile otomatik durur)
    egitilmis_model = build_and_train_lstm(X, y, epochs=50, batch_size=16)

    # Egitilen Modeli Kaydet
    model_kayit_yolu = os.path.join(os.path.dirname(__file__), '..', 'models', 'lstm_doluluk_modeli.keras')
    egitilmis_model.save(model_kayit_yolu)

    print(f"\nModel başarıyla eğitildi ve kaydedildi: {model_kayit_yolu} ✅")