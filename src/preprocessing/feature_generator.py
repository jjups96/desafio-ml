import pandas as pd

class FeatureGenerator:
    """Genera features basadas en datos pasados para forecasting."""

    def __init__(self, lags=3, rolling_window=3):
        """
        Inicializa el generador de features.

        Parameters:
        lags (int): Número de lags a generar.
        rolling_window (int): Tamaño de la ventana para la media móvil.
        """
        self.lags = lags
        self.rolling_window = rolling_window

    def add_lag_features(self, df):
        """
        Agrega columnas con ventas pasadas (lags).

        Parameters:
        df (DataFrame): DataFrame con los datos originales.

        Returns:
        DataFrame: DataFrame con las columnas de lags agregadas.
        """
        for lag in range(1, self.lags + 1):
            df[f"Sales_Lag_{lag}"] = df["Sales"].shift(lag)

        return df

    def add_rolling_features(self, df):
        """
        Agrega media móvil como feature.

        Parameters:
        df (DataFrame): DataFrame con los datos originales.

        Returns:
        DataFrame: DataFrame con la media móvil agregada.
        """
        df["Sales_Rolling_Mean"] = df["Sales"].shift(1).rolling(self.rolling_window).mean()
        return df

    def transform(self, df):
        """
        Transforma el DataFrame agregando las features de lags y media móvil.

        Parameters:
        df (DataFrame): DataFrame con los datos originales.

        Returns:
        DataFrame: DataFrame transformado con las nuevas features.
        """
        df = self.add_lag_features(df)
        df = self.add_rolling_features(df)
        return df.dropna().reset_index(drop=True)  # Drop NA después de generar lags