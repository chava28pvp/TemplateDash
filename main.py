import pandas as pd


def estimate_memory(df: pd.DataFrame, overhead_factor: float = 1.5):
    """
    Estima el uso de memoria de un DataFrame de pandas en MB.

    Parámetros:
        df : pd.DataFrame
            El DataFrame a analizar.
        overhead_factor : float
            Factor de sobrecosto por copias temporales, índices,
            gráficos y caché (1.5 = 50% extra).

    Retorna:
        dict con memoria_base_MB y memoria_estimado_MB
    """
    # memoria base medida directamente
    mem_bytes = df.memory_usage(deep=True).sum()
    mem_mb = mem_bytes / 1024 ** 2

    # memoria estimada con overhead
    mem_est = mem_mb * overhead_factor

    return {
        "memoria_base_MB": round(mem_mb, 2),
        "memoria_estimado_MB": round(mem_est, 2),
        "factor_overhead": overhead_factor
    }


# Ejemplo de uso
if __name__ == "__main__":
    import numpy as np

    # dataset de prueba (1M filas, 10 numéricas y 2 de texto)
    n = 1_000_000
    df = pd.DataFrame({
        f"num_{i}": np.random.rand(n) for i in range(10)
    })
    df["texto1"] = ["cadena_de_prueba"] * n
    df["texto2"] = ["otro_texto"] * n

    result = estimate_memory(df)
    print(result)
