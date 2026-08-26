"""
===============================================================================
MODELO DE INFERENCIA DE INGRESO — SALARIO_PROMEDIO_U6M
===============================================================================
Objetivo : estimar el salario promedio de los ultimos 6 meses de un cliente
           a partir de variables demograficas, laborales y de buro de credito.

Diseno   : XGBoost sobre log(salario) + target encoding suavizado fuera de
           muestra para variables de alta cardinalidad (EMPRESA, RUBRO) +
           validacion out-of-time (el ultimo mes de carga nunca se entrena) +
           correccion de retransformacion de Duan (smearing).

Uso      :
    python modelo_salario_u6m.py                # entrena y evalua
    ...luego, para puntear nuevos casos:
    from modelo_salario_u6m import cargar_modelo, predecir
    modelo = cargar_modelo("modelo_salario_u6m.joblib")
    df_nuevo["salario_estimado"] = predecir(modelo, df_nuevo)
===============================================================================
"""

from __future__ import annotations

import glob
import os
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

# =============================================================================
# 1. CONFIGURACION
# =============================================================================
CONFIG = {
    # Carpeta donde estan los archivos parte_*.xlsx
    "RUTA_DATOS": "/mnt/user-data/uploads",
    "PATRON": "parte_*.xlsx",
    "CACHE": "datos_cache.pkl",           # evita releer los xlsx en cada corrida

    "OBJETIVO": "SALARIO_PROMEDIO_U6M",
    "COL_FECHA": "FECHA_CARGA",           # AAAAMM
    "COL_ID": "LEGAL",

    # ¿El salario del mes corriente esta disponible al momento de puntear?
    #   True  -> modelo de "suavizado": se conoce SALARIO y se estima su
    #            promedio movil de 6 meses (muy preciso, uso limitado).
    #   False -> modelo de "inferencia de ingreso": el salario NO se observa
    #            (prospectos, no-planilla). Es el caso de uso real de riesgos.
    "USAR_SALARIO_ACTUAL": False,

    # Si USAR_SALARIO_ACTUAL = True, desestacionalizar antes de usarlo.
    # Sin esto el modelo aprende el catorceavo y sobreestima en junio.
    "DESESTACIONALIZAR_SALARIO": True,
    # "mediana"  : divide todo el mes por un solo factor (simple, pero asume
    #              que el bono es proporcional para todos).
    # "cuantiles": mapeo cuantil a cuantil contra la distribucion de meses
    #              normales. Corrige la forma de la distribucion, no solo el
    #              nivel; es lo que hay que usar si el bono no es parejo.
    "METODO_AJUSTE": "cuantiles",

    # Saneamiento del objetivo
    "SALARIO_MINIMO": 1_000.0,            # por debajo = registro basura
    "PODAR_COLA_SUPERIOR": 0.999,         # recorta el 0.1% superior al entrenar

    # Variables categoricas
    "CAT_ALTA_CARD": ["EMPRESA", "EMPRESA_INTERNA", "RUBRO"],  # target encoding
    "CAT_BAJA_CARD": ["ZONA", "GENERO"],                        # one-hot
    "SUAVIZADO_TE": 30.0,                 # k del encoder: a mayor k, mas prior
    "N_FOLDS_TE": 5,                      # folds para encoding fuera de muestra

    # GENERO es predictivo pero su uso en modelos que alimentan decisiones de
    # credito es un riesgo regulatorio/reputacional. Dejar en True para excluirlo.
    "EXCLUIR_VARIABLES_SENSIBLES": True,

    "SEMILLA": 42,
    "ARCHIVO_MODELO": "modelo_salario_u6m.joblib",
}

PARAMS_XGB = dict(
    n_estimators=3000,
    learning_rate=0.03,
    max_depth=7,
    min_child_weight=20,
    subsample=0.8,
    colsample_bytree=0.7,
    reg_lambda=2.0,
    reg_alpha=0.5,
    objective="reg:squarederror",
    eval_metric="rmse",
    early_stopping_rounds=100,
    n_jobs=-1,
    max_bin=128,          # menos memoria en el histograma
    tree_method="hist",
    random_state=CONFIG["SEMILLA"],
)


# =============================================================================
# 2. CARGA
# =============================================================================
def cargar_datos(ruta=None, patron=None, cache=None) -> pd.DataFrame:
    """Lee y concatena todos los parte_*.xlsx. Ignora archivos corruptos."""
    ruta = ruta or CONFIG["RUTA_DATOS"]
    patron = patron or CONFIG["PATRON"]
    cache = cache if cache is not None else CONFIG["CACHE"]

    if cache and os.path.exists(cache):
        print(f"[carga] usando cache {cache}")
        return pd.read_pickle(cache)

    archivos = sorted(glob.glob(os.path.join(ruta, patron)))
    if not archivos:
        raise FileNotFoundError(f"No hay archivos que cumplan {ruta}/{patron}")

    partes, fallidos = [], []
    for a in archivos:
        try:
            partes.append(pd.read_excel(a))
            print(f"[carga] {os.path.basename(a):<22} {partes[-1].shape}")
        except Exception as e:                      # xlsx invalido / corrupto
            fallidos.append((os.path.basename(a), type(e).__name__))

    if fallidos:
        print("[carga] ADVERTENCIA, archivos ilegibles:")
        for nombre, err in fallidos:
            print(f"        - {nombre} ({err})")

    df = pd.concat(partes, ignore_index=True)
    print(f"[carga] total: {df.shape[0]:,} filas x {df.shape[1]} columnas")
    if cache:
        df.to_pickle(cache)
    return df


# =============================================================================
# 3. LIMPIEZA
# =============================================================================
def limpiar(df: pd.DataFrame, entrenamiento: bool = True) -> pd.DataFrame:
    df = df.copy()
    obj = CONFIG["OBJETIVO"]

    # Texto: mayusculas, sin espacios dobles
    for c in CONFIG["CAT_ALTA_CARD"] + CONFIG["CAT_BAJA_CARD"]:
        if c in df.columns:
            df[c] = (df[c].astype("string").str.upper().str.strip()
                     .str.replace(r"\s+", " ", regex=True)
                     .fillna("DESCONOCIDO"))

    # Columnas 100% vacias
    vacias = [c for c in df.columns if df[c].isna().all()]
    if vacias:
        print(f"[limpieza] columnas 100% nulas eliminadas: {vacias}")
        df = df.drop(columns=vacias)

    # Edades imposibles -> NaN (XGBoost trata el NaN como categoria propia)
    if "EDAD" in df.columns:
        df.loc[(df["EDAD"] < 18) | (df["EDAD"] > 90), "EDAD"] = np.nan

    if entrenamiento and obj in df.columns:
        n0 = len(df)
        df = df[df[obj].notna() & (df[obj] >= CONFIG["SALARIO_MINIMO"])]
        tope = df[obj].quantile(CONFIG["PODAR_COLA_SUPERIOR"])
        df = df[df[obj] <= tope]
        print(f"[limpieza] {n0 - len(df):,} filas descartadas "
              f"(objetivo nulo, < {CONFIG['SALARIO_MINIMO']:,.0f} o > {tope:,.0f})")

    return df.reset_index(drop=True)


# =============================================================================
# 4. INGENIERIA DE VARIABLES
# =============================================================================
def ingenieria_features(df: pd.DataFrame) -> pd.DataFrame:
    """Variables derivadas del comportamiento crediticio. La capacidad de pago
    revelada (cuotas que el cliente sostiene) es el mejor proxy de ingreso
    cuando no se observa la planilla."""
    df = df.copy()

    productos = {
        "tc": "cantidad_tc_vigentes",
        "ptmo": "cantidad_ptmos_vigentes",
        "auto": "cantidad_auto_vigente",
        "hipoteca": "cantidad_hipotecas_vigentes",
        "comercial": "cantidad_deucomer_vigentes",
    }

    # Banderas de tenencia: el patron de productos separa segmentos de ingreso
    for nombre, col in productos.items():
        if col in df.columns:
            df[f"tiene_{nombre}"] = (df[col].fillna(0) > 0).astype(int)

    cols_tenencia = [f"tiene_{n}" for n in productos if f"tiene_{n}" in df.columns]
    if cols_tenencia:
        df["num_tipos_producto"] = df[cols_tenencia].sum(axis=1)

    cols_cant = [c for c in productos.values() if c in df.columns]
    if cols_cant:
        df["total_productos_vigentes"] = df[cols_cant].fillna(0).sum(axis=1)

    # Cuota total y su reparto: un cliente hipotecario gana distinto que uno
    # con la misma cuota concentrada en tarjetas
    if "cuota_total_global" in df.columns:
        ct = df["cuota_total_global"].fillna(0)
        for nombre, col in [("tc", "cuota_total_tc"), ("ptmo", "cuota_total_ptmo"),
                            ("auto", "cuota_total_auto"),
                            ("hipoteca", "cuota_total_hipoteca"),
                            ("comercial", "cuota_total_comercial")]:
            if col in df.columns:
                df[f"prop_cuota_{nombre}"] = np.where(ct > 0, df[col].fillna(0) / ct, np.nan)
        df["log_cuota_total"] = np.log1p(ct)

    # Limite de tarjeta: el banco ya asigno un cupo en funcion de un ingreso
    if "limite_maxima_tc" in df.columns:
        df["log_limite_max_tc"] = np.log1p(df["limite_maxima_tc"].fillna(0))
    if {"cuota_total_tc", "limite_maxima_tc"}.issubset(df.columns):
        df["ratio_cuota_limite_tc"] = np.where(
            df["limite_maxima_tc"].fillna(0) > 0,
            df["cuota_total_tc"].fillna(0) / df["limite_maxima_tc"], np.nan)

    # Antiguedad crediticia maxima (meses en el sistema financiero)
    cols_ant = [c for c in df.columns if c.startswith("antiguedad_max_")]
    if cols_ant:
        df["antiguedad_credito_max"] = df[cols_ant].max(axis=1)

    # Estacionalidad: en Honduras el 13o y 14o mes mueven el promedio movil
    if CONFIG["COL_FECHA"] in df.columns:
        f = df[CONFIG["COL_FECHA"]].astype(int)
        df["anio"] = f // 100
        df["mes"] = f % 100

    if "EDAD" in df.columns:
        df["edad2"] = df["EDAD"] ** 2          # el ingreso es concavo en la edad

    return df


# =============================================================================
# 4b. DIAGNOSTICO DE ESTACIONALIDAD  (revisar ANTES de modelar)
# =============================================================================
def diagnostico_estacionalidad(df: pd.DataFrame) -> pd.DataFrame:
    """Compara el salario del mes contra el promedio movil de 6 meses.

    En Honduras el 13o (diciembre) y el 14o (junio) inflan el salario del mes
    sin que el ingreso recurrente cambie. Si el ratio se dispara en algun mes,
    SALARIO NO puede usarse crudo como predictor: el modelo aprende el bono y
    sobreestima el ingreso justo en los meses de mayor originacion.
    """
    obj, colf = CONFIG["OBJETIVO"], CONFIG["COL_FECHA"]
    if "SALARIO" not in df.columns:
        return pd.DataFrame()
    t = df.groupby(colf).agg(n=("SALARIO", "size"),
                             salario_mes=("SALARIO", "median"),
                             promedio_u6m=(obj, "median")).round(0)
    t["ratio"] = (t["salario_mes"] / t["promedio_u6m"]).round(3)
    print("\n--- ESTACIONALIDAD: salario del mes / promedio U6M ---")
    print(t.to_string())
    sospechosos = t.index[t["ratio"] > 1.25].tolist()
    if sospechosos:
        print(f"    ALERTA: meses con bono detectados -> {sospechosos}")
        print("    Recomendacion: mantener USAR_SALARIO_ACTUAL = False,")
        print("    o desestacionalizar SALARIO antes de usarlo.")
    return t


# =============================================================================
# 4c. DESESTACIONALIZACION DEL SALARIO DEL MES
# =============================================================================
def calcular_indice_estacional(df, meses_referencia=None, mediana_ref=None):
    """Indice de nivel salarial por mes de carga.

        indice_m = mediana(SALARIO del mes m) / mediana de referencia

    Solo usa la distribucion poblacional de SALARIO, NUNCA el objetivo, asi
    que es calculable al momento de puntear un lote. La referencia es la
    mediana de las medianas mensuales del periodo de entrenamiento: un mes
    con bono no la arrastra.
    """
    colf = CONFIG["COL_FECHA"]
    med_mes = df.groupby(colf)["SALARIO"].median()
    if mediana_ref is None:
        base = med_mes.loc[meses_referencia] if meses_referencia is not None else med_mes
        mediana_ref = float(base.median())
    indice = (med_mes / mediana_ref).clip(0.5, 3.0)
    return indice.to_dict(), mediana_ref


def ajustar_por_cuantiles(df, referencia_valores, meses_a_ajustar, n_puntos=1001):
    """Mapeo cuantil a cuantil.

    A cada cliente se le calcula su percentil DENTRO de su mes y se lo lleva
    al salario que corresponde a ese mismo percentil en los meses normales.
    Corrige que el catorceavo no sea parejo: quien entro hace poco o trabajo
    mes parcial no recibe bono completo, y un factor unico lo castiga.
    """
    colf = CONFIG["COL_FECHA"]
    df = df.copy()
    df["salario_ajustado"] = df["SALARIO"].astype(float)
    pp = np.linspace(0, 100, n_puntos)
    q_ref = np.percentile(referencia_valores, pp)

    for m in meses_a_ajustar:
        mask = df[colf] == m
        if mask.sum() < 500:
            continue
        v = df.loc[mask, "SALARIO"].astype(float).values
        q_mes = np.percentile(v, pp)
        pct = np.interp(v, q_mes, pp)              # percentil dentro del mes
        df.loc[mask, "salario_ajustado"] = np.interp(pct, pp, q_ref)
    return df


def aplicar_ajuste_estacional(df, indice, mediana_ref):
    """Agrega el salario desestacionalizado y la carga financiera sobre el."""
    df = df.copy()
    colf = CONFIG["COL_FECHA"]
    factor = df[colf].map(indice).astype(float).fillna(1.0)
    df["indice_mes"] = factor
    if "salario_ajustado" not in df.columns:
        df["salario_ajustado"] = df["SALARIO"] / factor
    df["log_salario_ajustado"] = np.log1p(df["salario_ajustado"])
    if "cuota_total_global" in df.columns:
        # Carga financiera revelada: cuanto del ingreso recurrente ya esta
        # comprometido. Es la variable que despues reusas en originacion.
        df["carga_financiera"] = (df["cuota_total_global"].fillna(0)
                                  / df["salario_ajustado"].replace(0, np.nan))
        df["carga_financiera"] = df["carga_financiera"].clip(0, 5)
    return df


# =============================================================================
# 5. TARGET ENCODING SUAVIZADO (fuera de muestra)
# =============================================================================
class TargetEncoderSuavizado:
    """Codifica una categoria por la media del objetivo, encogida hacia la
    media global segun el tamano del grupo:

        codigo_c = (n_c * media_c + k * media_global) / (n_c + k)

    En entrenamiento se calcula fuera de muestra (K-fold) para no filtrar el
    objetivo dentro de la misma fila.
    """

    def __init__(self, columnas, k=30.0, n_folds=5, semilla=42):
        self.columnas = columnas
        self.k = k
        self.n_folds = n_folds
        self.semilla = semilla
        self.mapas_, self.conteos_, self.prior_ = {}, {}, None

    def fit_transform(self, X, y):
        X = X.copy()
        self.prior_ = float(np.mean(y))
        kf = KFold(self.n_folds, shuffle=True, random_state=self.semilla)

        for col in self.columnas:
            if col not in X.columns:
                continue
            oof = np.full(len(X), np.nan)
            for idx_tr, idx_va in kf.split(X):
                mapa = self._ajustar_mapa(X[col].iloc[idx_tr], y[idx_tr])
                oof[idx_va] = X[col].iloc[idx_va].map(mapa).fillna(self.prior_).values
            # Mapa final (todos los datos) para aplicar a validacion/produccion
            self.mapas_[col] = self._ajustar_mapa(X[col], y)
            self.conteos_[col] = X[col].value_counts().to_dict()
            X[f"te_{col}"] = oof
            X[f"n_{col}"] = X[col].map(self.conteos_[col]).astype(float)

        return X.drop(columns=[c for c in self.columnas if c in X.columns])

    def transform(self, X):
        X = X.copy()
        for col in self.columnas:
            if col not in X.columns:
                continue
            X[f"te_{col}"] = X[col].map(self.mapas_[col]).fillna(self.prior_)
            X[f"n_{col}"] = X[col].map(self.conteos_[col]).fillna(0).astype(float)
        return X.drop(columns=[c for c in self.columnas if c in X.columns])

    def to_dict(self):
        return {"columnas": self.columnas, "k": self.k, "n_folds": self.n_folds,
                "semilla": self.semilla, "mapas": self.mapas_,
                "conteos": self.conteos_, "prior": self.prior_}

    @classmethod
    def from_dict(cls, d):
        enc = cls(d["columnas"], d["k"], d["n_folds"], d["semilla"])
        enc.mapas_, enc.conteos_, enc.prior_ = d["mapas"], d["conteos"], d["prior"]
        return enc

    def _ajustar_mapa(self, s, y):
        aux = pd.DataFrame({"cat": s.values, "y": np.asarray(y)})
        g = aux.groupby("cat")["y"].agg(["mean", "count"])
        suavizado = (g["count"] * g["mean"] + self.k * self.prior_) / (g["count"] + self.k)
        return suavizado.to_dict()


# =============================================================================
# 6. MATRIZ DE DISENO
# =============================================================================
def construir_matriz(df, encoder=None, y=None, columnas_ref=None):
    """Devuelve X lista para XGBoost. Si encoder es None, lo ajusta (train)."""
    obj = CONFIG["OBJETIVO"]
    excluir = {obj, CONFIG["COL_ID"], CONFIG["COL_FECHA"]}
    if not CONFIG["USAR_SALARIO_ACTUAL"]:
        excluir.update(["SALARIO", "salario_ajustado", "log_salario_ajustado",
                        "carga_financiera", "indice_mes"])
    elif CONFIG["DESESTACIONALIZAR_SALARIO"]:
        excluir.add("SALARIO")          # se usa solo la version ajustada
    if CONFIG["EXCLUIR_VARIABLES_SENSIBLES"]:
        excluir.add("GENERO")

    X = df.drop(columns=[c for c in excluir if c in df.columns])

    # One-hot para baja cardinalidad
    bajas = [c for c in CONFIG["CAT_BAJA_CARD"] if c in X.columns]
    if bajas:
        X = pd.get_dummies(X, columns=bajas, prefix=bajas, dtype=float)

    # Target encoding para alta cardinalidad
    if encoder is None:
        encoder = TargetEncoderSuavizado(
            [c for c in CONFIG["CAT_ALTA_CARD"] if c in X.columns],
            k=CONFIG["SUAVIZADO_TE"], n_folds=CONFIG["N_FOLDS_TE"],
            semilla=CONFIG["SEMILLA"])
        X = encoder.fit_transform(X, y)
    else:
        X = encoder.transform(X)

    # Cualquier texto residual fuera
    X = X.drop(columns=X.select_dtypes(include=["object", "string"]).columns)

    # Alinear columnas con el entrenamiento
    if columnas_ref is not None:
        X = X.reindex(columns=columnas_ref, fill_value=np.nan)

    return X.astype("float32"), encoder


# =============================================================================
# 7. METRICAS
# =============================================================================
def evaluar(y_real, y_pred, etiqueta=""):
    y_real, y_pred = np.asarray(y_real, float), np.asarray(y_pred, float)
    err = y_pred - y_real
    ape = np.abs(err) / np.maximum(y_real, 1)
    ss_res = np.sum(err ** 2)
    ss_tot = np.sum((y_real - y_real.mean()) ** 2)

    m = {
        "n": len(y_real),
        "MAE": np.mean(np.abs(err)),
        "MedAE": np.median(np.abs(err)),
        "RMSE": np.sqrt(np.mean(err ** 2)),
        "MAPE_%": 100 * np.mean(ape),
        "MdAPE_%": 100 * np.median(ape),
        "R2": 1 - ss_res / ss_tot,
        "R2_log": np.corrcoef(np.log1p(y_real), np.log1p(y_pred))[0, 1] ** 2,
        "dentro_±20%": 100 * np.mean(ape <= 0.20),
        "dentro_±30%": 100 * np.mean(ape <= 0.30),
        "sesgo_%": 100 * (y_pred.mean() / y_real.mean() - 1),
    }
    if etiqueta:
        print(f"\n--- {etiqueta} ---")
        for k, v in m.items():
            print(f"  {k:<14}: {v:,.2f}" if k != "n" else f"  {k:<14}: {v:,}")
    return m


def tabla_calibracion(y_real, y_pred, n_grupos=10):
    """Compara predicho vs real por decil de prediccion. Es lo que se lleva al
    comite: importa que no haya sesgo sistematico por nivel de ingreso."""
    d = pd.DataFrame({"real": y_real, "pred": y_pred})
    d["decil"] = pd.qcut(d["pred"], n_grupos, labels=False, duplicates="drop") + 1
    t = d.groupby("decil").agg(n=("real", "size"),
                               real_prom=("real", "mean"),
                               pred_prom=("pred", "mean"),
                               real_mediana=("real", "median"),
                               pred_mediana=("pred", "median")).round(0)
    t["sesgo_%"] = (100 * (t["pred_prom"] / t["real_prom"] - 1)).round(1)
    return t


# =============================================================================
# 8. ENTRENAMIENTO
# =============================================================================
def entrenar(df: pd.DataFrame) -> dict:
    obj, colf = CONFIG["OBJETIVO"], CONFIG["COL_FECHA"]

    # --- Particion out-of-time: el ultimo mes de carga no se entrena nunca ---
    periodos = sorted(df[colf].unique())
    mes_test = periodos[-1]
    mes_valid = periodos[-2]
    print(f"\n[particion] entrena: {periodos[:-2]}")
    print(f"[particion] valida (early stopping): {mes_valid}")
    print(f"[particion] prueba out-of-time     : {mes_test}")

    indice, mediana_ref = None, None
    if CONFIG["USAR_SALARIO_ACTUAL"] and CONFIG["DESESTACIONALIZAR_SALARIO"]:
        indice, mediana_ref = calcular_indice_estacional(df, meses_referencia=periodos[:-2])
        if CONFIG["METODO_AJUSTE"] == "cuantiles":
            # Meses normales del periodo de entrenamiento como referencia
            normales = [m for m in periodos[:-2] if indice[m] <= 1.15]
            ref_vals = df.loc[df[colf].isin(normales), "SALARIO"].dropna().values
            anormales = [m for m in periodos if indice[m] > 1.15]
            print(f"[estacional] referencia: {normales} | ajustados: {anormales}")
            df = ajustar_por_cuantiles(df, ref_vals, anormales)
        df = aplicar_ajuste_estacional(df, indice, mediana_ref)
        print(f"[estacional] mediana de referencia: {mediana_ref:,.0f} L")
        print("[estacional] indice por mes: "
              + ", ".join(f"{m}={v:.2f}" for m, v in indice.items()))

    df_tr = df[~df[colf].isin([mes_valid, mes_test])]
    df_va = df[df[colf] == mes_valid]
    df_te = df[df[colf] == mes_test]

    # --- log del objetivo: el ingreso es lognormal; sin log el modelo persigue
    #     la cola alta y se equivoca en el 90% de la cartera ---
    y_tr_log = np.log(df_tr[obj].values)
    y_va_log = np.log(df_va[obj].values)

    X_tr, enc = construir_matriz(df_tr, y=y_tr_log)
    X_va, _ = construir_matriz(df_va, encoder=enc, columnas_ref=X_tr.columns)
    X_te, _ = construir_matriz(df_te, encoder=enc, columnas_ref=X_tr.columns)
    print(f"[matriz] {X_tr.shape[1]} variables predictoras")

    modelo = XGBRegressor(**PARAMS_XGB)
    modelo.fit(X_tr, y_tr_log, eval_set=[(X_va, y_va_log)], verbose=False)
    print(f"[xgboost] mejor iteracion: {modelo.best_iteration}")

    # --- Correccion de Duan (smearing) ---------------------------------------
    # exp(E[log Y]) subestima E[Y]. El factor corrige el nivel promedio.
    resid = y_tr_log - modelo.predict(X_tr)
    smearing = float(np.mean(np.exp(resid)))
    print(f"[smearing] factor de retransformacion: {smearing:.4f}")

    pred = lambda X: np.exp(modelo.predict(X)) * smearing

    evaluar(df_tr[obj], pred(X_tr), "ENTRENAMIENTO")
    evaluar(df_va[obj], pred(X_va), f"VALIDACION ({mes_valid})")
    m_test = evaluar(df_te[obj], pred(X_te), f"PRUEBA OUT-OF-TIME ({mes_test})")

    # --- Referencia tonta: mediana del ingreso por empresa --------------------
    if "EMPRESA" in df.columns:
        med = df_tr.groupby("EMPRESA")[obj].median()
        base = df_te["EMPRESA"].map(med).fillna(df_tr[obj].median())
        evaluar(df_te[obj], base, "BASELINE (mediana por empresa)")

    print("\n--- CALIBRACION POR DECIL (out-of-time) ---")
    print(tabla_calibracion(df_te[obj].values, pred(X_te)).to_string())

    imp = (pd.Series(modelo.feature_importances_, index=X_tr.columns)
           .sort_values(ascending=False))
    print("\n--- TOP 20 VARIABLES (ganancia) ---")
    print((imp.head(20) * 100).round(2).to_string())

    artefacto = {"indice_estacional": indice, "mediana_ref": mediana_ref,
                 "modelo": modelo, "encoder": enc.to_dict(), "columnas": list(X_tr.columns),
                 "smearing": smearing, "config": CONFIG.copy(),
                 "metricas_oot": m_test, "importancias": imp}
    joblib.dump(artefacto, CONFIG["ARCHIVO_MODELO"])
    print(f"\n[guardado] {CONFIG['ARCHIVO_MODELO']}")
    return artefacto


# =============================================================================
# 9. PRODUCCION
# =============================================================================
def cargar_modelo(ruta=None):
    art = joblib.load(ruta or CONFIG["ARCHIVO_MODELO"])
    art["encoder"] = TargetEncoderSuavizado.from_dict(art["encoder"])
    return art


def predecir(artefacto, df_nuevo: pd.DataFrame) -> np.ndarray:
    """Estima el salario promedio U6M en lempiras para filas nuevas."""
    d = ingenieria_features(limpiar(df_nuevo, entrenamiento=False))
    if artefacto.get("indice_estacional"):
        d = aplicar_ajuste_estacional(d, artefacto["indice_estacional"],
                                      artefacto["mediana_ref"])
    X, _ = construir_matriz(d, encoder=artefacto["encoder"],
                            columnas_ref=artefacto["columnas"])
    return np.exp(artefacto["modelo"].predict(X)) * artefacto["smearing"]


# =============================================================================
def main():
    df = cargar_datos()
    df = limpiar(df)
    diagnostico_estacionalidad(df)
    df = ingenieria_features(df)
    return entrenar(df)


if __name__ == "__main__":
    main()
