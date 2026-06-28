import streamlit as st
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import io
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Dağılım Fitting", layout="wide")
st.title("📈 Dağılım Fitting Aracı")

# ── Dağılım fonksiyonları ──────────────────────────────────────────

def weibull(t, A, lam, k):
    return A * np.exp(-np.power(t / lam, k))

def exponential(t, A, b):
    return A * np.exp(-b * t)

def power(t, A, b):
    return A * np.power(t, b)

def inv_power(t, A, b):
    return A / np.power(t, b)

DISTS = {
    "Weibull":       {"func": weibull,      "params": ["A", "λ", "k"],
                      "p0": lambda x,y: [float(np.max(y)), float(np.median(x)), 1.5],
                      "bounds": ([1e-9,1e-9,0.05],[np.inf,np.inf,20])},
    "Exponential":   {"func": exponential,  "params": ["A", "b"],
                      "p0": lambda x,y: [float(np.max(y)), 0.05],
                      "bounds": ([1e-9,1e-9],[np.inf,np.inf])},
    "Power":         {"func": power,        "params": ["A", "b"],
                      "p0": lambda x,y: [float(y[0]) if y[0]>0 else 1.0, 1.0],
                      "bounds": ([1e-9,-20],[np.inf,20])},
    "Inverse Power": {"func": inv_power,    "params": ["A", "b"],
                      "p0": lambda x,y: [float(y[0]*x[0]), 1.0],
                      "bounds": ([1e-9,0.01],[np.inf,20])},
}

COLORS = {
    "Weibull": "#e34948",
    "Exponential": "#1baf7a",
    "Power": "#eda100",
    "Inverse Power": "#4a3aa7",
}

# ── Fitting fonksiyonu ─────────────────────────────────────────────

def fit(name, x, y):
    d = DISTS[name]
    starts = [d["p0"](x, y)]
    if name == "Weibull":
        starts += [[np.max(y)*0.5, np.median(x)*0.5, 0.8],
                   [np.max(y)*1.5, np.median(x)*1.5, 2.5]]
    if name == "Inverse Power":
        starts += [[y[0]*x[0]*0.5, 0.5], [y[0]*x[0]*2, 1.5]]

    best, best_sse = None, np.inf
    for s in starts:
        try:
            popt, _ = curve_fit(d["func"], x, y, p0=s,
                                bounds=d["bounds"], maxfev=10000, method="trf")
            sse = float(np.sum((y - d["func"](x, *popt))**2))
            if sse < best_sse:
                best_sse = sse
                best = popt
        except:
            continue

    if best is None:
        return None

    fitted = d["func"](x, *best)
    ss_res = np.sum((y - fitted)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r2   = float(max(0, 1 - ss_res/ss_tot))
    rmse = float(np.sqrt(np.mean((y - fitted)**2)))
    mae  = float(np.mean(np.abs(y - fitted)))
    n, k = len(y), len(best)
    aic  = float(n * np.log(ss_res/n + 1e-300) + 2*k)

    return {"params": best, "fitted": fitted,
            "r2": r2, "rmse": rmse, "mae": mae, "aic": aic}

# ── Sidebar ────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Ayarlar")

    uploaded = st.file_uploader("CSV veya Excel yükle", type=["csv","xlsx","xls"])

    st.divider()
    st.subheader("Filtreler")
    excl_below1 = st.checkbox("1'in altındaki değerleri dışla")

    use_range = st.checkbox("Dönem aralığı belirle")
    p_from, p_to = None, None
    if use_range:
        p_from = st.number_input("Başlangıç", min_value=0, value=1)
        p_to   = st.number_input("Bitiş",     min_value=0, value=100)

    use_extra = st.checkbox("Ekstra dönem tahmini")
    n_extra = 0
    if use_extra:
        n_extra = st.number_input("Kaç dönem?", min_value=1, max_value=200, value=5)

    st.divider()
    st.subheader("Dağılımlar")
    sel = {name: st.checkbox(name, value=True) for name in DISTS}

    go = st.button("Fit Et", type="primary", use_container_width=True)

# ── Ana alan ──────────────────────────────────────────────────────

if uploaded is None:
    st.info("Sol panelden dosya yükleyin.")
    st.stop()

# Dosya oku
try:
    if uploaded.name.endswith(".csv"):
        sample = uploaded.read(2048).decode("utf-8", errors="replace")
        uploaded.seek(0)
        sep = ";" if sample.count(";") > sample.count(",") else ","
        df_raw = pd.read_csv(uploaded, sep=sep)
    else:
        df_raw = pd.read_excel(uploaded)
except Exception as e:
    st.error(f"Dosya okunamadı: {e}")
    st.stop()

st.subheader("Yüklenen veri")
st.dataframe(df_raw.head(10), use_container_width=True)
st.caption(f"{len(df_raw)} satır · {len(df_raw.columns)} sütun")

# Sütun seçimi
cols = list(df_raw.columns)
c1, c2 = st.columns(2)
col_t = c1.selectbox("Dönem sütunu", cols, index=0)
col_y = c2.selectbox("Değer sütunu",  cols, index=min(1, len(cols)-1))

if not go:
    st.stop()

# Veri hazırlama
df = df_raw[[col_t, col_y]].copy()
df.columns = ["t", "y"]
df["t"] = pd.to_numeric(df["t"], errors="coerce")
df["y"] = pd.to_numeric(df["y"], errors="coerce")
df = df.dropna().sort_values("t").reset_index(drop=True)

if excl_below1:
    df = df[df["y"] >= 1].reset_index(drop=True)
if use_range and p_from is not None and p_to is not None:
    df = df[(df["t"] >= p_from) & (df["t"] <= p_to)].reset_index(drop=True)

if len(df) < 3:
    st.error("En az 3 satır gerekli.")
    st.stop()

x = df["t"].values.astype(float)
y = df["y"].values.astype(float)

step = float((x[-1]-x[0]) / max(len(x)-1, 1))
x_extra = np.array([x[-1]+(i+1)*step for i in range(n_extra)]) if n_extra > 0 else np.array([])

# Fitting
active = [name for name, checked in sel.items() if checked]
results = {}
with st.spinner("Hesaplanıyor..."):
    for name in active:
        r = fit(name, x, y)
        if r:
            results[name] = r
        else:
            st.warning(f"{name}: fit başarısız.")

if not results:
    st.error("Hiçbir dağılıma fit yapılamadı.")
    st.stop()

# ── Karşılaştırma tablosu ─────────────────────────────────────────

st.divider()
st.subheader("Karşılaştırma")

best_r2 = max(r["r2"] for r in results.values())
rows = []
for name, r in results.items():
    d = DISTS[name]
    pstr = "  |  ".join(f"{d['params'][i]}={r['params'][i]:.4g}"
                        for i in range(len(r["params"])))
    rows.append({
        "Dağılım": name,
        "R²":   round(r["r2"],   6),
        "RMSE": round(r["rmse"], 4),
        "MAE":  round(r["mae"],  4),
        "AIC":  round(r["aic"],  2),
        "Parametreler": pstr,
        "_best": r["r2"] >= best_r2 - 1e-9,
    })

df_comp = pd.DataFrame(rows)

def hl(row):
    if row["_best"]:
        return ["background-color:#eaf3de; font-weight:bold"] * len(row)
    return [""] * len(row)

show_cols = ["Dağılım","R²","RMSE","MAE","AIC","Parametreler"]
st.dataframe(
    df_comp[show_cols+["_best"]].style.apply(hl, axis=1)
        .format({"R²":"{:.6f}","RMSE":"{:.4f}","MAE":"{:.4f}","AIC":"{:.2f}"}),
    use_container_width=True, hide_index=True,
    column_config={"_best": None}
)

best_name = max(results, key=lambda k: results[k]["r2"])
st.success(f"En iyi fit: **{best_name}** — R² = {results[best_name]['r2']:.6f}")

# ── Grafik ────────────────────────────────────────────────────────

st.divider()
st.subheader("Grafik")

fig, ax = plt.subplots(figsize=(12, 5))
ax.scatter(x, y, color="#2a78d6", s=35, zorder=5, label="Gözlenen", alpha=0.85)

for name, r in results.items():
    ax.plot(x, r["fitted"], color=COLORS[name], linewidth=2,
            linestyle="--", label=f"{name} (R²={r['r2']:.4f})")
    if len(x_extra) > 0:
        y_ex = DISTS[name]["func"](x_extra, *r["params"])
        ax.plot(x_extra, y_ex, color=COLORS[name], linewidth=1.5,
                linestyle=":", alpha=0.7)

if len(x_extra) > 0:
    ax.axvline(x[-1], color="#aaa", linewidth=0.8, linestyle="--")
    ax.text(x[-1] + step*0.15, ax.get_ylim()[1]*0.95,
            "← veri  |  tahmin →", fontsize=8, color="#888")

ax.set_xlabel("Dönem")
ax.set_ylabel("Değer")
ax.legend(fontsize=9, framealpha=0.85)
ax.grid(True, alpha=0.2)
plt.tight_layout()
st.pyplot(fig)

# ── Değerler tablosu ──────────────────────────────────────────────

st.divider()
st.subheader("Fit edilmiş değerler")

all_t = np.concatenate([x, x_extra]) if len(x_extra) > 0 else x
df_out = pd.DataFrame({"Dönem": all_t})
df_out["Gözlenen"] = list(y) + [None]*len(x_extra)

for name, r in results.items():
    ex = list(DISTS[name]["func"](x_extra, *r["params"])) if len(x_extra)>0 else []
    df_out[f"Fit_{name}"]  = list(r["fitted"]) + ex
    df_out[f"Hata_{name}"] = list(y - r["fitted"]) + [None]*len(x_extra)

def hl2(row):
    if pd.isna(row["Gözlenen"]):
        return ["background-color:#e6f1fb"] * len(row)
    return [""] * len(row)

float_cols = [c for c in df_out.columns if c != "Dönem"]
st.dataframe(
    df_out.style.apply(hl2, axis=1)
        .format({c: "{:.4f}" for c in float_cols}, na_rep="—"),
    use_container_width=True, hide_index=True, height=300
)

# ── İndirme ───────────────────────────────────────────────────────

st.divider()
c1, c2 = st.columns(2)

csv_buf = io.StringIO()
df_out.to_csv(csv_buf, index=False, float_format="%.6f")
c1.download_button("⬇️ CSV indir", csv_buf.getvalue().encode(),
                   "fitting_sonuclari.csv", "text/csv", use_container_width=True)

xl_buf = io.BytesIO()
with pd.ExcelWriter(xl_buf, engine="openpyxl") as w:
    df_out.to_excel(w, sheet_name="Fit Değerleri", index=False)
    df_comp[show_cols].to_excel(w, sheet_name="Karşılaştırma", index=False)
c2.download_button("⬇️ Excel indir", xl_buf.getvalue(),
                   "fitting_sonuclari.xlsx",
                   "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                   use_container_width=True)
