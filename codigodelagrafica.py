import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os

# -------------------------------
# CONFIGURACIÓN VISUAL PROFESIONAL
# -------------------------------
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (18, 14)
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['axes.labelsize'] = 13

def student_productivity_dashboard(file_path):
    # Resolve absolute path relative to the script
    if not os.path.isabs(file_path):
        file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), file_path)

    print(f"Cargando datos desde {file_path}...")
    
    df = pd.read_csv(file_path)
    df = df.dropna()

    # -------------------------------
    # 1️⃣ CORRELACIÓN CON PRODUCTIVITY
    # -------------------------------
    # Identify productivity score column (it might be named differently)
    possible_names = ['productivity_score', 'Productivity_Score', 'productivity']
    actual_productivity_col = next((col for col in df.columns if col.lower() in [p.lower() for p in possible_names]), df.columns[-1])
    
    print(f"Usando columna de productividad: {actual_productivity_col}")

    numeric_cols = df.select_dtypes(include=np.number).columns
    correlations = df[numeric_cols].corr()[actual_productivity_col].drop(actual_productivity_col)
    correlations = correlations.sort_values()

    # -------------------------------
    # 2️⃣ SEGMENTACIÓN PRODUCTIVIDAD
    # -------------------------------
    top_25 = df[df[actual_productivity_col] >= df[actual_productivity_col].quantile(0.75)]
    bottom_25 = df[df[actual_productivity_col] <= df[actual_productivity_col].quantile(0.25)]

    # Check for social media column name
    social_media_col = 'social_media_hours' if 'social_media_hours' in df.columns else 'social_media_usage'

    lifestyle_vars = ['study_hours_per_day', 'sleep_hours', 'phone_usage_hours',
                      social_media_col, 'gaming_hours', 'exercise_minutes']

    top_means = top_25[lifestyle_vars].mean()
    bottom_means = bottom_25[lifestyle_vars].mean()

    # -------------------------------
    # CREAR DASHBOARD
    # -------------------------------
    fig, axs = plt.subplots(2, 2)
    plt.subplots_adjust(hspace=0.3, wspace=0.25)

    # 🔹 GRÁFICO 1: Barras de Correlación
    colors = ['#e74c3c' if val < 0 else '#2ecc71' for val in correlations]
    axs[0, 0].barh(correlations.index, correlations.values, color=colors)
    axs[0, 0].axvline(0, color='black')
    axs[0, 0].set_title('Impacto de Hábitos en la Productividad')
    axs[0, 0].set_xlabel('Coeficiente de Correlación')

    # 🔹 GRÁFICO 2: Heatmap
    sns.heatmap(df[numeric_cols].corr(),
                cmap='coolwarm',
                center=0,
                ax=axs[0, 1])
    axs[0, 1].set_title('Mapa de Correlaciones')

    # 🔹 GRÁFICO 3: Scatter con línea de tendencia
    sns.regplot(data=df,
                x='study_hours_per_day',
                y=actual_productivity_col,
                ax=axs[1, 0],
                scatter_kws={'alpha':0.3})
    axs[1, 0].set_title('Horas de Estudio vs Productividad')

    # 🔹 GRÁFICO 4: Comparación Alto vs Bajo Productividad
    x = np.arange(len(lifestyle_vars))
    width = 0.35

    axs[1, 1].bar(x - width/2, top_means, width, label='Top 25%')
    axs[1, 1].bar(x + width/2, bottom_means, width, label='Bottom 25%')

    axs[1, 1].set_xticks(x)
    axs[1, 1].set_xticklabels([v.replace('_', ' ').title() for v in lifestyle_vars], rotation=45)
    axs[1, 1].set_title('Comparación Hábitos: Alta vs Baja Productividad')
    axs[1, 1].legend()

    plt.suptitle('Dashboard Analítico - Student Productivity & Digital Distraction Dataset',
                 fontsize=22, fontweight='bold')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig("student_productivity_dashboard.png", dpi=300)
    plt.show()

    print("\nDashboard generado exitosamente.")

if __name__ == "__main__":
    student_productivity_dashboard("student_productivity_distraction_dataset_20000.csv")