import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os

# -------------------------------
# CONFIGURACIÓN VISUAL PROFESIONAL / PROFESSIONAL VISUAL CONFIGURATION
# -------------------------------
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (18, 14)
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['axes.labelsize'] = 13

def student_productivity_dashboard(file_path):
    """
    Carga el dataset, calcula estadísticas y genera un dashboard 2x2.
    """
    # Resolver la ruta absoluta en relación con el script
    if not os.path.isabs(file_path):
        file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), file_path)

    print(f"Cargando datos desde {file_path}...")
    
    # Manejo de errores (Error handling): Qué hacer si el archivo no existe
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"❌ Error: No se encontró el archivo CSV en la ruta: {file_path}")
        print("Asegúrate de que el archivo CSV esté en la misma carpeta que este script.")
        return
        
    df = df.dropna()

    # -------------------------------
    # 1️⃣ CORRELACIÓN CON PRODUCTIVITY / PRODUCTIVITY CORRELATION
    # -------------------------------
    # Identificar la columna de puntuación de productividad dinámicamente / Identify productivity score column dynamically
    possible_names = ['productivity_score', 'Productivity_Score', 'productivity']
    actual_productivity_col = next((col for col in df.columns if col.lower() in [p.lower() for p in possible_names]), df.columns[-1])
    
    print(f"Usando columna de productividad: {actual_productivity_col}")

    numeric_cols = df.select_dtypes(include=np.number).columns
    correlations = df[numeric_cols].corr()[actual_productivity_col].drop(actual_productivity_col)
    correlations = correlations.sort_values()

    # -------------------------------
    # 2️⃣ SEGMENTACIÓN PRODUCTIVIDAD / PRODUCTIVITY SEGMENTATION
    # -------------------------------
    # Cuartiles superior e inferior
    top_25 = df[df[actual_productivity_col] >= df[actual_productivity_col].quantile(0.75)]
    bottom_25 = df[df[actual_productivity_col] <= df[actual_productivity_col].quantile(0.25)]

    # Verificar el nombre de la columna de redes sociales / Check for social media column name
    social_media_col = 'social_media_hours' if 'social_media_hours' in df.columns else 'social_media_usage'

    lifestyle_vars = ['study_hours_per_day', 'sleep_hours', 'phone_usage_hours',
                      social_media_col, 'gaming_hours', 'exercise_minutes']

    # Filtrar solo las variables que realmente existen en el DataFrame
    lifestyle_vars = [var for var in lifestyle_vars if var in df.columns]

    top_means = top_25[lifestyle_vars].mean()
    bottom_means = bottom_25[lifestyle_vars].mean()

    # -------------------------------
    # 3️⃣ CREAR DASHBOARD / CREATE DASHBOARD
    # -------------------------------
    fig, axs = plt.subplots(2, 2)
    plt.subplots_adjust(hspace=0.3, wspace=0.25)

    # 🔹 GRÁFICO 1: Barras de Correlación
    colors = ['#e74c3c' if val < 0 else '#2ecc71' for val in correlations]
    axs[0, 0].barh(correlations.index, correlations.values, color=colors)
    axs[0, 0].axvline(0, color='black')
    axs[0, 0].set_title('Impacto de Hábitos en la Productividad')
    axs[0, 0].set_xlabel('Coeficiente de Correlación')

    # 🔹 GRÁFICO 2: Heatmap (Matriz de correlación)
    sns.heatmap(df[numeric_cols].corr(),
                cmap='coolwarm',
                center=0,
                ax=axs[0, 1])
    axs[0, 1].set_title('Mapa de Correlaciones')

    # 🔹 GRÁFICO 3: Scatter con línea de tendencia (Regresión)
    # Validamos que exista la columna de horas de estudio
    study_col = 'study_hours_per_day' if 'study_hours_per_day' in df.columns else numeric_cols[0]
    sns.regplot(data=df,
                x=study_col,
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

    plt.suptitle('Dashboard Analítico - Productividad Estudiantil y Distracción Digital',
                 fontsize=22, fontweight='bold')

    # Guardar y mostrar
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig("student_productivity_dashboard.png", dpi=300)
    plt.show()

    print("\n✅ Dashboard generado y guardado exitosamente como 'student_productivity_dashboard.png'.")

# -------------------------------
# PUNTO DE ENTRADA DEL SCRIPT (ENTRY POINT)
# -------------------------------
if __name__ == "__main__":
    # Nombre del archivo que tienes en tu repositorio
    nombre_archivo = "student_productivity_distraction_dataset_20000.csv"
    student_productivity_dashboard(nombre_archivo)