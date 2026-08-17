### Data quality observations

- La mayoria de datos faltantes se tratan de las casas de apuesta, muchos de estos se deben por el hecho de que no todos participan en las mismas temporadas
- El identificador es compuesto por `Season`, `Div`, `Date`, `Time`, `HomeTeam` y `AwayTeam` donde la Season, HomeTeam y AwayTeam son las variables mas probables a afectar en los resultados
- No se encuentran valores negativos dentro de las validaciones realizadas
- Se encuentra una incongruencia dentro de los tiros y tiros hacia el arco, en un partido lo que no es logico, pero despues de una busqueda del partido se encontraron los datos originales.
- Se han encontrado 454 partidos que han resultados en empate
- `HomeTeam` y `AwayTeam` tienen la misma cardinalidad pero por el hecho de que contamos con 27 equipos
- Hay centros de apuestas que faltan un 80% de sus datos, por lo que se esta planteando no tomarlos tanto en cuenta como los centros de 1XB






def match_outcomes_summary(data, group_cols, target_col):
    """
    Resume la cantidad de partidos y el desglose de resultados (H, D, A) 
    para cualquier agrupación de variables.
    """
    if isinstance(group_cols, str):
        group_cols = [group_cols]
    
    # 1. Agrupamos y contamos las frecuencias de cada clase (H, D, A)
    summary = (
        data.groupby(group_cols, observed=True)[target_col]
        .value_counts()
        .unstack(fill_value=0) # Convierte las filas de clases en columnas
    )
    
    # 2. Calculamos el total de partidos para ese grupo
    summary["Total"] = summary.sum(axis=1)
    
    # 3. Calculamos los porcentajes dinámicamente para cada clase existente
    # Usamos las clases reales encontradas en la tabla pivote
    classes = [col for col in summary.columns if col != "Total"]
    
    for col in classes:
        summary[f"{col}_pct"] = (summary[col] / summary["Total"] * 100).round(2)
        
    return summary

# --- APLICACIÓN ---

# Desglose del resultado final (FTR) por Temporada
season_ftr_summary = match_outcomes_summary(premier_df, "Season", "FTR")

season_htr_summary = match_outcomes_summary(premier_df, "HTR", "FTR")

referee_summary = match_outcomes_summary(premier_df, "Referee", "FTR")

nombre_casa = match_outcomes_summary(premier_df, "HomeTeam", "FTR")
nombre_visitante = match_outcomes_summary(premier_df, "AwayTeam", "FTR")
division_completo = match_outcomes_summary(premier_df, "Div", "FTR")


display(season_ftr_summary)
display(season_htr_summary)
display(referee_summary)
display(nombre_casa)
display(nombre_visitante)
display(division_completo)
