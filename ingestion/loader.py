# ingestion/loader.py (melhoria sugerida)

def _transform_iea_gevo(self, df: pd.DataFrame) -> pd.DataFrame:
    """Transformação robusta com filtros claros"""
    
    # 1. Filtrar apenas dados relevantes
    df = df[df['category'].isin(['Historical', 'Projection-STEPS'])]  # ou incluir CPS
    
    # 2. Pivotar para formato wide por tipo de veículo
    pivot = df.pivot_table(
        index=['region_country', 'year', 'category'],
        columns=['parameter', 'mode', 'powertrain'],
        values='value',
        aggfunc='sum'
    ).reset_index()
    
    # 3. Renomear colunas importantes
    pivot = pivot.rename(columns={
        ('EV sales', 'Cars', 'EV'): 'ev_sales_cars',
        ('EV stock', 'Cars', 'EV'): 'ev_stock_cars',
        ('EV sales', '2 and 3 wheelers', 'EV'): 'ev_sales_2w3w',
        # ... adicionar mais
    })
    
    # 4. Criar colunas consolidadas
    pivot['ev_sales_total'] = pivot.filter(like='ev_sales').sum(axis=1, skipna=True)
    pivot['total_sales_proxy'] = pivot['ev_sales_total'] * 4  # proxy temporário
    
    return pivot