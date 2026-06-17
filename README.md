# EV Transition Intelligence Lab (ETIL)

**Tagline:** *A quantitative framework for modeling the dynamics, regimes, and future paths of global electric vehicle adoption.*

ETIL é um laboratório de pesquisa quantitativa para análise da transição energética no setor automotivo. Combina econometria, modelos de difusão, detecção de regimes, machine learning Bayesiano e simulações de Monte Carlo para responder perguntas como:
- Em qual estágio da transição cada país se encontra?
- Qual a velocidade de adoção?
- Qual a probabilidade de mudança de regime?
- Quando um mercado atinge maturidade?
- Como cenários alternativos alteram a trajetória?

## Estrutura
- `app.py`: aplicação Streamlit
- `ingestion/`: carga e validação de dados
- `preprocessing/`: limpeza, transformação e engenharia de features
- `models/`: modelos de difusão, regimes, previsão e score EVQTI
- `analytics/`: análises agregadas e ranking
- `visualization/`: gráficos e mapas
- `streamlit_pages/`: páginas do terminal quant

## Instalação
```bash
pip install -r requirements.txt