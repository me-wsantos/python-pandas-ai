import pandas as pd
from pandasai import SmartDataframe
from pandasai.llm import OpenAI

llm = OpenAI(model="gpt-3.5-turbo", temperature=0.1, api_token="YOUR_API_KEY")

# Criar DataFrame
df = pd.DataFrame({
    "vendas": [100, 200, 300, 400, 500],
    "mês": ["Jan", "Fev", "Mar", "Abr", "Mai"]
})

# Criar SmartDataframe
sdf = SmartDataframe(df, config={"llm": llm})

# Usar
resposta = sdf.chat("Qual o mês com a pior venda?")
print(resposta)