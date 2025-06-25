import pandas as pd
from pandasai import SmartDataframe
from pandasai.llm import OpenAI
import os
from dotenv import load_dotenv
import json

load_dotenv()
openai_key=os.getenv("OPENAI_API_KEY")

llm = OpenAI(model="gpt-3.5-turbo", temperature=0.1, api_token=openai_key)

data = pd.read_csv("population.csv")

smart_df = SmartDataframe(data, config={ "llm": llm, "enable_cache": False })
df_pandas = smart_df.dataframe

def chat_with_df_v2(df, question):
    """Versão mais robusta que lida com qualquer tipo de resposta"""
    try:
        response = df.chat(question)
        
        # Log para debug
        print(f"Tipo: {type(response).__name__}")
        print(f"Módulo: {type(response).__module__}")
        
        if response is None:
            return {"status": "error", "message": "Nenhuma resposta foi gerada."}
        
        # Converter resposta para formato serializável
        def serialize_response(obj):
            # Se for DataFrame-like (tem to_dict e columns)
            if hasattr(obj, 'to_dict') and hasattr(obj, 'columns'):
                return obj.to_dict(orient='records')
            
            # Se for Series-like (tem to_dict mas não columns)
            elif hasattr(obj, 'to_dict') and hasattr(obj, 'index'):
                return obj.to_dict()
            
            # Se for numpy array
            elif hasattr(obj, 'tolist'):
                return obj.tolist()
            
            # Se já for serializável
            elif isinstance(obj, (dict, list, str, int, float, bool)):
                return obj
            
            # Se tiver representação em DataFrame
            elif hasattr(obj, 'dataframe'):
                return obj.dataframe.to_dict(orient='records')
            
            # Último recurso: converter para string
            else:
                return str(obj)
        
        serialized_data = serialize_response(response)
        
        return {
            "status": "success",
            "data": serialized_data,
            "type": type(response).__name__
        }
        
    except Exception as e:
        import traceback
        return {
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc()
        }

# Usar a versão v2
result = chat_with_df_v2(smart_df, "Liste o top 5 países com maior população.")
print(json.dumps(result, indent=2, ensure_ascii=False))