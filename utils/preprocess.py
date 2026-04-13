import pandas as pd

def create_target(df):
    df['avg_score'] = (df['math score'] + df['reading score'] + df['writing score']) / 3

    def get_attention(score):
        if score > 80:
            return "High"
        elif score > 50:
            return "Medium"
        else:
            return "Low"

    df['Attention'] = df['avg_score'].apply(get_attention)
    return df