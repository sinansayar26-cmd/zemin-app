# Deprem Risk İzleme (Prototip)

Bu proje, USGS + AFAD deprem verileri ile dummy sensörlerden (GNSS/Strain/InSAR/Yeraltı suyu) üretilen basit özellikleri bir araya getirip `risk_model.py` içinde **kaba bir risk skoru** hesaplar ve Dash/Plotly ile bir dashboardta gösterir.

## Kurulum
```bash
python -m pip install dash plotly requests numpy