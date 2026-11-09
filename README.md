# DataForge
Система загрузки, очистки, преобразования и анализа наборов данных.

## Установка
pip install -r requirements.txt

## Быстрый старт
python main.py generate --rows 1000 --cols 6 --nulls 0.1 --dups 0.1 --out data/gen.csv/n  
python main.py inspect data/gen.csv  
python main.py validate data/gen.csv  
python main.py clean data/gen.csv --rules configs_clean.json --out processed/clean.csv  
python main.py analyze processed/clean.csv  
python main.py anomalies processed/clean.csv --column value_1 --method iqr  
python main.py benchmark  

## Тесты
pytest tests -v
