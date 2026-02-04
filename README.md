# Ice-company-Video-games-analysis-by-Julian-De-La-Garza-Lepe
Bootcamp Tripleten sprint #6 

 Análisis de Datos de Videojuegos - Proyecto Integrado

## Descripción del Proyecto
Análisis de datos de ventas de videojuegos para la tienda online Ice. El objetivo es identificar patrones que determinen el éxito de un videojuego y planificar campañas publicitarias efectivas para 2017.

## Objetivos del Análisis
- Identificar patrones de éxito en videojuegos
- Analizar ventas por región, plataforma y género
- Realizar pruebas de hipótesis estadísticas
- Crear perfiles de usuarios por región
- Detectar proyectos prometedores

## Dataset
- **Fuente:** Datos históricos de ventas de videojuegos hasta 2016
- **Archivo:** `games.csv`
- **Variables:** Plataforma, año, género, ventas por región, puntuaciones de críticos y usuarios, clasificación ESRB

## Metodología
1. **Preparación de datos:** Limpieza, conversión de tipos, manejo de valores ausentes
2. **Análisis exploratorio:** Distribuciones, tendencias temporales, análisis por categorías
3. **Análisis estadístico:** Cálculo de medidas de tendencia central y dispersión
4. **Pruebas de hipótesis:** Comparación de medias entre grupos
5. **Perfiles de usuario:** Análisis por región (NA, EU, JP)

## Tecnologías Utilizadas
- Python
- Pandas
- NumPy
- Matplotlib (plotpy)
- SciPy (para pruebas estadísticas)

## Principales Hallazgos

- **Ciclo de vida de plataformas:** Las plataformas tienen un ciclo de vida de 5-10 años, con PS4 y Xbox One dominando las ventas recientes (2013-2016)

- **Diferencias regionales significativas:** 
  - Norteamérica prefiere juegos de Acción y Shooter
  - Europa sigue patrones similares a NA pero con menor volumen
  - Japón tiene preferencias únicas hacia RPG y juegos de Nintendo

- **Impacto de las puntuaciones:** Las puntuaciones de críticos muestran mayor correlación con las ventas globales que las puntuaciones de usuarios

- **Géneros más rentables:** Action, Sports y Shooter generan las mayores ventas totales y en potencial de mercado se mantienen igual  a nivel mundial

- **Clasificación ESRB:** Los juegos clasificados como "M" (Mature) y "E" (Everyone) tienen el mejor desempeño comercial

- **Estacionalidad:** Los lanzamientos en Q4 (octubre-diciembre) muestran mejores resultados de ventas

## Estructura del Proyecto
```
├── README.md
├── ICE company videogames industry project S6.py
├── ICE company videogames industry project S6.ipynb
├── games.csv
├── requirements.txt
└── graficos/
```
## Autor
[Julián De La Garza Lepe]

