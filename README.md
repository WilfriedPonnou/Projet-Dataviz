# Projet Dataviz Immo Data | Wilfried Ponnou
## Project Goal:
Make effecient vizualisations in order to explain these dataset of real estate transactions
## Vizualisations you will retreive in this app:
### Some of the vizualization include interactivity that is why streamlit_echarts and pyecharts have been used for. There are slicers, input, chechbox you can playt with don't hesitate to do so, and explore all the options!
1) Evolution of sales during one year(bar chart): in order to see trends of when people tend to buy or sale their real estate
and conclude when you should also buy or sell your real estate.
2) Histogram of  the nature of mutations: in order to know what type of selling is the best has the highest trend.
3) Pie chart of the type of real estate which have been sold: in order to see what type of real estate sells the best and in which you should invest.
4) Histogram of the best selling natures culture: in order for people to know in which culture they should invest.
5) Bar chart of the departments which sells the highest number of real estate: in order to know where you should invest. Or buy, in order to sell after for example.
6) Project: mean price by square meters according to: type of real estate, city, number of rooms.
## Libraries used:
Matplotlib, Numpy, Pandas, Folium, streamlit_echarts, pyecharts.
## Datasets have been loaded thanks to Jacques Tellier's website. I have tried to do it with Amazon S3 unsuccessfully, because the streamlit share application constantly crashed.
# !!!! Be conscious of the fact that a sampling has been used in order to speed up loading times, comment lines 166 to 169 if you want to work with the whole dataset in local and uncomment line 171!!! 
