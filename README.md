bezawit-A
## Data Science Portfolio 


## Project 1 - Impact of COVID-19 Data Analysis 


The purpose of this project is to investigate the impacts of COVID-19 in different parts of
the world. Essentially we are going to be analyzing COVID-19 data provided by John Hopkins
University. The data set contains daily level information on the number of COVID-19-affected
cases across the globe. Countries that are least economically developed countries have been
impacted the most due to COVID-19. This paper focuses on driving measurable insight using the
data to formulate which countries were impacted the most and what other factors such as climate
change and population density had a direct impact on the spread of the virus in those counties.
By aggregating the total number of cases for each country per day and graphing each result we
were able to deduce that most countries had exponential growth in confirmed cases over time.
Countries such as the US, Italy, Russia, Brazil, France, Italy, UK, and Brazil had the highest
reported number of confirmed cases and deaths. However, when we compute the ratio between
the number of confirmed cases vs the population size of each country the countries that had a
higher number of deaths and cases were not the same countries that had the highest ratio. The
countries with the highest ratios were Andorra, Denmark, Iceland, San Marino, and Israel. In
addition from our results, we are able to observe that countries that are least economically
developed such as Peru, Yemen, Sudan, and Mexico had the highest observed case fatality
ratio. The CFR measures the severity of the epidemic between countries showing the ones with
the highest numbers to have been impacted the most. Another hypothesis we tested was whether
warm weather slowed down the spread of the virus in those countries. According to the
correlation coefficient computed between the average weather per month vs the number of
confirmed cases for each month in 2021, the results did not indicate a negative correlation as the
coefficient was around 0.001. When we visualize the graph, however, the lines indicate there is a
correlation between the climate and number of cases. The paper will discuss factors that may
have resulted in higher-income countries having higher cases but will also evaluate how the
impact is measured using case fatality ratio, life expectancy, and demography of countries.


### Data Acquisition
The primary data used for this paper was extracted from the John Hopkins University
obtained from the course website. They collected daily data on COVID-19 cases from over 1.7
million people around the world. Downloading the data as CSV, I was able to extract it as a
python Dataframe using pandas, where I was able to manipulate, group, process and clean the
dataset.The data contained a list of 198 countries' reported cases(confirmed, recovered, and
deaths) for 816 reported days from 2020-01 to 2022-04. The data required some organizing and
cleaning as there were some records that were listed as countries that were not that were removed
using pandas drop methods to delete rows from a Dataframe. The first step to organizing the data 
was creating a daily single time series for each country, this required using the Groupby and sum
functions in pandas to get the aggregate value of daily cases for each country.
The second data set used was the World population dataset which included data on the
population size of each country. I merged the population data with the single time series
confirmed cases dataset using the country name as the key. However, since some countries had
different names in the datasets I changed the name of the countries to match the keys.
The third dataset I used was a weather dataset which was extracted from Kaggle (Kaggle
2012) where they collected the daily temperature of major cities around the world. The data was
downloaded as a CSV and manipulated using the sum and filter functions on excel to get the
aggregate monthly average temperature of a few selected countries( US, UK, and Italy). The file
CSV file was then extracted onto python and was merged to relevant datasets using pandas.

### Data Analysis 

The table below merges the dataset from John Hopkins University obtained from the
course web into a single time series for each country. 

<img width="357" alt="Screen Shot 2023-02-01 at 9 25 08 PM" src="https://user-images.githubusercontent.com/123434797/216218281-93971f3b-552b-47e8-95d1-76467876907f.png">

Below is the graph of the number of confirmed cases over time for each country:

<img width="684" alt="Screen Shot 2023-02-01 at 9 45 55 PM" src="https://user-images.githubusercontent.com/123434797/216221850-3f7b66da-84d6-4e10-9d7b-7f6a9c0f9831.png">


According to the graphs above most of the countries seem to be showing an exponential
growth. For example countries like Syria, Cuba, Burundi, Georgia, Austria, Belgium, Botswana
and Bolivia are leaving the exponential growth because we can see the line flatten towards April
2022.Countries like UK, Cyprus, France, Finland Greece and Ireland however are still showing
an exponential growth as there lines do not flatten, but still is at its peak in the month of April
2022. The graph below shows the number of deaths per 100 confirmed cases, for the 20 most
affected countries:

<img width="408" alt="Screen Shot 2023-02-01 at 10 10 19 PM" src="https://user-images.githubusercontent.com/123434797/216222193-4c57b9b1-ecd9-4193-a90b-ed8fc9b651d1.png">

From the bar graph above we see that Yemen had the highest number of deaths per 100
confirmed cases. Yemen is one of the poorest countries in the region spanning the Middle east to
North Africa (Haleem 2022). From this bar chart we can also observe that countries like Haiti,
Somalia, Niger, and Sudan are all lower income countries and have the highest number of deaths
per 100 confirmed cases.

The following table shows the ratio between the total number of confirmed cases and the
population size of each country:

<img width="171" alt="Screen Shot 2023-02-02 at 2 44 00 PM" src="https://user-images.githubusercontent.com/123434797/216434050-6e36d679-b03f-4d99-a3fd-40b9c3012499.png">

From the table above we can observe that none of the countries that had the highest
number of cases are listed to have the highest number of confirmed cases per population size.
This means the countries that had the highest number of cases did not have a lot of their
population impacted when comparing the cases to their population size. However it is important
to note such countries like Denmark, Iceland have a smaller population size compare to countries
like US and Germany.

The graph below shows the confirmed cases vs the average monthly weather for USA,
Italy and UK.


