# Amazon Products and Networks Graph Explorer 


A Streamlit app generates amazon products summaries and helps find similar products and potential customers. The app is deployed
(https://amazonge.herokuapp.com/)

1. Business objective
----Customers conducted successive drill-down searches to finish investigations of products, business reputations. Time-consuming.
Business competitors are not visible and it is hard for them to “mine” relationships between each other.

2. Data ingestion
---The dataset can be downloaded at https://jmcauley.ucsd.edu/data/amazon/
Amazon Beauty ( ~ 1 GB, updated to 2018)
This dataset includes reviews (ratings, text, helpfulness votes), product metadata (descriptions, category information, price, brand, and image features), and links (also viewed/also bought graphs).
The data is largely wrangled using pandas, including select rows with nonempty brand, image url links, and concatenate all positive reviews and negative reviews respectively. 

3. Visualizations
----The data mainly contains texts. WordCloud is used to visualize the key information of good and bad reviews. 
Matplotlib, Pandas, and altair are major tools. 

4. A demonstration of at least one of the following:
---- Machine learning
NLTK, CountVectorizer is used to find similar products and acquire the similarity scores
---- An interactive website
The app is deployed and users can select the products they are interested to investigate.

5. A deliverable
---- A webpage is available.


