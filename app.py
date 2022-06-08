import streamlit as st
import pandas as pd
import altair as alt
from urllib.error import URLError
from PIL import Image
from wordcloud import WordCloud, STOPWORDS
import plotly.express as px 
import numpy as np
import matplotlib.pyplot as plt
import streamlit.components.v1 as components
import codecs
import webbrowser
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import altair as alt


## Load Data for Web Page 1

@st.cache
def load_dataframe():
    return pd.read_csv('data/data_page1_withimages.csv')

def make_wordcloud_plot(text,flag):

    wc= WordCloud(background_color = 'white',colormap = 'viridis',
    max_words = 150, stopwords = STOPWORDS,
    ).generate(text)
    fig, ax = plt.subplots(figsize = (24, 16))
    ax.imshow(wc, interpolation = 'bilinear')

    plt.axis('off')
    if flag == 1:
        txt="Positive Reviews"
    elif flag == 0:
        txt = "Negtive Reviews"
    fig.text(0.15, 0.9, txt, fontsize= 50)

    return fig




def app():
    df = load_dataframe()
    df1 = df.set_index("title",drop = False)

    ## Setup two view modes on the sidebar
    with st.sidebar:
        select_mode = st.sidebar.radio('View Mode 🔎', ('View Products', 'View People'))

    if select_mode == 'View Products':    
        st.title("Amazon Products Graph Explorer")
        titles = st.multiselect(
                "Choose products", list(df1.index), ["Beauty Without Cruelty Herbal Cream Facial Cleanser, 8.5 Ounces"])

        if len(titles)>= 1:
            data = df1.loc[titles,"star_1":"star_5"]
            ## d and dff are for horizontal bar chart
            d = {'Rating':["star_1","star_2","star_3","star_4","star_5"],'Reviews': data.loc[titles[-1],:].T }
            dff = pd.DataFrame(data = d)

            l1 = data.loc[titles[-1],:].T.to_list()
            l2 = [1,2,3,4,5]
            avgR = sum([l1[i] * l2[i] for i in range(len(l1))])*1.0/sum(l1)
            avgR = round(avgR,2)

            st.write("#### Avg Score of the LAST Product You Select is:    ",str(avgR), data.tail(1))

            image_url = df1['imageURLHighRes'][titles[-1]]
            desc = df1['description'][titles[-1]]
            price_ = df1['price'][titles[-1]]
            ## col1: product images col2 : product price and descriptions
            col1, mid, col2 = st.columns([1,1,10])
            with col1:   
                st.image(
                       image_url,
                        width=100, 
                    )            

            with col2:
                if len(str(price_)) < 20:
                    st.write("###### Price: ",price_ )
                if desc != '[]':
                    st.write('Description: ',desc)

            ##  create horizontal bar chart               
            chart = alt.Chart(dff).mark_bar().encode(
                    alt.X('Reviews'), 
                    alt.Y('Rating',sort=alt.EncodingSortField(field="Rating",order='descending')),color='Rating:O',tooltip= ['Rating','Reviews']).configure_axis(grid=False)

            st.altair_chart(chart, use_container_width=True) 

            pos = df1
            pos = df1['pos_reviews'][titles[-1]]
            neg = df1['neg_reviews'][titles[-1]]


            col3, col4 = st.columns(2)
            with col3:
                
                if str(pos) != 'nan':
                    fig_pos = make_wordcloud_plot(pos,1)
                    st.pyplot(fig_pos)  
                else:
                    st.write('### No Good Reviews Left')
                    
            with col4:
                
                if str(neg)!='nan':           
                    fig_neg = make_wordcloud_plot(neg,0)
                    st.pyplot(fig_neg)   
                else:
                    st.write('### No Bad Reviews Left')

    if select_mode == 'View People':
        df2 = pd.read_csv("data/findsimilar.csv")
        df3 = pd.read_csv("data/data_page2.csv")

        sort_S_1= np.load('data/sort_S_1.npy')
        S = np.load('data/S.npy')
        from spacy.lang.en.stop_words import STOP_WORDS

        STOP_WORDS = STOP_WORDS.difference({'he','his','her','hers'})
        STOP_WORDS = STOP_WORDS.union({'ll', 've'})

        st.title("Amazon Networks Explorer")    
        titles = st.multiselect(
                "Choose products", list(df2.title), ["12 Classic Color Elegant Blush Set"])  

        def find_similar(prod):
            prod_index = df2.loc[df2['title'] == prod].index[0]
            similar_index = sort_S_1[prod_index][1:]

            result = pd.DataFrame({
            'title':df2['title'][similar_index],
            'Brand': df2['brand'][similar_index],
            'SaleRank': df2['rank'][similar_index],
            'Similarity':S[prod_index,1:],
        })

            blankIndex = ['']*len(result)
            result.index = blankIndex

            return result

        def find_customers(prod):

            intersection = find_similar(prod)
            dff = df3.merge(intersection,how = 'inner',on = 'title')
            N = len(dff)
            M = min(100,N)
            result = dff['reviewerName'].sample(n= M, random_state=42)
            # result = np.array(temp)

            blankIndex = ['']*len(result)
            result.index = blankIndex

            return result

        if len(titles):
            st.write("Similar Products of the LAST Product You Select are:")
            st.write(find_similar(titles[-1]))
            input_ = find_customers(titles[-1]).tolist()    
            list2text = ', '.join(map(str, input_))
            st.text_area('Up to 100 Potential Customers For You:', value= list2text, height=180, max_chars=None, key=None)



if __name__ == '__main__':
    app()
