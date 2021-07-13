import pandas as pd
import numpy as np
import streamlit as st
from streamlit_pandas_profiling import st_profile_report
import seaborn as sns
import plotly.express as px
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn import mixture
from sklearn import metrics
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
plt.style.use('seaborn')

#Widgets libraries
import ipywidgets as wd
from IPython.display import display, clear_output
from ipywidgets import interactive

#Data Analysis Library
from pandas_profiling import ProfileReport

st.title('Rock Type Analysis')

st.text('Data analysis for computing the Rock Type using unsupervised learning techniques')

st.title('Dataset Report')

well_log=pd.read_csv('C:/Users/cheta/OneDrive - Georgia Institute of Technology/Desktop/Sound QI/Log Data.csv')

#Creating a profile report

report=ProfileReport(well_log, title='Profiling Reoprt')

if st.checkbox('Preview Profile Report'):
    st_profile_report(report)

#Data Cleaning
# k-NN to determine the missing Gamma Ray values
from sklearn.impute import KNNImputer
df=well_log.loc[:,['GammaRay(API)','ShaleVolume','Resistivity','Sonic']]
imputer=KNNImputer(n_neighbors=2,weights='uniform')
well_log['GammaRay_imputed']=imputer.fit_transform(df)[:,0]

#Outliers or Incorrect values
#Density
well_log.Density.replace(0, method = 'ffill', inplace = True )

#Depth
well_log['Depth(m)'] = np.where(well_log['Depth(m)']>3000 , np.nan, well_log['Depth(m)'])
transformed_depth = well_log['Depth(m)'].interpolate(method = 'linear')
well_log['Depth(m)_imputed'] = transformed_depth

# Now creating a df for analysis by only using the relevent columns

df=well_log.drop(columns=['NeutronPorosity','GammaRay(API)','Depth(m)'])

#Rock type Analysis
# Preprocessing the dataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn import metrics
from sklearn.metrics import silhouette_score

X=df
scale = StandardScaler()
x_scaled = scale.fit_transform(X)
well_logs_scaled = pd.DataFrame(x_scaled)

# Selecting the method for analysis   
st.title('Analysis Method')

Select_Method=st.selectbox('Select an Analysis Method', ('KMean','GMM'))

#Cluster selection

#For GMM
def SelBest(arr:list, X:int)->list:
    '''
    returns the set of X configurations with shorter distance
    '''
    dx=np.argsort(arr)[:X]
    return arr[dx]

def cluster_analysis(ana):
    
    if (ana=='KMean'):
        inertias = []
        clusters_list = range(1,11)
        # Compute Inertia value for different number of cluster
        for num_clusters in clusters_list:
            # Create a KMeans instance with k clusters: model
            model=KMeans(n_clusters=num_clusters , random_state=123)
            # Fit model to samples
            model.fit(well_logs_scaled)
            # Append the inertia to the list of inertias
            inertias.append(model.inertia_)
        #Visualizing the cluster for KMeans
        fig=px.line(x=clusters_list, y=inertias).update_traces(mode='lines+markers')
        fig.update_layout(showlegend=True, height=500, width=1000,title_text='Elbow Plot')

        # Update axis properties
        fig.update_yaxes(title_text='Inertia')
        fig.update_xaxes(title_text='N-Cluster')

        # Plot!
        st.text('Elbow method is used to choose the appropriate number of clusters for KMeans')
        st.plotly_chart(fig, use_container_width=True)
        #return clusters_list, inertias

    else:
        n_clusters=np.arange(2, 10)
        sils=[]
        sils_err=[]
        iterations=10
        for n in n_clusters:
            tmp_sil=[]
            for _ in range(iterations):
                gmm=GaussianMixture(n, n_init=2).fit(well_logs_scaled) 
                labels=gmm.predict(well_logs_scaled)
                sil=metrics.silhouette_score(well_logs_scaled, labels, metric='euclidean')
                #print("finished sil")
            tmp_sil.append(sil)
            val=np.mean(SelBest(np.array(tmp_sil), int(iterations/5)))
            err=np.std(tmp_sil)
            sils.append(val)
            sils_err.append(err)

        #Visualizing the cluster for KMeans
        fig1 = px.line(x=n_clusters,y=sils).update_traces(mode='lines+markers')
        fig1.update_layout(showlegend=True, height=500, width=1000,title_text='Silhouette Plot')

        # Update axis properties
        fig1.update_yaxes(title_text='Score')
        fig1.update_xaxes(title_text='N-Cluster')

        # Plot!
        st.text('Silhouette score checks how much the clusters are compact and well separated')
        st.plotly_chart(fig1, use_container_width=True)
        #return n_clusters,sils,sils_err
    
st.title('Clusters Analysis for Selected Method')
cluster_analysis(Select_Method)

st.title('Selecting Clusters')

value = st.slider('Select a cluster value',1, 10)
#st.write('Number of Clusters:', value)

#Now we can do the analysis for both KMean and GMM

st.title('Projecting Features for {} method'.format(Select_Method))

well_logs_scaled_embedded = TSNE(n_components=2,learning_rate=200,random_state=10,perplexity=50).fit_transform(well_logs_scaled) #t-SNE

def plot_cluster(ana):

    if (ana=='KMean'):
        # k-means implementation with 3 clusters
        kmeans = KMeans(n_clusters=value)
        kmeans.fit(well_logs_scaled)
        labels_rocks = kmeans.predict(well_logs_scaled)
        rocktypes = pd.DataFrame({'RockType':labels_rocks})
        well_log['KMean'] = rocktypes.RockType

        # Projecting the well log features into 2d projection using t-SNE

        fig2=px.scatter(well_logs_scaled_embedded,x=0, y=1, color=well_log.KMean, labels={'color': 'KMean'})
        fig2.update_layout(showlegend=True, height=500, width=1000,title_text='t-SNE 2D Projection')
        fig2.update_layout(showlegend=True,height=500, width=1000)

        # Plot!
        st.plotly_chart(fig2, use_container_width=True)

    else:
        gmm=GaussianMixture(n_components=value)
        gmm.fit(well_logs_scaled)
        labels_rocks1 = gmm.predict(well_logs_scaled)
        rocktypes1 = pd.DataFrame({'RockType':labels_rocks1})
        well_log['GMM'] = rocktypes1.RockType

        # Projecting the well log features into 2d projection using t-SNE

        fig3=px.scatter(well_logs_scaled_embedded,x=0, y=1, color=well_log.GMM, labels={'color': 'GMM'})
        fig3.update_layout(showlegend=True, height=500, width=1000,title_text='t-SNE 2D Projection of well logs')
        fig3.update_layout(showlegend=True,height=500, width=1000)

        # Plot!
        st.plotly_chart(fig3, use_container_width=True)

    

plot_cluster(Select_Method)

#Finally Plotting the data

st.title('Displaying the Well-Logs')

def analysis(ana):
    
    if (ana=='KMean'):
        # k-means implementation with 3 clusters
        kmeans = KMeans(n_clusters=value)
        kmeans.fit(well_logs_scaled)
        labels_rocks = kmeans.predict(well_logs_scaled)
        rocktypes = pd.DataFrame({'RockType':labels_rocks})
        well_log['KMean'] = rocktypes.RockType
        return well_log

    else:
        gmm=GaussianMixture(n_components=value)
        gmm.fit(well_logs_scaled)
        labels_rocks1 = gmm.predict(well_logs_scaled)
        rocktypes1 = pd.DataFrame({'RockType':labels_rocks1})
        well_log['GMM'] = rocktypes1.RockType
        return well_log

df_new=analysis(Select_Method)

# Display the well logs
list_KMean = ['ShaleVolume', 'Resistivity', 'GammaRay_imputed','Density','KMean']
list_GMM= ['ShaleVolume', 'Resistivity', 'GammaRay_imputed','Density','GMM']

# Display the well logs

def well_log_display(top_depth,bottom_depth,df,list_columns):
    
    #section of the log to plot
    sec=df[(df['Depth(m)_imputed']>=top_depth) & (df['Depth(m)_imputed']<=bottom_depth)]
    
    fig5=make_subplots(rows=1,cols=len(list_columns), shared_yaxes=True)
    
    for i in range(len(list_columns)):
        if (list_columns[i]!='KMean') & (list_columns[i]!='Resistivity') & (list_columns[i]!='GMM'):
            fig5.append_trace(go.Scatter(x=sec[list_columns[i]], y=sec['Depth(m)_imputed'], name=list_columns[i]), row=1, col=i+1)
        
            # Update axis properties
            fig5.update_xaxes(title_text=list_columns[i], row=1, col=i+1)
            
        elif (list_columns[i]=='Resistivity'):
            fig5.append_trace(go.Scatter(x=sec[list_columns[i]], y=sec['Depth(m)_imputed'],name=list_columns[i]), row=1, col=i+1)
        
            # Update axis properties
            fig5.update_xaxes(title_text=list_columns[i], row=1, col=i+1, type='log', range=[-1,.1])
            
        elif (list_columns[i]=='KMean'):
            fig5.append_trace(go.Bar(x=sec[list_columns[i]]+2, y=sec['Depth(m)_imputed'],name=list_columns[i],orientation='h', marker=dict(color=sec[list_columns[i]], coloraxis="coloraxis"),marker_coloraxis=None), row=1, col=i+1)
            
            # Update axis properties
            fig5.update_xaxes(title_text='KMean',range=[0,max(sec['KMean'])],row=1, col=i+1, showticklabels=False)
            
        else:
            fig5.append_trace(go.Bar(x=sec[list_columns[i]]+2, y=sec['Depth(m)_imputed'],name=list_columns[i],orientation='h', marker=dict(color=sec[list_columns[i]], coloraxis="coloraxis"),marker_coloraxis=None), row=1, col=i+1)
            
            # Update axis properties
            fig5.update_xaxes(title_text='GMM',range=[0,max(sec['GMM'])],row=1, col=i+1, showticklabels=False)
            
    layout = go.Layout(xaxis=dict(side='top'),xaxis2=dict(side='top'),xaxis3=dict(side='top'),xaxis4=dict(side='top'),xaxis5=dict(side='top'),xaxis6=dict(side='top'))
    fig5['layout'].update(layout)      
    fig5.update_yaxes(title_text='Depth', autorange="reversed", row=1, col=1)      
    fig5.update_layout(showlegend=False, height=1500, width=1500)

    # Plot!
    st.plotly_chart(fig5, use_container_width=True)


def plotting(ana):
    
    if (ana=='KMean'):
        well_log_display(df_new['Depth(m)_imputed'].min(),df_new['Depth(m)_imputed'].max(), df_new, list_KMean)
    else:
        well_log_display(df_new['Depth(m)_imputed'].min(),df_new['Depth(m)_imputed'].max(), df_new, list_GMM)

plotting(Select_Method)

    




 
