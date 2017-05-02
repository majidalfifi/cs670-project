
# coding: utf-8

# # Dense-Block Detection in Terabyte-Scale Tensors

# In[13]:

# Hide code
from IPython.display import HTML

HTML('''<script>
code_show=true; 
function code_toggle() {
 if (code_show){
 $('div.input').hide();
 } else {
 $('div.input').show();
 }
 code_show = !code_show
} 
$( document ).ready(code_toggle);
</script>
<form action="javascript:code_toggle()"><input type="submit" value="Click here to toggle on/off the raw code."></form>''')


# In[1]:

# general imports
sc.setLogLevel("INFO")

from IPython.display import Image, HTML
import pickle

from pyspark.sql.types import *

import plotly
from plotly import tools
import plotly.graph_objs as go
plotly.offline.init_notebook_mode()

import math
import numpy as np
import pandas as pd

import pyspark.sql.functions as F

pd.set_option('display.max_colwidth', 600)


# In[3]:

# read users that were suspended by Twitter
from pyspark.sql.types import StructType, StructField, StringType, LongType

path = '/user/aralytics-compliance/scraped/20161226'

schema = StructType([StructField('user_id',LongType(),True),StructField('status',StringType(),True)])

user_status = spark.read.csv(
    path=path,
    encoding='UTF-8',
    sep='\t',
    schema=schema,
    quote="",
    mode='DROPMALFORMED',
    ignoreLeadingWhiteSpace=True)

users_suspended = user_status.filter(F.instr(user_status.status,'Account Suspended')>0).select('user_id').cache()


# ## Twitter datasets
# 0. User vs. URLs
# 0. User vs. Mentions
# 0. User vs. Hashtags

# In[2]:

# tweets = spark.read.parquet('/user/aralytics-parquet/2015/01')

# tweets_count   = tweets.count()
# users_count    = tweets.select('user.id').distinct().count()
# hashtags_count = tweets.select(F.explode('entities.hashtags.text')).distinct().count()
# urls_count     = tweets.select(F.explode('entities.urls.expanded_url')).distinct().count()


# ## Run D-Cube on Twitter Users & Hashtags, URLs, and Mentions
# 
# **Input**: Arabic Tweets generated in January 2015
# 
# | Set    | Size          |
# |--------|---------------|
# | Tweets | 1,231,415,376 |
# | users (unique)  | 8,242,091   |
# | Hashtags (unique)| 1,971,871 |
# | URLs (unique)| 27,649,967   |
# | Mentions (unique)| 4,435,200   |
# 
# 
# **Output**: 
# 0. Potential fraudulent users and *hashtags* they participated in.
# 0. Potential fraudulent users and *urls* they shared.a
# 0. Potential fraudulent users and other users they *mentioned*.

# **Experiment 1: user and hashtags**
# 
# | Set    | Size          |
# |--------|---------------|
# | users  | 4,660,526   |
# | Hashtags| 1,971,871 |
# | total tuples| 433,618,654   |

# In[4]:

# generate datasets from raw tweets
run=False 
if run: # long running operation
    
    tweets = spark.read.parquet('/user/aralytics-parquet/2015/01')

    hashtags = tweets.select(F.col('user.id').alias('user_id'),
                             'created_at',
                             F.explode('entities.hashtags.text').alias('hashtag'))

    urls     = tweets.select(F.col('user.id').alias('user_id'),
                             'created_at',
                             F.explode('entities.urls.expanded_url').alias('url'))
    
    mentions = tweets.select(F.col('user.id').alias('user_id'),
                             'created_at',
                             F.explode('entities.user_mentions.id').alias('mention_id'))

    hashtags.write.parquet('dcube/data-raw/user-hashtags-2015-01')
    urls.write.parquet('dcube/data-raw/user-urls-2015-01')
    mentions.write.parquet('dcube/data-raw/user-mentions-2015-01')


# In[5]:

# index hashtags
run=False 

if run: # long running operation
    hashtags = spark.read.parquet('dcube/data-raw/user-hashtags-2015-01')
    hashtags_indexed = hashtags.select('hashtag')                           .distinct()                           .sort('hashtag')                           .rdd                           .map(lambda x: x[0])                           .zipWithIndex()                           .toDF(['hashtag','hashtag_index']).cache()

    user_ids_indexed = hashtags.select('user_id')                           .distinct()                           .sort('user_id')                           .rdd                           .map(lambda x: x[0])                           .zipWithIndex()                           .toDF(['user_id','user_id_index'])

    hashtags = hashtags.join(hashtags_indexed,['hashtag'])                       .join(user_ids_indexed,['user_id'])                       .withColumn('timestamp', 24*(F.dayofmonth('created_at')-1) + F.hour('created_at'))
    
    hashtags.write.parquet('dcube/data/hashtags-indexed')

hashtags = spark.read.parquet('dcube/data/hashtags-indexed')

# show a sample
print 'Sample of User Hashtags Relation:'                        
hashtags.filter(F.instr(F.col('hashtag'),'m')>0).show(10,False)


# In[6]:

# index URLs
run=False 

if run: # long running operation
    urls = spark.read.parquet('dcube/data-raw/user-urls-2015-01')
    urls_indexed = urls.select('url')                   .distinct()                   .sort('url')                   .rdd                   .map(lambda x: x[0])                   .zipWithIndex()                   .toDF(['url','url_index'])

    user_ids_indexed = urls.select('user_id')                    .distinct()                    .sort('user_id')                    .rdd                    .map(lambda x: x[0])                    .zipWithIndex()                    .toDF(['user_id','user_id_index'])

    urls = urls.join(urls_indexed,['url'])               .join(user_ids_indexed,['user_id'])               .withColumn('timestamp', 24*(F.dayofmonth('created_at')-1) + F.hour('created_at'))
    
    urls.write.parquet('dcube/data/urls-indexed')

urls = spark.read.parquet('dcube/data/urls-indexed')

# show a sample
print 'Sample of User-URLs Relation:'                        
urls.filter(F.length('url')<40).show(10,False)


# In[7]:

# index mentions
run=False 

if run: # long running operation
    mentions = spark.read.parquet('dcube/data-raw/user-mentions-2015-01')
    user_ids_indexed = mentions.select('user_id')                           .distinct()                           .sort('user_id')                           .rdd                           .map(lambda x: x[0])                           .zipWithIndex()                           .toDF(['user_id','user_id_index'])

    mention_ids_indexed = mentions.select('mention_id')                           .distinct()                           .sort('mention_id')                           .rdd                           .map(lambda x: x[0])                           .zipWithIndex()                           .toDF(['mention_id','mention_id_index'])

    mentions = mentions.join(user_ids_indexed,['user_id'])               .join(mention_ids_indexed,['mention_id'])               .withColumn('timestamp', 24*(F.dayofmonth('created_at')-1) + F.hour('created_at'))
    
    mentions.write.parquet('dcube/data/mentions-indexed')

mentions = spark.read.parquet('dcube/data/mentions-indexed')

#show a sample
print 'Sample of User-Mentions Relation:'                        
mentions.show(10,False)


# In[8]:

# save relations in a tensor format ready for D-Cube
run =False
if run:
    hashtags.groupby('user_id_index','hashtag_index','timestamp').count().write.csv('dcube/data/hashtags-tensor')
    urls.groupby('user_id_index','hashtag_index','timestamp').count().write.csv('dcube/data/urls-tensor')
    mentions.groupby('user_id_index','hashtag_index','timestamp').count().write.csv('dcube/data/mentions-tensor')


# ## Run D-Cube offline
# <pre>
# java -Xmx24048m -Xms24048m -cp target/dcube-1.0-SNAPSHOT.jar dcube.Proposed hashtags-data.txt hashtags-output 3 geo density 1
# input_path: hashtags-data.txt
# output_path: hashtags-output
# dimension: 3
# density_measure: geo
# policy: density
# num_of_blocks: 1
# 
# computing proper buffer size
# 
# storing the input tensor in the binary format...
# Preprocess,134340
# 
# running the algorithm...
# 
# Block: 1
# Volume: 2740 X 28 X 744
# Density: 185661.31603659378
# Mass: 71485057
# Running time: 70.488 seconds
# Writing outputs...
# Outputs were written. 4.566 seconds was taken.
# </pre>

# # Analysis 1: User-Hashtags

# In[14]:

# Read d-cube output

schema = StructType([StructField(col, LongType(), True) for col in ['user_id_index','hashtag_index','timestamp','count']])

dcube_output_hashtags = spark.read.csv(
    path='dcube/output/hashtags-output/block_1.tuples',
    sep=',',
    schema=schema)\
    .join(hashtags.select('hashtag', 'hashtag_index').distinct(), ['hashtag_index'])\
    .join(hashtags.select('user_id', 'user_id_index').distinct(), ['user_id_index'])                

print 'Sample of Tuples found by D-Cube'
dcube_output_hashtags.filter(F.instr(F.col('hashtag'),'a')>0).show()


# In[36]:

# users found by D-cube
dcube_users_count = dcube_output_hashtags.select('user_id').distinct().count()
dcube_hashtags_count =dcube_output_hashtags.select('hashtag').distinct().count()
dcube_users_suspended = dcube_output_hashtags.select('user_id').distinct().join(users_suspended, 'user_id').count()
suspended_by_twitter = hashtags.select('user_id').distinct().join(users_suspended,'user_id').count()
print 'D-Cube found:'
print "  {:,}".format(dcube_users_count).rjust(9), ' users.'
print "     of which {:,}".format(dcube_users_suspended).rjust(9), ' were suspended.'
print "  {:,}".format(dcube_hashtags_count).rjust(9), ' hashtags.'
print "Total users in dataset suspended by Twitter ", "{:,}".format(suspended_by_twitter).rjust(9)


# In[55]:

circle1_text = 'Suspended by Twitter '+"{:,}".format(suspended_by_twitter).rjust(9)
circle2_text = "{:,}".format(dcube_users_suspended).rjust(9)
circle3_text = 'Found by D-Cube '+ "{:,}".format(dcube_users_count).rjust(9)
trace0 = go.Scatter(
    x=[1, 1.75, 2.5],
    y=[1, 1, 1],
    text=[circle1_text, circle2_text, circle3_text],
    mode='text',
    textfont=dict(
        color='black',
        size=18,
        family='Arail',
    )
)

data = [trace0]

layout = {
    'xaxis': {
        'showticklabels': False,
        'autotick': False,
        'showgrid': False,
        'zeroline': False,
    },
    'yaxis': {
        'showticklabels': False,
        'autotick': False,
        'showgrid': False,
        'zeroline': False,
    },
    'shapes': [
        {
            'opacity': 0.3,
            'xref': 'x',
            'yref': 'y',
            'fillcolor': 'blue',
            'x0': 0,
            'y0': 0,
            'x1': 2.5,
            'y1': 2,
            'type': 'circle',
            'line': {
                'color': 'blue',
            },
        },
        {
            'opacity': 0.3,
            'xref': 'x',
            'yref': 'y',
            'fillcolor': 'gray',
            'x0': 1.5,
            'y0': 0,
            'x1': 3,
            'y1': 2,
            'type': 'circle',
            'line': {
                'color': 'gray',
            },
        }
    ],
    'margin': {
        'l': 20,
        'r': 20,
        'b': 100
    },
    'height': 600,
    'width': 800,
}

fig = go.Figure(data=data, layout=layout)

plotly.offline.iplot(fig, show_link=False)


# # Next User-URLS ...

# In[57]:

# Read d-cube output

schema = StructType([StructField(col, LongType(), True) for col in ['user_id_index','url_index','timestamp','count']])

dcube_output_urls = spark.read.csv(
    path='dcube/output/urls-output/block_1.tuples',
    sep=',',
    schema=schema)\
    .join(urls.select('url', 'url_index').distinct(), ['url_index'])\
    .join(urls.select('user_id', 'user_id_index').distinct(), ['user_id_index'])                

print 'Sample of Tuples found by D-Cube'

dcube_output_urls.filter(F.length('url')<40).show(10,False)


# In[58]:

# users found by D-cube

urls_dcube_users_count = dcube_output_urls.select('user_id').distinct().count()
urls_dcube_hashtags_count =dcube_output_urls.select('url').distinct().count()
urls_dcube_users_suspended = dcube_output_urls.select('user_id').distinct().join(users_suspended, 'user_id').count()
urls_suspended_by_twitter = urls.select('user_id').distinct().join(users_suspended,'user_id').count()

print 'D-Cube found:'
print "  {:,}".format(urls_dcube_users_count).rjust(9), ' users.'
print "     of which {:,}".format(urls_dcube_users_suspended).rjust(9), ' were suspended.'
print "  {:,}".format(urls_dcube_users_suspended).rjust(9), ' urls.'
print "Total users in dataset suspended by Twitter ", "{:,}".format(urls_suspended_by_twitter).rjust(9)


# # Next User-Mentions

# In[61]:

# Read d-cube output

schema = StructType([StructField(col, LongType(), True) for col in ['user_id_index','mention_id_index','timestamp','count']])

dcube_output_mentions = spark.read.csv(
    path='dcube/output/mentions-output/block_1.tuples',
    sep=',',
    schema=schema)\
    .join(mentions.select('mention_id', 'mention_id_index').distinct(), ['mention_id_index'])\
    .join(mentions.select('user_id', 'user_id_index').distinct(), ['user_id_index'])                

print 'Sample of Tuples found by D-Cube'

dcube_output_mentions.show(10,False)


# In[66]:

# users found by D-cube

mentions_dcube_users_count = dcube_output_mentions.select('user_id').distinct().count()
mentions_dcube_hashtags_count =dcube_output_mentions.select('mention_id').distinct().count()
mentions_dcube_users_suspended = dcube_output_mentions.select('user_id').distinct().join(users_suspended, 'user_id').count()
mentions_suspended_by_twitter = mentions.select('user_id').distinct().join(users_suspended,'user_id').count()
mentions_suspended = mentions.select(F.col('mention_id').alias('user_id')).distinct().join(users_suspended,'user_id').count()

print 'D-Cube found:'
print "  {:,}".format(mentions_dcube_users_count).rjust(9), ' users.'
print "     of which {:,}".format(mentions_dcube_users_suspended).rjust(9), ' were suspended.'
print "  {:,}".format(mentions_dcube_users_suspended).rjust(9), ' mentioned users'
print "     of which {:,}".format(mentions_suspended).rjust(9), ' were suspended.'
print "Total users in dataset suspended by Twitter ", "{:,}".format(mentions_suspended_by_twitter).rjust(9)


# # Appendix A (D-Cube in Spark)
# We found out that the MapReduce framework is not suitable for D-Cube because the design of D-Cube requires starting many MapReduce jobs to do the subtasks. In this case, the overhead of starting and shuttding down the jobs is much more than the logic to be executed. For this reason, and to ease the integration with other Spark analytical tools, we implemented the algorithm in Spark.
# 
# Below we show an implementation of D-Cube in Spark and a sample of execution on an example dataset.
# 
# Notes:
# 0. We only implemented the geo_density measure, other measures should be an easy extension
# 0. We only evaluated the Spark implementation on the first block generated

# In[71]:

cols = ['col1', 'col2', 'col3', 'measure']

import math
import numpy as np
import pandas as pd
from datetime import datetime

from pyspark.sql.types import *
import pyspark.sql.functions as F

import sys

def get_mass(df):
    return df.select(F.sum('measure')).collect()[0][0]
    
def get_cardinalities(df):
    return [df.select(F.col(col)).distinct().count() for col in cols[:-1]]

def multiply_list(l):
    result = 1
    for x in l:
        result = result *x
    return result
#     return np.power(10,sum(np.log10(l)))

def geo_density(df):
    producOfCardinalities = multiply_list(get_cardinalities(new_subtensor))
    if producOfCardinalities==0:
        return -1
    
    return df.select(F.sum('measure')/
                F.pow(F.lit(producOfCardinalities),1.0/(len(cols)-1))).collect()[0][0]

def get_threshold(df, col):
    return get_mass(df) * 1.0 / df.select(F.col(col)).distinct().count()


# In[69]:

schema = StructType([StructField(col, LongType(), True) for col in cols])

data = spark.read.csv(
    path='dcube/data/example_data.txt.gz',
    sep=',',
    schema=schema).cache()

data.show()


# In[72]:

get_ipython().system(u'hadoop fs -rm -r /tmp/subtensor*')

max_density = -sys.maxint

subtensor = data
iteration = 0

while subtensor.count()>1:
    max_mode = -1
    temp_max_score=-sys.maxint    
    print 'iteration: ',iteration
    iteration += 1
    
    print get_cardinalities(subtensor)

    for index, col in enumerate(cols[:-1]): # find best dimension to delete from
        print col    
        threshold = get_threshold(subtensor, col)
        print 'threshold', threshold

        to_remove = subtensor.groupby(col).agg(F.sum('measure').alias('mass')).filter(F.col('mass')<=threshold)
#         print 'to_remove', to_remove.count()

        removedMassSum = get_mass(subtensor.join(to_remove, col))
        print 'removedMassSum', removedMassSum
        new_subtensor = subtensor.join(to_remove, col,'leftanti')
        
        density = geo_density(new_subtensor)

        print 'density', density

        if density>temp_max_score:
            temp_max_score = density
            next_subtensor = new_subtensor
            
    filename = '/tmp/subtensor-'+str(iteration)
    next_subtensor.write.parquet(filename)
    subtensor = spark.read.parquet(filename).cache()
    
    if temp_max_score > max_density:
        max_density=temp_max_score
        dense_tensor_filename = filename
        
    print '------------------------'

dense_tensor = spark.read.parquet(dense_tensor_filename).cache()
print 'cardinalities of the densest tensor', get_cardinalities(dense_tensor)


# # Future Work
# 0. Train a classifier (e.g. Naive Bayes) using the set of suspended users as ground truth and check how many of the users in the blocks found by D-Cube exhibit similar behavior to suspended users.
# 0. Explore the feasibility of *learning* a density function to serve in place of the generic suspeciousness metric
