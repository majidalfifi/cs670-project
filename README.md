D-Cube: Dense-Block Detection in Terabyte-Scale Tensors
========================
**Authors**: Kijung Shin, Bryan Hooi, Jisu Kim, and Christos Faloutsos   
**D-Cube (Disk-based Dense-block Detection)** is an algorithm for detecting dense subtensors in tensors.
**D-Cube** has the following properties:
 * *scalable*: **D-Cube** handles large data not fitting in memory or even on a disk.
 * *fast*: Even when data fit in memory, **D-Cube** outperforms its competitors in terms of speed.
 * *accurate*: **D-Cube** gives high accuracy in real-world data as well as theoretical accuracy guarantees.  
**Original Repo**: https://github.com/kijungs/dcube

Our contributions:
=======================
0. Spark implementation
0. Evaluation of the algorithm on real data from Twitter, Amazon, and Yelp.

Building and Running D-Cube
========================

`mvn clean package`

**Single version:**

`java -cp dcube-1.0-SNAPSHOT.jar dcube.Proposed dataset.txt.gz output 3 susp density 3`

**Hadoop version:**

`hadoop fs -put dataset.txt.gz .`

`hadoop jar dcube-1.0-SNAPSHOT.jar dcube.hadoop.ProposedHadoop dataset.txt.gz output 3 geo density 3 4 log`

**Spark version:**
See notebook: [???]

Datasets:
====================
Amazon Dataset:  
[249 MB] http://students.cse.tamu.edu/kaghazgaran/Dataset/AmazonDataset.zip

Twitter Dataset:  
[3.3G] http://students.cse.tamu.edu/alfifima/data/hashtags-data.txt.gz  
[4.0G] http://students.cse.tamu.edu/alfifima/data/urls-data.txt.gz  
[2.4G] http://students.cse.tamu.edu/alfifima/data/mentions-data.txt.gz  
