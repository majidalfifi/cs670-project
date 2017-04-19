D-Cube: Dense-Block Detection in Terabyte-Scale Tensors
========================
**Authors**: Kijung Shin, Bryan Hooi, Jisu Kim, and Christos Faloutsos

**D-Cube (Disk-based Dense-block Detection)** is an algorithm for detecting dense subtensors in tensors.
**D-Cube** has the following properties:
 * *scalable*: **D-Cube** handles large data not fitting in memory or even on a disk.
 * *fast*: Even when data fit in memory, **D-Cube** outperforms its competitors in terms of speed.
 * *accurate*: **D-Cube** gives high accuracy in real-world data as well as theoretical accuracy guarantees.


Building and Running D-Cube
========================

`mvn clean package`

**Single version:**

`java -cp target/dcube-1.0-SNAPSHOT.jar dcube.Proposed example_data.txt output 3 geo density 3`

**Hadoop version:**

`hadoop fs -put example_data.txt .`

`hadoop jar target/dcube-1.0-SNAPSHOT.jar dcube.hadoop.ProposedHadoop example_data.txt dcube_output 3 geo density 3 4 log`