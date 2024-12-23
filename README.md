# mKGR

The official code of paper **Learning to Reason over Multi-Granularity Knowledge Graph for Zero-shot Urban Land-Use Mapping**.

**Abstract:** In mKGR, we introduce the concept of multi-granularity knowledge graph (MGKG) tailored for the ufrban landscape. Leveraging the flexibility of MGKG, mKGR seamlessly integrates multimodal geospatial data, aggregating them into varying granularity entities within the knowledge graph. Subsequently, considering the real world contains certain noise, we develop a fault-tolerance knowledge graph embedding method to complete the graph. The completed knowledge graph finally yields the land-use mapping result.

- [x] Products: Publicly available in [ArcGIS Online](https://www.geosceneonline.cn/geoscene/apps/mapviewer/index.html?webmap=ad747de4b4ad4b558141c638e23960ca).
- [x] Code: Publicly available in this rep.
- [x] Dataset: To be released upon paper publication (Email for access).

**Method's framework**
![framework](./images/framework.jpg)

## Requirement
Ubuntu 20.04 (or other Linux distribution), one GPU (video memory greater than 12GB and support cuda)
* python>=3.11.5
* numpy>=1.26.2
* pytorch>=2.2.1
* pandas>=2.2.2
* geopandas>=0.14.0

## MGKG Construction

Option 1: Directly download the [constructed graph](https://zenodo.org/records/11311869?preview=1).

Option 2：Construct the graph in the [KG_construction](./KG_construction/) folder.

## MGKG Reasoning for Land-use Mapping

Train the graph embedding in the [KG_embedding](./KG_embedding/) folder and infer to obtain the land-use mapping result. 


## Ohter Code

 [figure_script](./figure_script/): The code for generating the figures in the paper.

 [landuse_app](./landuse_app/): The code for 15-minute city application of land-use mapping results.


## The reuslt of our mKGR and its application

![result](./images/china_result.jpg)
**a**, The samples are distributed across various regions of China. From i to v are Beijing, Suzhou, Zhengzhou, Chengdu, and Shenzhen respectively. At the same time, it shows the different OA in each province of China. **b**, An application case for inclusive 15-minute walkability assessment. The shortest walking time from each residential parcel to other land-use categories are calculated and the average walkability values are visualized.

We have published the land-use mapping results of China in [ArcGIS Online](https://www.geosceneonline.cn/geoscene/apps/mapviewer/index.html?webmap=ad747de4b4ad4b558141c638e23960ca).
## Contact
If you have any questions about it, please let me know. (Create an 🐛 issue or 📧 email: wangfaye@whu.edu.cn)
