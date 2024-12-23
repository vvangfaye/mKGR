# mKGR

The official code of paper **Learning to Reason over Multi-Granularity Knowledge Graph for Zero-shot Urban Land-Use Mapping**.

**Abstract:** This paper introduces a multi-granularity knowledge graph reasoning (mKGR) framework. Only with indirect supervision from other tasks, mKGR can automatically integrate multimodal geospatial data as varying granularity entities and rich spatial-semantic interaction relationships. Subsequently, mKGR incorporates a novel fault-tolerant knowledge graph embedding method to establish relationships between geographic units and land-use categories, thereby reasoning land-use mapping outcomes. Extensive experiments demonstrate that mKGR not only outperforms existing zero-shot approaches but also exceeds those with direct supervision. Furthermore, this paper reveals the superiority of mKGR in large-scale holistic reasoning, an essential aspect of land-use mapping. Benefiting from mKGR's zero-shot classification and large-scale holistic reasoning capabilities, a comprehensive urban land-use map of China is generated with low-cost.

- [x] Products: Publicly accessible on [ArcGIS Online](https://www.geosceneonline.cn/geoscene/apps/mapviewer/index.html?webmap=ad747de4b4ad4b558141c638e23960ca).
- [x] Code: Publicly available in this repository.
- [ ] Dataset: Restricted access on [Zenodo](https://zenodo.org/records/11311869) (to be released upon paper publication; email us for early access).

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

Option 1: Directly download the [constructed graph](https://zenodo.org/records/11311869).

Option 2：Construct the graph in the [KG_construction](./KG_construction/) folder.

## MGKG Reasoning for Land-use Mapping

Train the graph embedding in the [KG_embedding](./KG_embedding/) folder and infer to obtain the land-use mapping result. 


## Other Code

 [figure_script](./figure_script/): The code for generating the figures in the paper.

 [landuse_app](./landuse_app/): The code for 15-minute city application of land-use mapping results.


## The reuslt of our mKGR and its application

![result](./images/china_result.jpg)
**a**, The samples are distributed across various regions of China. From i to v are Beijing, Suzhou, Zhengzhou, Chengdu, and Shenzhen respectively. At the same time, it shows the different OA in each province of China. **b**, An application case for inclusive 15-minute walkability assessment. The shortest walking time from each residential parcel to other land-use categories are calculated and the average walkability values are visualized.

We have published the land-use mapping results of China in [ArcGIS Online](https://www.geosceneonline.cn/geoscene/apps/mapviewer/index.html?webmap=ad747de4b4ad4b558141c638e23960ca).
## Contact
If you have any questions about it, please let me know. (Create an 🐛 issue or 📧 email: wangfaye@whu.edu.cn)
