# SIADS-696 Milestone II Project Repository  

This Repository contains code, data and other data created during the Milestone II course (SIADS-696) of the MADS (Master of Applied Data Science) program at the Universtiy of Michigan, School of Information.

## Predicting how substances interact with cells through cell image embeddings  
Authors: Eleanor Siler, Florian Gillet, Matthias Huebscher  

### Project Introduction  
Understanding and predicting how small molecules affect living cells is important for developing and understanding their mechanism of action (MoA). Knowing the MoA of small molecules is important for drug development, among other things. However, discovering the MoA of chemical compounds is complex and costly to evaluate. Today's technologies use high-content-screening methods to evaluate the effects of small molecules on cells. In high-content-screening, cells are exposed to different compounds and then imaged using high-throughput microscopy; these images are then used to explore and identify how these compounds affect the cells.
Our proposal is to predict MoA based on these microscopy images taken after exposing cells to different compounds. We will create and use image embeddings of these pictures for our analysis. Each microscopy image is labeled with the name of the compound that was used to treat the cell. A 2021 study used a similar approach to identify MoA of compounds [^1]. Our approach varies in a couple of different ways. The paper describes a couple of advanced methods used in their models. Our aim is to use simpler models to achieve comparable results. In contrast to paper, we want to put more emphasis on the supervised prediction task to balance more between unsupervised and supervised learning. Unlike the previous publication, we will apply transfer learning, a growing field widely used in image analysis, to encode the images.  

### GitHub Structure
|Folder|Description|
|------|-----------|  
|data|Contains mainly CSV data files used in the project. Due to storage limitations the intermediate artifacts as well as the file containing the full image embedding vectors is not stored in this directory|
|images|Contains images of plots such as UMAP or t-SNE scatter plots which are used in the final project report|
|models|Contains a feed-forward neural network model used to predict MoAs. Model implemented in PyTorch|
|notebooks|Contains Jupyter notebooks covering data exploration, unsupervised clustering and dimensionality reduction and supervised model building|
|scripts|Contains Python and Shell scripts used to download the dataset (CSV and image files), do the transfer learing and the crossvalidation of the neural network|  

We used image set BBBC021v1 [Caie et al., Molecular Cancer Therapeutics, 2010], available from the Broad Bioimage Benchmark Collection [Ljosa et al., Nature Methods, 2012].


[^1]: Janssens, Rens, et al. "Fully unsupervised deep mode of action learning for phenotyping 
high-content cellular images." *Bioinformatics* 37.23 (2021): 4548-4555.
https://doi.org/10.1093/bioinformatics/btab497
