# Report - Juan Diego Niebles Navas - Pipeline

This report is for doing a summary of my experience with the 20-Newsgroup Dataset exploratoy feature analysis with the agent. First, we saw the corpus of the dataset that consisted in 2,257 documents that follow this distribution of categories: soc.religion.christian (599), sci.med (594), comp.graphics (584), and alt.atheism (480). We run some tool for quality assurance, in which we found there are no duplicates or nulls in the dataset. We could also see a high amount of caracters, with a mean length of 1,992.96 characters, a median of 1,235.0 characters, and a maximum document length of 60,713.0 characters. Below we are going to see a boxplot of the text frecuency, in which we can see that they are often of the same size but between 0 and 10000, but there are some higher text lenght that are outliers.  
 
![alt text](..\plots\Juan_Diego_Niebles_Navas_agentic-pipeline_plot_describe_data_5.png)

Then, to prepare the textual data for the analysis tools, we discuss with the agent what params use for creating the Document-term-matrix, in which we choose using unigrams tokens with a maximum vocabulary size of 1000 terms and a relative fecuency of 25% requiring the presence of 5-6 documents to be considered. This decision was for filtering mainly high used terms with an only dimension, too focus mainly on how each word can influence. With this, the resulting DTM 12,814 non-zero entries out of 2,257,000 total elements, corresponding to a matrix sparsity of 90.57% (see the heatmap below to get an idea), but at the moment of evaluate the high frequencies we saw lot of stop-words such as "the", "of", "to", and so on. To eliminate this a variance filtering was applied with a threshold of 0.05, dropping 20 near constant words.

![alt text](..\plots\Juan_Diego_Niebles_Navas_agentic-pipeline_plot_dtm_heatmap_9.png)

Next, with the agent we run Spearman and Pearson correlations to see which one had better results. Here by my one judgement, and taking into account the results, I believe there was more information with Spearman because the algorithm, probably because the algorithm can handle on a better way the high sparsity of the matrix. With this, we found the next plots with the leading terms for each category:
### Atheist:
![alt text](..\plots\Juan_Diego_Niebles_Navas_agentic-pipeline_plot_turn154.png)
### Computer graphics
![alt text](..\plots\Juan_Diego_Niebles_Navas_agentic-pipeline_plot_turn176.png)
### Religion
![alt text](..\plots\Juan_Diego_Niebles_Navas_agentic-pipeline_plot_turn180.png)
###   Sci.med
![alt text](..\plots\Juan_Diego_Niebles_Navas_agentic-pipeline_plot_turn170.png)  

After that, we used the frequent pattern mining tool, and toguether we choose that the best parameters where MaxFP-Growth under tf-idf with a threshold of 30 occurencies. Here we had mainly structural headers and like lines, subject, organization, edu, writes, and article. Some thimes it was also possible to see topic-specific sub-patters. For example, in atheism it was also terms such as islamic or god; in religion god, people and church; in comp.graphics looking, help; in sci.med help or case. However, on this run there were better results on the spearman_filter_tool than on the frequent pattern mining, probably indicating that the tool used the original dtf without the variance filtering.  
  
Finally, we run tool for reducing the dimensions and to compare document similarity. Regarding the document similarity we just compared between the first 5 indexes, and the ranges of similarity was between 0.4715 and 0.5016 even on different categories, probably because there were lot of similar words used on both documents. Regarding the dimension reduction, below you can see the results, but it even though the best algorith for visualization seem to be umap, the data highly overlapse between clusters, probably because there are lot of words shared between documents, but there still are som isolated group of terms of each cluster which maybe are the ones that more clearly represent their own context.

![alt text](..\plots\Juan_Diego_Niebles_Navas_agentic-pipeline_plot_reduce_dimensions_19.png)
![alt text](..\plots\Juan_Diego_Niebles_Navas_agentic-pipeline_plot_reduce_dimensions_20.png)
![alt text](..\plots\Juan_Diego_Niebles_Navas_agentic-pipeline_plot_reduce_dimensions_21.png)