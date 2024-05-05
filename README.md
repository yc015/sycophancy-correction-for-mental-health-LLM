# Detecting and Correcting Sycophancy of Chatbot Therapist

Team Members: [Yida Chen](https://yc015.github.io/), [Catherine Yeh](https://catherinesyeh.github.io/), [Jerry Huang](https://www.linkedin.com/in/jerryh01?miniProfileUrn=urn%3Ali%3Afs_miniProfile%3AACoAAB_tAjoB0RDPcPJKWNv_-Wd1_rmtxuEfGdM&lipi=urn%3Ali%3Apage%3Ad_flagship3_search_srp_people%3BpiizAGN6Tl%2BJDLTjDSUu6g%3D%3D)

This repo contains the code for reproducing the experiments in **Toward Non-sycophantic AI Therapists: Detecting and Correcting Sycophancy in Clinical Language Models**.

## What is this project about?
> Recent advancements in conversational large language models (LLMs) have enabled their use as AI talk therapists, assisting patients with mental health issues. However, [a recent incident](https://www.euronews.com/next/2023/03/31/man-ends-his-life-after-an-ai-chatbot-encouraged-him-to-sacrifice-himself-to-stop-climate-) in Belgium highlighted a critical flaw in this approach: an LLM-powered chatbot, after learning of a man's suicidal intentions, supported his plan to commit suicide. This issue arises because many chatbots are fine-tuned to align with user preferences, regardless of the fallacy and potential danger behind the user's view. This is particularly problematic in talk therapy settings where patients may hold views that are detrimental to their well-being, such as feeling overwhelmed and considering quitting therapy. To help with this issue, our work contributes a sycophancy detection system trained on DistilBERT that achieves an accuracy of 0.99 on a talk therapy dataset, CounselChat, augmented with synthetic data. Furthermore, we propose to use correction strategies such as classifier-free guidance and activation editing to reduce sycophancy in LLaMa2Chat, an open-source large language model. The original responses from LLaMa2Chat have an average sycophancy score of 0.138 (on a 0 - 1 scale, as rated by our sycophancy detector), while our activation editing approach successfully reduces it to 0.002. Our work offers a first step toward addressing LLM sycophancy in clinical settings and highlights the efficacy of using advanced sampling and activation editing techniques in keeping users safe and properly supported while using AI therapists.

## Content
- [data_exploration.ipynb](https://github.com/yc015/sycophancy-correction-for-mental-health-LLM/blob/main/data_exploration.ipynb)

  > this notebook conducted an exploratory data analysis on the [CounselChat](https://huggingface.co/datasets/nbertagnolli/counsel-chat) dataset.  
- [autogenerate_sycophantic_answers](https://github.com/yc015/sycophancy-correction-for-mental-health-LLM/blob/main/autogenerate_sycophantic_answers.ipynb)

  > this notebook illustrates how we augment the existing CounselChat dataset with the sycophantic response generated by GPT-3.5 model.  
- [Sycophancy_Detector.ipynb](https://github.com/yc015/sycophancy-correction-for-mental-health-LLM/tree/main)

  > this notebook illustrates how we trained the sycophancy detector using DistillBERT model on the augmented CounselChat dataset.  
- [generate_response_llama2.ipynb](https://github.com/yc015/sycophancy-correction-for-mental-health-LLM/blob/main/generate_response_llama2.ipynb)

  > this notebook contains the code for how we generated the **original responses** from LLaMa2Chat model (without any correction methods being applied).  
- [cfg_generate_response_llama2.ipynb](https://github.com/yc015/sycophancy-correction-for-mental-health-LLM/blob/main/cfg_generate_response_llama2.ipynb)

  > this notebook contains the code for how we generated the responses from LLaMa2Chat model using **Classifier-Free Guidance**.  
- [contrast_decoding_generate_response_llama2.ipynb](https://github.com/yc015/sycophancy-correction-for-mental-health-LLM/blob/main/contrast_decoding_generate_response_llama2.ipynb)

  > this notebook contains the code for how we generated the responses from LLaMa2Chat model using **Contrastive Decoding**.  
- [activation_editing_generate_response_llama2.ipynb](https://github.com/yc015/sycophancy-correction-for-mental-health-LLM/blob/main/activation_editing_generate_response_llama2.ipynb)

  > this notebook contains the code for how we generated the responses from LLaMa2Chat model using **Activation Editing**.  
- [internal_model_of_sycophancy.ipynb](https://github.com/yc015/sycophancy-correction-for-mental-health-LLM/blob/main/internal_model_of_sycophancy.ipynb)

  > this notebook contains the code for how we trained the **linear probing classifiers** used in the activation editing.

## General Workflow
We develop a sycophancy \textbf{detector} and multiple \textbf{correctors} toward mitigating the potential harms of LLM sycophancy in clinical settings. 
### Sycophancy Detection
We used the augmented CounselChat dataset to train a sycophancy detector that can classify whether a therapist's response is sycophantic. Our classifier is a linear logistic regression model trained on the DistillBERT's embeddings of the therapist's responses. Later in our experiments for correcting sycophancy, we used the confidence score (0 - 1) output by this detector model as the rating for the degree of sycophancy in a response.
![detection_pipeline](https://github.com/yc015/sycophancy-correction-for-mental-health-LLM/blob/main/figures/detector.png)

### Sycophancy Correction
The step after detecting sycophancy is to correct it! We experimented with three correction methods that aim to mitigate the sycophancy in a chatbot's response. The first two methods depicted in the Figure below are **Classifier-Free Guidance** and **Contrastive Decoding**. These two methods sample the chatbot therapist's responses from a modified distribution where the sycophantic responses are less likely to appear.
![correction_method](https://github.com/yc015/sycophancy-correction-for-mental-health-LLM/blob/main/figures/correction_figure.png)

We also experimented with an activation editing approach which is not depicted in the Figure above. This method relies on the possibility that an LLM may have an internal representation of the abstract concept, such as sycophancy. By extracting such representation from the model and purposely modified it, we can get a response with less sycophancy from the chatbot model. For more information about this approach, see this prior work: [Activation Addition: Steering Language Models Without Optimization](https://arxiv.org/abs/2308.10248).

## Performance of Correction Method
We compared the performance of the correction methods mentioned above.
### Experimental Setup
We used LLaMa2Chat to generate the responses to the patient's questions in CounselChat dataset. In particular, we compared the responsed generated under 4 conditions:

1. without any correction method applied
2. with **Classifier-Free Guidance**
3. with **Contrastive Decoding**
4. with **Activation Editing**. 

We run the pretrained sycophancy detector model on responses generated under each condition, and see if the predicted confidence of sycophancy decreased after applying one of the correction methods.
![correction_experiment](https://github.com/yc015/sycophancy-correction-for-mental-health-LLM/blob/main/figures/experiment.png)

### Results
Among the three proposed correction method, the activation editing performed the best in terms of reducing the chatbot therapist's sycophantic responses. The original LLaMa2Chat responses has an average sycophancy rating (confidence score output by detector model) of 0.138. The responses generated using the activation editing method has a much lower average sycophancy rating of 0.002. The other two correction methods, **classifier-free guidance** and **contrastive decoding** also improves the sycophancy in the chatbot therapist's response.

<p align="center">
  <img src="https://github.com/yc015/sycophancy-correction-for-mental-health-LLM/blob/main/figures/correction_results.png" alt="correction_result" width=600px/>
</p>
