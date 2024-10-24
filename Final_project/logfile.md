# Logfile group BCI

## Week 1
_Meeting 1 (11-Sept-2024)_

Kick-off meeting with supervisors. Explaining the original project as described in the forum. 

_Work_

Reading papers about the project, getting acquinted with the code the supervisors provided. Difficulties with understanding the concepts. 

## Week 2
_Meeting 2 (17-Sept-2024)_

Re-organized the project into a more suitable starting point given our backgrounds. Lowering the expectations of the output. The project as a whole was too much of a challenge to begin with. Starting from an exising EEG data set with 3 different emotions to train different models. From there, the project can be building up towards a more challenging project. 

_Work_

Reading and running the notebook with the EEG data and understanding the code and concepts (preprocessing the data, training a model with 3 different emotions). This went well as a new starting point. We trained different models, ending up with a NN gaining the highest accuracy. 

## Week 3
_Meeting 3 (24-Sept-2024)_

Expanding the code with a diffusion model (Stable Diffusion from Hugging Face). Turning predictions of the model into an image (a bottle with positive or negative emotions).

_Work_

With our saved trained model, we could access this again and make single predictions from the data. We were gaining some more understanding of working with the PyTorch library. With the predictions of the model we made use of Stable Diffusion to create an image for us. Running this in CPU was extremely slow on our computers (to generate 1 image the duration was 30 mins). Running with CUDA was more effective (10 mins for 1 image) but still slow. We tried running the code on Google Colab and with this runtime we gained a much faster processing time. We were able to get different pictures out of the model. 


## Week 4
_Meeting 4 (01-Oct-2024)_

Expanding the project some more since our progress was very fast for this week. Adjusting and expanding different parts of the project. Finding a bigger dataset with more than 2 emotions, change or choose a different model that we are using, expanding the predictions of the model with word relations into a broader definition of the emotions (with help of Llama2). And generate an image based on that. 

_Work_

We had some difficulties finding a good accesible dataset with more than 2 emotions in it. But we managed to find a dataset with two categories (valence and arousal), which could be splitted into 4 segments (low/high valence and low/high arousal) to obtain 4 different categories of emotions (happy, calm, bored, angry). The code itself was a bit more difficult and unstructured as well which made it more difficult to understand. The final model that was trained in this code also got a very low accuracy. On the other hand, we had some good results with using Llama 2 for text generation, although it seems not possible to generate images with Llama2 itself and combining it with Stable Diffusion in one go is using to much RAM. So if we are proceeding with the combination of the two models we have to split this into two sessions in Colab. 

## Week 5
_Meeting 5 (08-Oct-2024)_

Discussion points 
- Difficulty to find free and good dataset with more than 2 emotions
- Notebook corresponding to the dataset that we found is a bit unstructured but we got it running
- We managed to train the model, although the accuracy is not very high - first predictions are not that good - do we still have to use this? 
    - The dimensions of the data are very different than our previously trained model 
- Llama2 is not generating images, only text. We got the text word relations working. Combination of this and Stable diffusion works in seperate notebooks. 

For the following weeks:
- either clean the code that we already have to only what we need OR get another dataset and model to have different emotions based on Valence/Arousal
- get predictions of the model
- put that prediction in Llama2 to get list of related words
- From the list of words
    - put this into stable diffusion to create image out of the related words of the emotion
    - Put the list of words back in LLama2 to create a conversation about these related words

# Week 6
_Midterm presentation (18-Oct-2024)_

Feedback
- Lack of clear problem statement 
- Lack of ethics consideration 
- Choose 1 data set and model, not different datasets to compare with different models

# Week 7
_Meeting 6 (22-Oct-2024)_

We discussed a clear plan for the coming 3 weeks. The outline of the project will be as follows: 

**Problem statement**: Can EEG be used as input for generative models such as text and image models to generate emotion-guided/ emotion-sensitive outputs? 

**Hypothesis**: Emotion-sensitive outputs can be created with a generative model using EEG input. 

**Application**: We use design as a tool to test this hypothesis, because emotions are involved in opinions about design. The project will be a proof of concept to test if this can be applied. 

1. Read dataset DEAP (EEG dataset with corresponding valence, arousal labels)
2. Regression model to get arousal and valence scales
3. Get two predictions with varying arousal and valence scores 
4. From [this outline on empathetic prompting ](https://arteliers.notion.site/Empathetic-Prompting-POC-12749f16dd2980cda39cf574e3a0dca0), use the basline/ neutral prompt. 
5. Create 3 different images with Stable Diffusion 
    - One baseline image (baseline prompt)
    - One that can be categorized in the "positive" field of valence/arousal 
    - One that can be categorized in the "negative" field of valence/arousal 
6. Create 3 different design briefs corresponding to the 3 categories  
7. Test with a sentiment analysis tool to test if the design briefs correctly represent the emotion specified 
8. Test with a user study how the images and descriptions are classified
9. Compare these results to the actual EEG data labels and test if they are the same (statistical analysis)

Conclusion will provide insight if EEG data predictions from the ML model can be a valuable asset for generative models to incorporate emotion into the translation to perception. 

**Experimental design** 
- Include different theories from crowdsourcing design 
- Inlcude emotion scale categories (maybe some grid)
- Analysis: 
    - check if they differ from the baseline 
    - check if they significantly differ from original labels

