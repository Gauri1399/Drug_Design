# Install required libraries (uncomment if running for the first time)
# pip install transformers
# pip install torch

from transformers import pipeline

# Load a pre-trained NLP model for summarization
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

# Abstract text from the paper (Sun et al.)
text = """
Ninety percent of clinical drug development fails despite implementation of many successful strategies, which raised the question whether certain aspects in target validation and drug optimization are overlooked? Current drug optimization overly emphasizes potency/specificity using structure‒activity-relationship (SAR) but overlooks tissue exposure/selectivity in disease/normal tissues using structure‒tissue exposure/selectivity–relationship (STR), which may mislead the drug candidate selection and impact the balance of clinical dose/efficacy/toxicity.
We propose structure‒tissue exposure/selectivity–activity relationship (STAR) to improve drug optimization, which classifies drug candidates based on drug's potency/selectivity, tissue exposure/selectivity, and required dose for balancing clinical efficacy/toxicity.
Class I drugs have high specificity/potency and high tissue exposure/selectivity, which needs low dose to achieve superior clinical efficacy/safety with high success rate. 
Class II drugs have high specificity/potency and low tissue exposure/selectivity, which requires high dose to achieve clinical efficacy with high toxicity and needs to be cautiously evaluated. 
Class III drugs have relatively low (adequate) specificity/potency but high tissue exposure/selectivity, which requires low dose to achieve clinical efficacy with manageable toxicity but are often overlooked. 
Class IV drugs have low specificity/potency and low tissue exposure/selectivity, which achieves inadequate efficacy/safety, and should be terminated early. 
STAR may improve drug optimization and clinical studies for the success of clinical drug development.
"""

# Summarize the abstract
summary = summarizer(text, max_length=200, min_length=20, do_sample=False)

# Output the summary
print("Summary of the abstract:")
print(summary[0]['summary_text'])
