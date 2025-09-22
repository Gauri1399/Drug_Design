{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": []
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "code",
      "execution_count": 4,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "xH4DXKWfCgg-",
        "outputId": "11690e87-8c71-42e4-a60c-4227f5e9a924"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Requirement already satisfied: transformers in /usr/local/lib/python3.11/dist-packages (4.47.1)\n",
            "Requirement already satisfied: filelock in /usr/local/lib/python3.11/dist-packages (from transformers) (3.17.0)\n",
            "Requirement already satisfied: huggingface-hub<1.0,>=0.24.0 in /usr/local/lib/python3.11/dist-packages (from transformers) (0.27.1)\n",
            "Requirement already satisfied: numpy>=1.17 in /usr/local/lib/python3.11/dist-packages (from transformers) (1.26.4)\n",
            "Requirement already satisfied: packaging>=20.0 in /usr/local/lib/python3.11/dist-packages (from transformers) (24.2)\n",
            "Requirement already satisfied: pyyaml>=5.1 in /usr/local/lib/python3.11/dist-packages (from transformers) (6.0.2)\n",
            "Requirement already satisfied: regex!=2019.12.17 in /usr/local/lib/python3.11/dist-packages (from transformers) (2024.11.6)\n",
            "Requirement already satisfied: requests in /usr/local/lib/python3.11/dist-packages (from transformers) (2.32.3)\n",
            "Requirement already satisfied: tokenizers<0.22,>=0.21 in /usr/local/lib/python3.11/dist-packages (from transformers) (0.21.0)\n",
            "Requirement already satisfied: safetensors>=0.4.1 in /usr/local/lib/python3.11/dist-packages (from transformers) (0.5.2)\n",
            "Requirement already satisfied: tqdm>=4.27 in /usr/local/lib/python3.11/dist-packages (from transformers) (4.67.1)\n",
            "Requirement already satisfied: fsspec>=2023.5.0 in /usr/local/lib/python3.11/dist-packages (from huggingface-hub<1.0,>=0.24.0->transformers) (2024.10.0)\n",
            "Requirement already satisfied: typing-extensions>=3.7.4.3 in /usr/local/lib/python3.11/dist-packages (from huggingface-hub<1.0,>=0.24.0->transformers) (4.12.2)\n",
            "Requirement already satisfied: charset-normalizer<4,>=2 in /usr/local/lib/python3.11/dist-packages (from requests->transformers) (3.4.1)\n",
            "Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.11/dist-packages (from requests->transformers) (3.10)\n",
            "Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.11/dist-packages (from requests->transformers) (2.3.0)\n",
            "Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.11/dist-packages (from requests->transformers) (2024.12.14)\n"
          ]
        }
      ],
      "source": [
        "pip install transformers"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "from transformers import pipeline\n",
        "\n",
        "# Load a pre-trained NLP model for summarization\n",
        "summarizer = pipeline(\"summarization\")\n",
        "\n",
        "# The abstract text provided for the paper from Sun et. al.\n",
        "text = \"\"\"\n",
        "Ninety percent of clinical drug development fails despite implementation of many successful strategies, which raised the question whether certain aspects in target validation and drug optimization are overlooked? Current drug optimization overly emphasizes potency/specificity using structure‒activity-relationship\n",
        "(SAR) but overlooks tissue exposure/selectivity in disease/normal tissues using structure‒tissue exposure/selectivity–relationship (STR),\n",
        "which may mislead the drug candidate selection and impact the balance of clinical dose/efficacy/toxicity.\n",
        "We propose structure‒tissue exposure/selectivity–activity relationship (STAR) to improve drug optimization,\n",
        "which classifies drug candidates based on drug's potency/selectivity, tissue exposure/selectivity, and required dose for balancing clinical efficacy/toxicity.\n",
        "Class I drugs have high specificity/potency and high tissue exposure/selectivity, which needs low dose to achieve superior clinical efficacy/safety with high\n",
        "success rate. Class II drugs have high specificity/potency and low tissue exposure/selectivity, which requires high dose to achieve clinical efficacy with high\n",
        "toxicity and needs to be cautiously evaluated. Class III drugs have relatively low (adequate) specificity/potency but high tissue exposure/selectivity,\n",
        "which requires low dose to achieve clinical efficacy with manageable toxicity but are often overlooked. Class IV drugs have low specificity/potency and low tissue\n",
        "exposure/selectivity, which achieves inadequate efficacy/safety, and should be terminated early. STAR may improve drug optimization and clinical studies for the\n",
        "success of clinical drug development.\n",
        "\"\"\"\n",
        "\n",
        "# Summarize the text\n",
        "summary = summarizer(text, max_length=200, min_length=20, do_sample=False)\n",
        "print(\"Summary of the failure:\")\n",
        "print(summary[0]['summary_text'])"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "0k8WF1YyCiUo",
        "outputId": "ceb0b298-03e6-4ac5-b5b3-7d35c98d6254"
      },
      "execution_count": 5,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "No model was supplied, defaulted to sshleifer/distilbart-cnn-12-6 and revision a4f8f3e (https://huggingface.co/sshleifer/distilbart-cnn-12-6).\n",
            "Using a pipeline without specifying a model name and revision in production is not recommended.\n",
            "Device set to use cpu\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Summary of the failure:\n",
            " Current drug optimization overly emphasizes potency/specificity using structure‒activity-relationship (SAR) but overlooks tissue exposure/selectivity in disease/normal tissues . Class I drugs have high specificity/potency, which needs low dose to achieve superior clinical efficacy/safety with high success rate .\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "References:\n",
        "* Sun D, Gao W, Hu H, Zhou S. Why 90% of clinical drug development fails and how to improve it? Acta Pharm Sin B. 2022 Jul;12(7):3049-3062. doi: 10.1016/j.apsb.2022.02.002. Epub 2022 Feb 11. PMID: 35865092; PMCID: PMC9293739.\n",
        "* Wolf, T., Debut, L., Sanh, V., Chaumond, J., Delangue, C., Moi, A., Cistac, P., Rault, T., Louf, R., Funtowicz, M., & Brew, J. (2020). Transformers: State-of-the-art natural language processing. Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations, 38–45. Association for Computational Linguistics. Retrieved from https://github.com/huggingface/transformers"
      ],
      "metadata": {
        "id": "WWR9vF6WIFg6"
      }
    }
  ]
}