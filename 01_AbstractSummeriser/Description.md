## Summary

**Paper Abstract Summarizer** is a lightweight Python tool that uses pre-trained NLP transformer models to generate concise summaries of scientific paper abstracts. Built with Hugging Face Transformers, the tool reduces lengthy abstracts into shorter summaries while preserving key information, making it easier to quickly review and compare research papers.

## Features

* Generates concise summaries from scientific paper abstracts
* Uses pre-trained transformer models such as DistilBART for abstractive summarization
* Supports adjustable summary length through a configurable word-count parameter
* Provides a simple Python interface for integration into research and literature-review workflows
* Helps streamline literature screening and extraction of key findings

## How It Works

1. Loads a pre-trained transformer-based summarization model from Hugging Face.
2. Takes a scientific abstract as input and processes the text using the model.
3. Generates an abstractive summary that condenses the main information from the original abstract.
4. Allows the desired summary length to be adjusted using the `word_count` parameter.
5. Users can replace the input text with any abstract they want to summarize.

## Technologies

* Python
* Hugging Face Transformers
* Pre-trained NLP models
* Natural Language Processing
