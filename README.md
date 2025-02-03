# Project Senior

## Overview

Project Senior is an AI-powered document and code analysis tool. It leverages advanced natural language processing (NLP) and machine learning models to analyze and extract insights from various types of documents and code files.

## Features

- **Document and Code Analysis**: Supports a wide range of file formats including PDF, TXT, Python, JavaScript, C/C++, Java, HTML, Markdown, CSV, Excel, and PowerPoint.
- **Customizable UI**: Allows users to customize the UI with different color schemes and font settings.
- **Chunking and Embedding**: Uses semantic chunking and embeddings to process and analyze documents.
- **Retrieval-based QA**: Provides a question-answering system that retrieves relevant information from the uploaded documents.

## Installation

1. Clone the repository:
    ```sh
    git clone <repository-url>
    cd <repository-directory>
    ```

2. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

1. Run the Streamlit application:
    ```sh
    streamlit run main.py
    ```

2. Upload your document or code file using the file uploader.

3. Ask questions related to the uploaded file and get insightful responses.

## Supported File Formats

- **Documents**: PDF, TXT
- **Code Files**: Python, JavaScript, C/C++, Java, HTML, Markdown, C#, PHP, Ruby, Go
- **Data Formats**: CSV, Excel (XLS, XLSX)
- **Presentations**: PowerPoint (PPTX)

## Customization

You can customize the UI settings such as chunk size, temperature, and top-p using the sidebar settings.

## License

This project is licensed under the MIT License.