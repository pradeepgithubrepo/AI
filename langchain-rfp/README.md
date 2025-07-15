# langchain-rfp

## Overview
The `langchain-rfp` project is designed to process Request for Proposal (RFP) documents using advanced text extraction and analysis techniques. It leverages Azure OpenAI for natural language processing to extract relevant information from various document formats.

## Project Structure
```
langchain-rfp
├── src
│   └── rfp.py          # Main logic for processing RFP documents
├── requirements.txt    # List of dependencies
├── README.md           # Project documentation
└── venv/               # Virtual environment for dependency isolation
```

## Setup Instructions

1. **Create a Virtual Environment**
   Navigate to the project directory and create a virtual environment:
   ```
   python -m venv venv
   ```

2. **Activate the Virtual Environment**
   - On Windows:
     ```
     venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```
     source venv/bin/activate
     ```

3. **Install Dependencies**
   Install the required packages using pip:
   ```
   pip install -r requirements.txt
   ```

## Usage
To run the application, execute the following command:
```
python src/rfp.py
```

Follow the prompts to process RFP documents and interact with the application.

## Contributing
Contributions are welcome! Please submit a pull request or open an issue for any enhancements or bug fixes.

## License
This project is licensed under the MIT License. See the LICENSE file for details.