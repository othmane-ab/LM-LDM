

## Readme for the Python script
This make_visual_details script is designed to:

Read data from a CSV file where each row represents an image path and a corresponding caption.
Create sliding windows over this data with a certain window size and stride. The number of sliding windows is also specified.
Construct prompts from each window by enumerating over the window's captions and prepending a prompt base to them.
Send each constructed prompt to the OpenAI API, which is expected to return a CSV response.
Append all the returned CSVs together and write them to an output CSV file.


## Requirements
To run this script, you need the following installed:

Python 3.6 or higher
pandas
requests
openai


Also, you need to have an OpenAI API key. In the script, you should replace the placeholder with your actual OpenAI API key.

openai.api_key  = "your-openai-api-key-here"

