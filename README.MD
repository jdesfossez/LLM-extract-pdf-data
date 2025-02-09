# Extract structured data from PDFs

This tool takes a directory as input, finds a list of PDF files and extracts
structured data from each.

The user has to pass:
* `--dir`: directory that contains the PDFs, can be a deep hierarchy
* `--base-prompt-file`: a text file that contains the prompt
* `--fields`: the list of fields expected to be extracted by the LLM
* `--url`: OpenAI compatible API URL, defaults to [Lambdalabs](https://api.lambdalabs.com/docs) API
* `--token`: a token for the API
* `--model`: the model to use, defaults to `llama3.3-70b-instruct-fp8` because it worked for the author, the list of models is available with `--list-models`

The prompt file has to specify the instructions for the LLM, which fields to extract.
This one seems to work nicely.

Example prompt:
```
In this document, extract the following information:
- Date of the tests
- Address of the test location
- Municipality of the test location
- Postal code of the test location
- Applicant’s phone number
- Applicant’s email
- Owner’s phone number if available, otherwise “missing”
- Owner’s email if available, otherwise “missing”
- Type of system

Present the result in JSON format as follows:
{"date": <date>, "adresse": <address>, "commune": <municipality>, "postal code": <postal code>, "applicant phone": <applicant's phone>, "applicant mail": <applicant's email>, "owner phone": <owner's phone>, "owner mail": <owner's email>, "system": <system type>}"
```

The output is a CSV file that contains all of those fields.
A state file is kept so the script can be stopped/re-run and it will start where it left of.
An error file is also kept for files that failed to process.
