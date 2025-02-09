#!/usr/bin/env python3

import sys
from openai import OpenAI
import csv
import argparse
import pathlib
import json

from docling.document_converter import DocumentConverter


def connect_openai(args):
    openai_api_key = args.token
    openai_api_base = args.url
    return OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", help="Directory where the PDF files are located")
    parser.add_argument(
        "--token",
        help="OpenAI token",
    )
    parser.add_argument(
        "--url", help="OpenAI URL", default="https://api.lambdalabs.com/v1"
    )
    parser.add_argument("--base-prompt-file", help="Text file with the base prompt")
    parser.add_argument(
        "--fields", help="CSV list of expected fields returned by the LLM"
    )
    parser.add_argument(
        "--list-models", help="List the available models", action="store_true"
    )
    parser.add_argument(
        "--model", help="List the available models", default="llama3.3-70b-instruct-fp8"
    )
    args = parser.parse_args()

    client = connect_openai(args)
    if args.list_models is True:
        model_list = client.models.list()
        print("Available models")
        for i in model_list:
            print(f"- {i.id}")
        sys.exit(0)

    pdf_dir = pathlib.Path(args.dir)
    if not pdf_dir.exists():
        print(f"Directory {pdf_dir} doesn't exist")
        sys.exit(1)

    if args.dir is None:
        print("Missing --dir")
        sys.exit(1)

    if args.base_prompt_file is None:
        print("Missing --base-prompt-file")
        sys.exit(1)

    return args, client


def load_state(pdf_dir):
    state_file = pdf_dir / "state.json"
    processed_filenames = []
    if state_file.exists():
        state = json.loads(state_file.read_text())
        for i in state:
            processed_filenames.append(i["filename"])
    else:
        state = []
    return state_file, state, processed_filenames


def open_pdf_dir(pdf_path):
    return pathlib.Path(pdf_path)


def list_pdfs(pdf_dir):
    all_pdfs = []
    files = pdf_dir.iterdir()
    for f in files:
        if f.suffix in [".pdf", ".PDF"]:
            all_pdfs.append(f)
    return all_pdfs


def process_text(filename, base_prompt, fields, model, client):
    print(f"Processing {filename}")
    converter = DocumentConverter()
    result = converter.convert(filename)
    doc = result.document.export_to_markdown()
    prompt = base_prompt + doc

    response = client.completions.create(
        prompt=prompt,
        temperature=0,
        model=model,
    )

    out_sanitized = response.choices[0].text.replace("```json", "").replace("```", "")

    try:
        data = json.loads(out_sanitized)
    except Exception:
        print("Failed to load LLM output as json")
        print(f"LLM reply: {response.choices[0].text}")
        return None, True

    data["filename"] = filename

    generation_error = False
    for f in fields:
        if f not in data.keys():
            print(f"Missing key {f}")
            generation_error = True

    if generation_error:
        print(f"LLM reply: {response.choices[0].text}")

    return data, generation_error


def check_failed(pdf_dir, processed_filenames, failed_files):
    state_failed = pdf_dir / "failed.json"
    if state_failed.exists():
        old_failed = json.loads(state_failed.read_text())
        failed = []
        # clear the old failures in case they are working now
        for f in old_failed:
            if f not in processed_filenames:
                failed.append(f)
    else:
        failed = []
    for f in failed_files:
        if f not in failed:
            failed.append(f)
    state_failed.write_text(json.dumps(failed))
    if len(failed) > 0:
        print(f"There were errors, the list is in {state_failed}")


def write_csv(state, fields):
    data_file = open("all.csv", "w")
    csv_writer = csv.writer(data_file)
    all_headers = ["filename"] + fields
    count = 0
    for i in state:
        if count == 0:
            csv_writer.writerow(all_headers)
            count += 1

        row = []
        for h in all_headers:
            row.append(i[h])
        csv_writer.writerow(row)
    data_file.close()


def process_all_pdfs(
    all_pdfs, base_prompt, fields, model, client, state, state_file, processed_filenames
):
    failed_files = []
    for _f in all_pdfs:
        filename = str(_f)
        if filename in processed_filenames:
            print(f"{filename} already processed, skipping")
            continue

        data, generation_error = process_text(
            filename, base_prompt, fields, model, client
        )

        if generation_error:
            failed_files.append(filename)
            continue
        state.append(data)
        processed_filenames.append(filename)
        state_file.write_text(json.dumps(state))
    return state, failed_files, processed_filenames


def main():
    args, client = parse_args()

    pdf_dir = open_pdf_dir(args.dir)
    state_file, state, processed_filenames = load_state(pdf_dir)
    all_pdfs = list_pdfs(pdf_dir)
    base_prompt = pathlib.Path(args.base_prompt_file).read_text()

    model = args.model

    fields = [s.strip() for s in args.fields.split(",")]

    print(f"Extracting fields {fields}")

    state, failed_files, processed_filenames = process_all_pdfs(
        all_pdfs,
        base_prompt,
        fields,
        model,
        client,
        state,
        state_file,
        processed_filenames,
    )

    check_failed(pdf_dir, processed_filenames, failed_files)
    write_csv(state, fields)


if __name__ == "__main__":
    main()
