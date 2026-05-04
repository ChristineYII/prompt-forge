import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--prompt_refine_model_name", default="google:gemini-2.5-flash", help="write as provider:model_name, e.g. google:gemini-2.5-flash")
parser.add_argument("--test_case_generate_model_name", default="deepseek:deepseek-v4-flash", help="write as provider:model_name, e.g. deepseek:deepseek-chat")
parser.add_argument("--output_path", default="./output/sale_agent", help="The path to the output file")

args = parser.parse_args()
