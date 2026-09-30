import argparse
from .core import compress_prompt

def main():
    parser = argparse.ArgumentParser(description='Compress prompts based on budget and task type.')
    parser.add_argument('--model', required=True, help='Model identifier')
    parser.add_argument('--prompt', required=True, help='Input prompt text')
    parser.add_argument('--budget', type=float, required=True, help='Budget between 0.0 and 1.0')
    parser.add_argument('--task-type', choices=['code', 'prose', 'conversation'], required=True, help='Type of task')
    
    args = parser.parse_args()
    
    if not 0.0 <= args.budget <= 1.0:
        parser.error("Budget must be between 0.0 and 1.0")
    
    result = compress_prompt(args.model, args.prompt, args.budget, args.task_type)
    print(result)

if __name__ == '__main__':
    main()
