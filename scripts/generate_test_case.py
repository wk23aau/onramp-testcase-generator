import torch
from transformers import T5ForConditionalGeneration, T5TokenizerFast

def generate_test_case(prompt, model, tokenizer, max_length=512, num_beams=5):
    # Encode the prompt into input IDs
    input_ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length).input_ids
    # Generate output text using beam search
    output_ids = model.generate(input_ids, max_length=max_length, num_beams=num_beams, early_stopping=True)
    # Decode and return the generated text
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return generated_text

def main():
    # Load the fine-tuned model and tokenizer
    model_path = "models/t5_testcase_generator"
    tokenizer = T5TokenizerFast.from_pretrained(model_path)
    model = T5ForConditionalGeneration.from_pretrained(model_path)
    
    # Option 1: Use a hard-coded sample prompt
    sample_prompt = (
        "Test Case Details:\n"
        "Title: EM - Offices - Delete Office\n"
        "Area Path: SG\\Elections\\Java\\EMS\n"
        "State: Ready\n\n"
        "Based on the above, generate a detailed, step-by-step test case in the following format:\n"
        "Step 1: <Step Action> -> Expected: <Step Expected>\n"
        "Step 2: <Step Action> -> Expected: <Step Expected>\n"
        "...\n\n"
        "Test Steps:"
    )
    
    # Option 2: Allow the user to input a custom prompt
    user_input = input("Enter a custom prompt (or press Enter to use the sample prompt): ")
    prompt = user_input if user_input.strip() else sample_prompt
    
    print("\nInput Prompt:")
    print(prompt)
    
    # Generate the test case
    generated = generate_test_case(prompt, model, tokenizer)
    print("\nGenerated Test Case:")
    print(generated)

if __name__ == "__main__":
    main()
