from utils import TextGeneator

if __name__ == "__main__":
    model_path = "model/trained_v0.0.1.pth"
    generator = TextGeneator(model_path)
    while True:
        try:
            generate_tokens = input("No of tokens to  generate: ")
            generator.generate(token_len=int(generate_tokens))
        except KeyboardInterrupt:
            break
