import numpy

def mnist_inference(data_path):
    # Placeholder for MNIST inference code
    print(f"MNIST inference function called with data path: {data_path}")
    for f in data_path:
        print(f"Processing file: {f}")

def ce_inference(data_path):
    # Placeholder for C. Elegans inference code
    print(f"C. Elegans inference function called with data path: {data_path}")
    for f in data_path:
        print(f"Processing file: {f}")

def main():
    print("Would you like to test the MNIST model or C. Elegans Model? Enter 'M' or 'CE':")
    model_choice = input().strip().upper()
    while model_choice not in ['M', 'CE']:
        print("Invalid choice. Please enter 'M' for MNIST or 'CE' for C. Elegans:")
        model_choice = input().strip().upper()
        
    print("Please enter the absolute path to the folder for inference data:")
    data_path = input().strip()
    
    if model_choice == 'M':
        mnist_inference(data_path)
    elif model_choice == 'CE':
        ce_inference(data_path)
            
if __name__ == "__main__":
    main()