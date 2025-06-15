import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
from typing import Dict, List, Optional
import gc

class CPUModelOptimizer:
    def __init__(self, model_name: str = "meta-llama/Llama-3.2-1B"):
        self.model_name = model_name
        print("Loading model on CPU...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,  # Use float32 for CPU
            device_map={"": "cpu"}  # Force CPU
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
    def prune_model(self, amount: float = 0.3) -> None:
        """
        Apply magnitude pruning to the model
        
        Args:
            amount: Percentage of weights to prune (0.0 to 1.0)
        """
        print(f"Applying {amount*100}% pruning...")
        
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                prune.l1_unstructured(module, name='weight', amount=amount)
                prune.remove(module, 'weight')  # Make pruning permanent
        
        # Clear memory
        gc.collect()
        torch.cuda.empty_cache()
        
        print("Pruning complete!")

    def dynamic_quantize(self) -> nn.Module:
        """
        Apply dynamic quantization to the model
        """
        print("Applying dynamic quantization...")
        
        # Configure quantization
        quantization_config = torch.quantization.default_dynamic_qconfig
        torch.quantization.qconfig = quantization_config
        
        # Prepare model for quantization
        model_prepared = torch.quantization.prepare(self.model, inplace=False)
        
        # Convert to quantized model
        quantized_model = torch.quantization.convert(model_prepared)
        
        # Clear memory
        gc.collect()
        torch.cuda.empty_cache()
        
        print("Quantization complete!")
        return quantized_model

    def static_quantize(self, calibration_data: Optional[List[str]] = None) -> nn.Module:
        """
        Apply static quantization to the model
        
        Args:
            calibration_data: List of text samples for calibration
        """
        if calibration_data is None:
            calibration_data = [
                "Once upon a time",
                "The quick brown fox",
                "In a galaxy far far away",
                "Write a function that"
            ]
        
        print("Applying static quantization...")
        
        # Configure static quantization
        self.model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        
        # Prepare model for quantization
        model_prepared = torch.quantization.prepare(self.model)
        
        # Calibrate with sample data
        print("Calibrating...")
        for text in calibration_data:
            inputs = self.tokenizer(text, return_tensors="pt")
            with torch.no_grad():
                model_prepared(**inputs)
        
        # Convert to quantized model
        quantized_model = torch.quantization.convert(model_prepared)
        
        # Clear memory
        gc.collect()
        torch.cuda.empty_cache()
        
        print("Static quantization complete!")
        return quantized_model

    def evaluate_model(self, 
                      model: nn.Module, 
                      test_inputs: Optional[List[str]] = None) -> List[Dict[str, str]]:
        """
        Evaluate model performance
        
        Args:
            model: Model to evaluate
            test_inputs: List of test prompts
        """
        if test_inputs is None:
            test_inputs = [
                "Explain quantum computing:",
                "Write a Python function to:",
                "What is machine learning?"
            ]
        
        results = []
        for text in test_inputs:
            inputs = self.tokenizer(text, return_tensors="pt")
            
            with torch.no_grad():
                outputs = model.generate(
                    inputs["input_ids"],
                    max_length=50,
                    num_return_sequences=1,
                    temperature=0.7
                )
            
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            results.append({
                "prompt": text,
                "response": generated_text
            })
        
        return results

    def get_model_size(self, model: nn.Module) -> float:
        """Calculate model size in MB"""
        return sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)

    def save_model(self, 
                  model: nn.Module, 
                  path: str,
                  save_tokenizer: bool = True) -> None:
        """Save the optimized model"""
        os.makedirs(path, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(path, "model.pt"))
        if save_tokenizer:
            self.tokenizer.save_pretrained(path)

def main():
    # Initialize optimizer
    optimizer = CPUModelOptimizer()
    
    # Get original size
    original_size = optimizer.get_model_size(optimizer.model)
    print(f"\nOriginal model size: {original_size:.2f} MB")
    
    # Prune model
    optimizer.prune_model(amount=0.3)
    pruned_size = optimizer.get_model_size(optimizer.model)
    print(f"Size after pruning: {pruned_size:.2f} MB")
    
    # Apply dynamic quantization
    dynamic_quantized = optimizer.dynamic_quantize()
    dynamic_size = optimizer.get_model_size(dynamic_quantized)
    print(f"Size after dynamic quantization: {dynamic_size:.2f} MB")
    
    # Apply static quantization
    static_quantized = optimizer.static_quantize()
    static_size = optimizer.get_model_size(static_quantized)
    print(f"Size after static quantization: {static_size:.2f} MB")
    
    # Evaluate models
    print("\nEvaluating models...")
    
    test_inputs = [
        "Write a simple function to calculate fibonacci numbers:",
        "Explain how neural networks work:"
    ]
    
    print("\nPruned Model Output:")
    pruned_results = optimizer.evaluate_model(optimizer.model, test_inputs)
    for result in pruned_results:
        print(f"\nPrompt: {result['prompt']}")
        print(f"Response: {result['response']}")
    
    print("\nDynamic Quantized Model Output:")
    dynamic_results = optimizer.evaluate_model(dynamic_quantized, test_inputs)
    for result in dynamic_results:
        print(f"\nPrompt: {result['prompt']}")
        print(f"Response: {result['response']}")
    
    print("\nStatic Quantized Model Output:")
    static_results = optimizer.evaluate_model(static_quantized, test_inputs)
    for result in static_results:
        print(f"\nPrompt: {result['prompt']}")
        print(f"Response: {result['response']}")
    
    # Save models
    optimizer.save_model(optimizer.model, "pruned_model")
    optimizer.save_model(dynamic_quantized, "dynamic_quantized_model")
    optimizer.save_model(static_quantized, "static_quantized_model")

if __name__ == "__main__":
    main()