# 🌞 TinySolar-248m-4k Fine-tuning Project

## 🚀 Project Overview

This project aims to fine-tune the TinySolar-248m-4k model on various datasets using advanced fine-tuning techniques. The goal is to create efficient models for different tasks while exploring various fine-tuning methodologies.

## 📂 Project Structure

```
.
├── code_feedback_fine_tuned_model_output/       # Output directory for code feedback fine-tuned model
├── code_feedback_lora_fine_tuned_model_output/  # Output directory for LoRA code feedback fine-tuned model
├── codefeedback_lora_finetuning.py              # Script for LoRA fine-tuning on code feedback
├── fine-tune-codefeedback.py                    # Script for fine-tuning on code feedback
├── fine_tuned_tinysolar_codealpaca/             # Output directory for CodeAlpaca fine-tuned model
├── finetune.py                                  # Main fine-tuning script
├── inference.ipynb                              # Notebook for testing the fine-tuned models
├── qlora_codefeedback.py                        # Script for QLoRA fine-tuning on code feedback
├── quantized_lora_fine_tuned_model_output/      # Output directory for quantized LoRA fine-tuned model
├── README.md                                    # This file
└── wandb/                                       # Weights & Biases logging directory
```

## 🏗️ Current Status

This project is actively progressing. We are implementing and experimenting with different fine-tuning techniques to optimize the performance of the TinySolar-248m-4k model on various tasks, including code generation and code feedback.

## 🧪 Implemented Fine-tuning Techniques

We have implemented and are comparing the following fine-tuning methods:

1. 🔧 Standard Fine-tuning
2. 🔬 LoRA (Low-Rank Adaptation)
3. 🧮 QLoRA (Quantized Low-Rank Adaptation)

Each of these techniques has been implemented and is being evaluated to determine the most effective approach for fine-tuning the TinySolar model on different datasets.

## 🛠️ Usage

### Fine-tuning

To run the various fine-tuning processes:

```bash
python finetune.py                      # For standard fine-tuning
python codefeedback_lora_finetuning.py  # For LoRA fine-tuning on code feedback
python qlora_codefeedback.py            # For QLoRA fine-tuning on code feedback
```

### Inference

To test the fine-tuned models, use the `inference.ipynb` notebook. This notebook can be used to evaluate the performance of different fine-tuned model versions.

## 📈 Monitoring

We use Weights & Biases (wandb) for experiment tracking and visualization. Check the `wandb/` directory for logs and the wandb dashboard for detailed metrics and comparisons between different fine-tuning approaches.

## 🔮 Future Work

- Implement additional fine-tuning techniques (e.g., GLoRA, DoRA)
- Compare and analyze the performance of each technique across different tasks
- Optimize hyperparameters for the best-performing methods
- Develop a comprehensive evaluation framework for various tasks (code generation, code feedback, etc.)

## 🤝 Contributing

As this project is actively progressing, contributions are welcome. Please open an issue to discuss proposed changes or improvements before submitting a pull request.

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

🌟 This project is actively evolving. Check back for updates on our progress with different fine-tuning techniques and their applications to various tasks!