# Replication Pack  
*Retrospective Analysis of Automating Agile User Story Quality Evaluation*

#### Steps to Reproduce

Prerequisites: Ensure you have Python `3.11.7` installed. 

1. Install Required Libraries

Run the following command to install all necessary libraries:

```bash
pip install -r requirements.txt
```

2. Extract the Dataset

After extracting the `dataset.tar`, ensure the directory structure is as follows:

```
dataset/
  ├── metric
  ├── llama
  └── codeqwen
```

3. Run the Benchmark

To benchmark the ML models, execute the following command:

```bash
python run_ml_models.py -i dataset/metric -e results/metric -d Porru_Dataset
```

Adjust the parameters as needed. 

If you encounter any problems, feel free to *open an issue*.

NOTE: The experiments were conducted on Debian 11 running on WSL2 with Cuda support.


### Citation
Will be added soon.
