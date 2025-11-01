# ReDD

Relational Deep Dive: Error-Aware Queries Over Unstructured Data

## Getting Started

### Prerequisites

Ensure you have Python 3.7 or higher installed. 
Additionally, install the required Python packages using:
```bash
bash set_up_env.sh
pip install -r requirements.txt
```
s
### Run Schema Genration

Initialize Schema Generation
```bash
python main_schemagen.py --config cfg/schemagen.yaml --init
```

Run Experiments on `spider/store_1/albums`
```bash
python main_schemagen.py --config cfg/schemagen_deepseek.yaml --exp spider_4d0_1 --api-key <Your DeepSeek api-key>
python main_schemagen.py --config cfg/schemagen_deepseek.yaml --exp spider_4d1_1 --api-key <Your DeepSeek api-key>
```

### Configuration
Create configuration files in the `cfg/` directory to tune the parameters and API settings according to your project needs.





# ReDD

**Relational Deep Dive**: Error-Aware Queries Over Unstructured Data

## Getting Started

### Prerequisites

Ensure you have Python 3.7 or higher installed.

Install required Python packages:

```bash
bash set_up_env.sh
pip install -r requirements.txt
```

---

## 🔧 Configuration

All experiments are controlled via YAML config files in the `cfg/` directory. These files contain model settings, experiment names, data sources, etc.

---

## 🏗️ Schema Generation

### Step 1: Initialize Schema Generation

```bash
python main_schemagen.py --config cfg/schemagen.yaml --init
```

### Step 2: Run Schema Experiments

```bash
python main_schemagen.py --config cfg/schemagen_deepseek.yaml --exp spider_4d0_1 --api-key <Your DeepSeek API key>
python main_schemagen.py --config cfg/schemagen_deepseek.yaml --exp spider_4d1_1 --api-key <Your DeepSeek API key>
```

---

## 📊 Data Population

Use `main_datapop.py` to run various data population methods (GPT, DeepSeek, Local).

### Step 1: Initialize Dataset (Optional)

```bash
python main_datapop.py --config cfg/datapop_cogital32b.yaml --exp college --init
```

### Step 2: Run Data Population

```bash
python main_datapop.py --config cfg/datapop_cogital32b.yaml --exp college
```

You can specify the mode (`cgpt`, `deepseek`, `ds7b`, `dsv2lite`, `cogito32b`, `cogito70b`) inside the config file.

### Optional Flags:

* `--api-key`: Needed for GPT/DeepSeek modes
* `--eval`: Run evaluation after data generation
* `--train-classifier`: Train error classifier after data generation

Example with evaluation and classifier training:

```bash
python main_datapop.py --config cfg/datapop_cogital32b.yaml --exp college --api-key <Your API key> --eval --train-classifier
```

---

## ✅ Error Correction

Use `main_correction.py` to test and ensemble error classifiers.

```bash
python main_correction.py --config cfg/datapop_cogital32b.yaml --exp college
```
