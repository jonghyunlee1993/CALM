import os
from transformers import pipeline
import torch
import pandas as pd
from tqdm import tqdm
import argparse

def get_prompt(report):
    prompt = f"""
    Read the given pathology report, and answer the below questions.

    **Instruction**
    - Use the information in the pathology report to answer each question.
    - Provide a concise but detailed answer (full sentences or semicolon-separated).
    - Do not include information that is not present in the report.
    - If the report does not contain information to answer a question, state "NA".
    - Combine all answers into a single structured paragraph, in the order of the questions.
    - Maximum length: up to ~512 tokens.
    - Follow the format of the example outputs.

    **CORE QUESTIONS** 
    1. What is the cancer type and primary site? 
    2. What is the tumor size (all dimensions if provided)? 
    3. What is the histologic differentiation grade or WHO/Gleason grade? 
    4. Is there evidence of invasion into adjacent tissues, organs, or specific structures? 
    5. Are there features of lymphovascular invasion (LVI) or perineural invasion (PNI)? 
    6. Are the surgical margins involved by tumor? If so, which margins? 
    7. How many lymph nodes were examined, and how many are positive for metastasis? 
    8. Is there extracapsular extension in lymph node metastases? 
    9. Is there evidence of distant metastasis? 
    10. What is the pathological AJCC stage (T, N, M, and overall stage group)? 

    **OPTIONAL QUESTIONS** 
    11. Are relevant immunohistochemistry or biomarker results reported (p16, HPV, EBV, ER, PR, HER2, Ki-67, PD-L1, MSI/MMR, p53, IDH1/2, MGMT, etc.)? 
    12. Are molecular test results reported (EGFR, KRAS, BRAF, ALK, ROS1, NTRK, TP53, etc.)? Include variants if present. 
    13. Are mitotic activity, necrosis, or proliferative indices (e.g., Ki-67, MIB-1) described? 
    14. Is there mention of treatment effect, regression score, or therapy response? 
    15. Any other clinically relevant remarks?     

    **Example output 1**     
    The cancer is a tubular adenocarcinoma of the stomach, intestinal type (Lauren classification), grade 3. The tumor measures 5.5 × 4.5 × 2.4 cm on the greater curvature within a resected stomach specimen of 18.5 × 16 cm and omentum 30 × 16 cm. It invades up to the peritoneal surface of the stomach but not into the omentum or adjacent organs. LVI and PNI are not described. Surgical margins, including proximal and distal, are negative. Three lymph node stations (lesser curvature, greater curvature, peripyloric) show only reactive lymphadenitis without metastasis (0 positive; exact count NA). Extracapsular extension not reported. No distant metastasis described. Pathological stage is pT3 pN0; M not reported. No biomarkers or molecular tests are mentioned. Chronic gastritis with focal intestinal metaplasia is noted in adjacent mucosa.  

    **Example output 2**  
    The cancer is a hepatocellular carcinoma of the liver (segments 4B and 5), solitary, measuring 3.4 cm. It is poorly differentiated (grade III) with sclerotic stroma. The carcinoma invades into the subserosa of the gallbladder but does not extend into the gallbladder wall. Perineural invasion is present; lymphatic and venous invasion are absent. The hepatic parenchymal margin is free of carcinoma, while the bile duct margin cannot be assessed. One periportal lymph node was examined and is negative for metastasis (0/1). Extracapsular extension is not reported. Distant metastasis cannot be assessed (pMX). The pathological AJCC stage is pT1 pN0 pMX. No immunohistochemistry, biomarker, or molecular test results are reported. Mitotic activity, necrosis, and proliferative indices are not described. No treatment effect or regression score is mentioned. Additional findings include a bile duct hamartoma (segment 3) and benign hepatic parenchyma in segment 2. Overall, this represents a solitary, poorly differentiated hepatocellular carcinoma with perineural invasion, negative lymph nodes, and negative parenchymal margin.

    **Example output 3**
    The cancer is an invasive ductal carcinoma of the left breast, measuring 3.5 × 3 × 3 cm in the upper inner quadrant. It is histologically grade 3 (NHG3; 3+3+3/36 mitoses/10 HPF). The tumor is located 0.2 cm from the deep base margin and 0.8 cm from the skin; invasive carcinoma is noted 0.1 cm from the base, but margins are not reported as overtly involved. It invades into adjacent breast tissue but no skeletal muscle involvement is described. Perineural and lymphovascular invasion are not mentioned (NA). Axillary lymph node dissection retrieved 18 nodes, with 3 positive for metastasis (3/18). Extracapsular extension is not reported. No distant metastases are documented. The pathological AJCC stage is pT2 pN1a; M category not stated. Immunohistochemistry shows ER-negative, PR-negative, and HER2-negative (score 0), consistent with a triple-negative phenotype. No molecular testing, proliferative indices beyond mitotic count, or treatment effect are described. Additional findings include fibrocystic mastopathy in background breast tissue. Overall, this represents a triple-negative, high-grade invasive ductal carcinoma of the left breast with nodal metastases (3/18) and very close but uninvolved margins.
            
    **PATHOLOGY REPORT**
    {report}
    
    **ANSWER**
    [YOUR ANSWER HERE]
    """

    return prompt

model_id = "meta-llama/Meta-Llama-3.1-70B-Instruct"

pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

df_report = pd.read_csv("/project/kimlab_tcga/JH_workspace/multimodality_prognosis_prediction/CALM/data/pathology_report/TCGA_Reports_prepared.csv")

if "structural_summary_long_context" not in df_report.columns:
    df_report.loc[:, "structural_summary_long_context"] = None

df_report.loc[:, "structural_summary_long_context"] = None

for i, row in tqdm(df_report, total=len(df_report)):
    if pd.isna(row['structural_summary_long_context']):
        report = row["text"]
        prompt = get_prompt(report)
        
        messages = [
            {"role": "user", "content": prompt},
        ]

        outputs = pipe(
            messages,
            max_new_tokens=550,
        )
        
        df_report.at[i, "structural_summary_long_context"] = outputs[0]["generated_text"][-1]['content']

    if i > 0 and i % 100 == 0:
        df_report.to_csv(f"/project/kimlab_tcga/JH_workspace/multimodality_prognosis_prediction/CALM/data/pathology_report/structured_summary.csv", index=False)
        
df_report.to_csv(f"/project/kimlab_tcga/JH_workspace/multimodality_prognosis_prediction/CALM/data/pathology_report/structured_summary.csv", index=False)