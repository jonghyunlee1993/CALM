import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

from transformers import pipeline
import torch
import torch
import pandas as pd
from tqdm import tqdm

def get_prompt(report):
    prompt = f"""
    Read the given pathology report, and answer the below question. 

    **Instruction** 
    The maximum length of the answer is 100 words. 
    Use the information in the pathology report to answer the questions. 
    Provide a concise answer that directly addresses each question. 
    Do not include any information that is not present in the report. 
    If the report does not contain information to answer a question, ignore that question. 
    Simply answer the questions without any additional explanation or context. 
    Answer must be written as a single concise paragraph. 
    ** Follow the format of the example output. ** 

    **QUESTIONS** 
    - What is the tumor size? Describe it with concrete numbers if provided and explain it. 
    - What is the differentiation of the lesion? 
    - Is there any evidence of the lesion invading adjacent tissues or organs? 
    - Any evidence of lymph node metastasis. This indicates whether the cancer has spread to the lymph nodes. 
    - Are the margins of the excised tissue clear of disease? 

    **Example output 1** 
    The tumor in the left upper lobe of the lung has a most significant dimension of 3.5 cm. Moderately to Poorly differentiated, with a poorly differentiated component representing less than 50% of the tumor; Metastatic carcinoma was found in 3 out of 18 lymph nodes; Primary tumor shows no direct extension or invasion of adjacent tissues; Margins are uninvolved by invasive carcinoma, indicating a negative surgical margin status (R0). 

    **Example output 2** 
    The prostate tumor is an adenocarcinoma with a dominant Gleason score of 4+4=8 and a significant Gleason 3 component, indicating poorly to moderately differentiated carcinoma. The tumor shows established extracapsular extension in the left anterior and right posterior mid regions and invades the right seminal vesicle, but the bladder neck is not involved. Lymph node dissections (right hypogastric, left pelvic, and right pelvic; total 16 nodes) show no metastatic disease. Surgical margins are free of tumor, consistent with negative margin status (R0). 

    **Example output 3** 
    The tumor in the right kidney measures 8.6 cm (8.6 × 8.3 × 6.2 cm) and is a clear cell renal cell carcinoma, Fuhrman grade 2 (moderately differentiated). The lesion is confined to the kidney without invasion into perirenal adipose tissue, Gerota’s fascia, renal sinus, renal vein, ureter, or other adjacent structures. No lymph nodes were submitted or identified, so nodal status cannot be assessed. All surgical margins, including perinephric adipose tissue, renal vein, artery, and ureter, are uninvolved by tumor, consistent with negative margins. 
        
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
    torch_dtype="auto",
    device_map="auto",
)

df_report = pd.read_csv("/project/kimlab_tcga/JH_workspace/multimodality_prognosis_prediction/CALM/data/pathology_report/TCGA_Reports_prepared.csv")

df_report.loc[:, "structured_report_short_context"] = None

for i, row in tqdm(df_report.iterrows(), total=len(df_report)):
    if pd.isna(row['structured_report_short_context']):
        report = row["text"]
        prompt = get_prompt(report)
        
        messages = [
            {"role": "user", "content": prompt},
        ]

        outputs = pipe(
            messages,
            max_new_tokens=100,
        )
        
        df_report.at[i, "structured_report_short_context"] = outputs[0]["generated_text"][-1]['content']

    if i > 0 and i % 100 == 0:
        df_report.to_csv("/project/kimlab_tcga/JH_workspace/multimodality_prognosis_prediction/CALM/data/pathology_report/structured_summary.csv", index=False)
        
df_report.to_csv("/project/kimlab_tcga/JH_workspace/multimodality_prognosis_prediction/CALM/data/pathology_report/structured_summary.csv", index=False)