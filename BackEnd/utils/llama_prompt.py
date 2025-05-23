import ollama
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
 
def generate_evaluation(transcript, test, job_role, skills_to_rate):
       model_name = "sshleifer/distilbart-cnn-12-6"
       tokenizer = AutoTokenizer.from_pretrained(model_name)
       model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
 
       def summarize_text(text, max_chunk_size=5000):
            chunks = [text[i:i + max_chunk_size] for i in range(0, len(text), max_chunk_size)]
            summary = ""
            for chunk in chunks:
                inputs = tokenizer.encode(chunk, return_tensors="pt", max_length=1024, truncation=True)
                summary_ids = model.generate(inputs, max_length=130, min_length=30, length_penalty=2.0, num_beams=4, early_stopping=True)
                summary += tokenizer.decode(summary_ids[0], skip_special_tokens=True) + " "
            return summary
 
       def extract_relevant_lines(text, skills):
            lines = text.splitlines()
            filtered = []
            for line in lines:
                for skill in skills:
                    if skill.lower() in line.lower():
                        filtered.append(line)
                        break
            return "\n".join(filtered)
        
        
 
       print("Summarizing the context...\n⏳ Please wait....\n")
        # Summarize & filter
       summarized_transcript = summarize_text(extract_relevant_lines(transcript, skills_to_rate))
       summarized_test = summarize_text(extract_relevant_lines(test, skills_to_rate))
       
       print("Generating mail... Please wait...\n⏳ Feeding into model...\n")
         # Generate email
       tone_instruction = ""
 
       if "senior" in job_role.lower():
            tone_instruction = "Be extremely strict and expect advanced coding skills, architectural thinking, and optimization."
       elif "junior" in job_role.lower():
            tone_instruction = "Be moderately strict. Focus more on fundamentals and learning potential."
       else:
            tone_instruction = "Be strict and unbiased. Expect solid coding logic and reasoning."
 
       prompt = f"""
            Act as a strict technical evaluator for the role of {job_role}.
            {tone_instruction}
   
     
 
            Coding Test Summary (70% weight):
            {summarized_test}
 
            Interview Transcript (30% weight):
            {summarized_transcript}
 
            Skills to Rate (only these):
            {', '.join(skills_to_rate)}
 
   
            You are provided with a coding test (weight 70%) and an interview transcript (weight 30%).
            Generate a summary to the hiring manager(DO NOT DISPLAY THIS).
            Your output must:
                - Job role: {job_role}
                -Mention the candidate's name and the date, if available.
                -Focus more on the test file: review Python code accuracy, efficiency, and SQL query logic.
                -Mention specific technical weaknesses, even minor ones.
                - Only evaluate and rate these exact skills: {', '.join(skills_to_rate)}.
                - Provide a skill-wise rating table (with score out of 5 and brief justification).
                - If a skill is not in the provided list ({skills_to_rate}), IGNORE its presence in the transcript or test.
                - Do not include additional skills or soft skills like communication unless explicitly listed.
                -Assign an overall rating (out of 5) that reflects mostly the test file based on skill-wise rating.
                -End with a strict "Hire" or "Do Not Hire" decision — no conditional wording.
                -If the candidate shows logic errors, unoptimized code, or lacks clarity in SQL — the final decision should be "Do Not Hire."
                -Ensure the rating and recommendation match. If rating < 3.0 → must be " Do Not Hire."
            Output only the email. Be professional and decisive.
   
            Avoid hallucinating. Only use information present in the input text.
            """
       
       response = ollama.chat(
       model='llama3:8b',
       messages=[{"role": "user", "content": prompt}],
       options={
            "temperature": 0.3,      # More focused answers
            "top_p": 0.9,            # Reduces randomness
            "num_predict": 350       # Reduce token size to increase speed
        }
    )  
       return response['message']['content']