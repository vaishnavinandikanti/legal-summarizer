from rouge_score import rouge_scorer

def calculate_metrics(reference_summary, generated_summary):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference_summary, generated_summary)
    
    return {
        "ROUGE-1": scores['rouge1'].fmeasure,
        "ROUGE-L": scores['rougeL'].fmeasure
    }

# Example Usage:
# ref = "The court directed the respondents to release arrear pension and gratuity to Smt. Rashmi Rekha Saikia."
# gen = "Petitioner prays for release of arrear pension and gratuity..."
# print(calculate_metrics(ref, gen)