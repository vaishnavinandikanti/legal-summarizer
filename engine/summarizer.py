import re
from transformers import pipeline
import streamlit as st
from collections import Counter


# 1. COURT & CASE PATTERNS


COURT_PATTERNS = [
    # Supreme Court
    r"IN THE (SUPREME COURT OF INDIA)",
    r"(SUPREME COURT OF INDIA)",
    r"BEFORE THE (SUPREME COURT OF INDIA)",
    
    # High Courts - Generic
    r"IN THE (HIGH COURT OF [A-Z\s]+ AT [A-Z\s]+)",
    r"IN THE (HIGH COURT OF [A-Z\s]+)",
    r"(HIGH COURT OF [A-Z\s]+ AT [A-Z\s]+)",
    r"(HIGH COURT OF [A-Z\s]+)",
    r"BEFORE THE (HIGH COURT OF [A-Z\s]+ AT [A-Z\s]+)",
    
    # Major High Courts
    r"(DELHI HIGH COURT)",
    r"(BOMBAY HIGH COURT)",
    r"(CALCUTTA HIGH COURT)",
    r"(MADRAS HIGH COURT)",
    r"(KARNATAKA HIGH COURT)",
    r"(KERALA HIGH COURT)",
    r"(GUJARAT HIGH COURT)",
    r"(ALLAHABAD HIGH COURT)",
    r"(PUNJAB AND HARYANA HIGH COURT)",
    r"(RAJASTHAN HIGH COURT)",
    r"(MADHYA PRADESH HIGH COURT)",
    r"(ANDHRA PRADESH HIGH COURT)",
    r"(TELANGANA HIGH COURT)",
    r"(ORISSA HIGH COURT)",
    r"(PATNA HIGH COURT)",
    r"(CHHATTISGARH HIGH COURT)",
    r"(JHARKHAND HIGH COURT)",
    r"(UTTARAKHAND HIGH COURT)",
    r"(HIMACHAL PRADESH HIGH COURT)",
    r"(JAMMU AND KASHMIR HIGH COURT)",
    r"(GAUHATI HIGH COURT)",
    r"(GUWAHATI HIGH COURT)",
    
    # Lower Courts
    r"(DISTRICT COURT OF [A-Z\s]+)",
    r"(DISTRICT COURT[,\s]+[A-Z\s]+)",
    r"IN THE COURT OF (DISTRICT JUDGE[,\s]+[A-Z\s]+)",
    r"(DISTRICT & SESSIONS COURT[,\s]+[A-Z\s]+)",
    r"(SESSIONS COURT[,\s]+[A-Z\s]+)",
    r"(SESSIONS COURT AT [A-Z\s]+)",
    r"(COURT OF SESSIONS JUDGE[,\s]+[A-Z\s]+)",
    r"(ADDITIONAL SESSIONS JUDGE[,\s]+[A-Z\s]+)",
    
    # Special Courts
    r"(SPECIAL COURT FOR [A-Z\s]+)",
    r"(CBI COURT[,\s]+[A-Z\s]+)",
    r"(FAMILY COURT[,\s]+[A-Z\s]+)",
    
    # Tribunals
    r"(NATIONAL GREEN TRIBUNAL)",
    r"(ARMED FORCES TRIBUNAL)",
    r"(CENTRAL ADMINISTRATIVE TRIBUNAL)",
]

CASE_NO_PATTERNS = [
    # Writ Petitions
    r"W\.?\s*P\.?\s*\(?(C|CR|Civil|Criminal|Crl\.?)\)?\s*(?:D\s+)?No\.?\s*\d+\s*(?:/|of)\s*\d{4}",
    r"WP\s*\(C\)\s*No\.?\s*\d+/\d{4}",
    r"WRIT PETITION\s*\((?:CIVIL|CRIMINAL)\)\s*NO\.?\s*\d+/\d{4}",
    r"Writ Petition Misc\.\s+Single No\.\s*\d+\s+of\s+\d{4}",
    r"WRIT\s+PETITION\s+\(CIVIL\)\s+NO\.\d+\s+OF\s+\d{4}",
    r"WRIT\s+PETITION\s+\(CRIMINAL\)\s+D\s+NO\.\d+\s+OF\s+\d{4}",
    # Appeals
    r"(?:Civil|Criminal)\s+Appeal\s*(?:No\.?)?\s*\d+\s*(?:/|of)\s*\d{4}",
    r"R/CRIMINAL APPEAL\s*\(AGAINST CONVICTION\)\s*NO\.\s*\d+\s+of\s+\d{4}",
    r"Criminal Appeal No\.\s*\d+\s+of\s+\d{4}",
    r"\bWRIT\s+PETITION\s*\(\s*(?:CIVIL|CRIMINAL)\s*\)\s*NO\.?\s*\d+\s*OF\s*\d{4}\b",
    r"\bWRIT\s+PETITION\s*\(\s*(?:CIVIL|CRIMINAL)\s*\)\s*D\s*NO\.?\s*\d+\s*OF\s*\d{4}\b",
    r"\b(?:CIVIL|CRIMINAL)\s+APPEAL\s+NO\.?\s*\d+\s*OF\s*\d{4}\b",
    r"\b(?:CIVIL|CRIMINAL)\s+APPEAL\s+NOS?\.?\s*\d+(?:\s*[-–]\s*\d+)?\s*OF\s*\d{2}\s*\d{2}\b",
    r"\bAPPEAL\s*(?:\(\s*CRL\.?\s*\))?\s*\d+\s*of\s*\d{4}\b",
    # Special Leave
    r"SLP\s*\(?(?:Civil|Crl\.?|Criminal|C|Crl)\)?\s*No\.?\s*\d+\s*(?:/|of)\s*\d{4}",
    r"SPECIAL LEAVE PETITION\s*\((?:CIVIL|CRIMINAL)\)\s*NO\.?\s*\d+/\d{4}",
    
    # Generic Case Numbers
    r"Case No\.\s*\d+\s+of\s+\d{4}",
    r"\b(?:CRLA|CRA|CWP|RSA|FAO|LPA|WA|CMP|COCP|MAT|WPC)\s*No\.?\s*\d+/\d{4}",
    
    # CNR Numbers
    r"\b\d{4}\s+IN[SHD]C\s+\d+\b",
    r"\b[A-Z0-9]{16}\b",
    
    # Fallback
    r"No\.?\s*\d+\s*(?:of|/)\s*\d{4}",
]

# 2. TEXT CLEANING & PREPROCESSING

def clean_legal_text(raw_text):
    """
    Removes common PDF artifacts and noise from legal documents.
    Returns cleaned text ready for NLP processing.
    """
    if not raw_text or len(raw_text) < 10:
        return raw_text
    
    text = raw_text
    
    # Remove page markers
    text = re.sub(r'\bPage\s+\d+(?:\s+of\s+\d+)?\b', '', text, flags=re.I)
    text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.M)
    
    # Remove download artifacts
    text = re.sub(r'https?://[^\s]+', '', text)
    text = re.sub(r'Downloaded from[^\n]+', '', text, flags=re.I)
    
    # Remove legal citations
    text = re.sub(r'\[\d+\]', '', text)
    text = re.sub(r'\(\d{4}\)', '', text)
    text = re.sub(r'\b(?:AIR|SCC|SCR)\s+\d{4}\s+\w+\s+\d+', '', text)
    
    # Detect and remove repeated lines (headers/footers)
    lines = text.split('\n')
    line_frequency = Counter([line.strip() for line in lines if len(line.strip()) > 10])
    repeated_lines = {line for line, count in line_frequency.items() if count >= 3}
    
    filtered_lines = [line for line in lines if line.strip() not in repeated_lines]
    text = '\n'.join(filtered_lines)
    
    # Remove very short lines (likely artifacts)
    lines = [line for line in text.split('\n') if len(line.strip()) > 2]
    text = '\n'.join(lines)
    
    # Normalize spacing
    text = re.sub(r' +', ' ', text)
    text = re.sub(r'\n\n+', '\n\n', text)
    
    return text.strip()

# 3. PARTY NAME NORMALIZATION

def normalize_party_name(raw_party_text):
    """
    Extracts clean party names from text blocks containing addresses,
    personal details, and legal markers.
    """
    lines = raw_party_text.split('\n')
    extracted_names = []
    
    # Common markers that indicate address/personal details start
    address_markers = [
        "s/o", "d/o", "w/o", "aged", "r/o", "resident", 
        "village", "dist", "pin", "po-", "ps-",
        "advocate", "counsel", "through", "represented by"
    ]
    
    for line in lines:
        line = line.strip()
        
        # Skip empty or very short lines
        if not line or len(line) < 3:
            continue
        
        # Check if line contains address markers
        line_lower = line.lower()
        has_address_marker = any(marker in line_lower for marker in address_markers)
        
        if has_address_marker:
            # Extract name before the marker
            for marker in address_markers:
                if marker in line_lower:
                    name_before_marker = line.split(marker, 1)[0].strip()
                    name_before_marker = re.sub(r'[,\.\-]+$', '', name_before_marker)
                    if len(name_before_marker) > 3:
                        extracted_names.append(name_before_marker)
                    break
            break  # Stop after finding address details
        
        # Remove common trailing labels
        line = re.sub(
            r"\.{3,}|\b\d+\.|\b(Petitioner|Respondent|Appellant|Accused|Applicant)\b.*", 
            "", 
            line, 
            flags=re.I
        ).strip()
        
        if line and len(line) > 2:
            extracted_names.append(line)
            
            # Usually 1-2 lines is sufficient for a name
            if len(extracted_names) >= 2:
                break
    
    # Return formatted result
    if not extracted_names:
        return "Party Details Not Found"
    
    result = " ".join(extracted_names[:2])
    if len(extracted_names) > 2:
        result += " & Ors"
    
    return result


# 4. ADVANCED PARTY EXTRACTION ENGINE

def extract_parties_advanced(document_text):
    """
    Multi-strategy party extraction system.
    Handles various Indian legal document formats.
    """
    if not document_text:
        return None
    
    # Normalize whitespace
    document_text = re.sub(r'\s+', ' ', document_text)
    header_section = document_text[:30000]
    
    # Remove common noise patterns
    header_section = re.sub(
        r'Approved\s+for\s+Reporting\s+(?:Yes|No|YesNo)', 
        '', 
        header_section, 
        flags=re.I
    )
    
    petitioner = None
    respondent = None
    
    # ========== STRATEGY 1: Multi-word Authority Format ==========
    # Pattern: "Authority Name\nPetitioner\nVersus\nRespondent Name\nRespondents"
    # Common in PIL cases and institutional petitions
    
    authority_pattern = re.search(
        r'([A-Z][A-Za-z\s&\.,()-]{10,150}?)\s+Petitioner\s+Versus\s+([A-Z][A-Za-z\s&\.,()-]{10,150}?)\s+Respondents?',
        header_section,
        re.IGNORECASE
    )
    
    if authority_pattern:
        petitioner = authority_pattern.group(1).strip()
        respondent = authority_pattern.group(2).strip()
        
        # Normalize "and others" variations
        petitioner = re.sub(r'\s*(?:and|&)\s*(?:others?|ors?\.?)', ' & Ors', petitioner, flags=re.I)
        respondent = re.sub(r'\s*(?:and|&)\s*(?:others?|ors?\.?)', ' & Ors', respondent, flags=re.I)
        
        return f"{petitioner}\n-vs-\n{respondent}"
    
    # ========== STRATEGY 2: Labeled Colon Format ==========
    # Pattern: "PETITIONER:\nName\nVs.\nRESPONDENT:\nName"
    
    colon_pattern = re.search(
        r'PETITIONER\s*:\s*(?:\d+\.\s*)?([A-Z][A-Za-z\s&\.,()-]+?)\s+Vs\.\s+RESPONDENT\s*:\s*([A-Z][A-Za-z\s&\.,()-]+)',
        header_section,
        re.IGNORECASE | re.DOTALL
    )
    
    if colon_pattern:
        petitioner = colon_pattern.group(1).strip()
        respondent = colon_pattern.group(2).strip()
        
        # Remove parenthetical content
        petitioner = re.sub(r'\s*\(.*?\)', '', petitioner)
        respondent = re.sub(r'\s*\(.*?\)', '', respondent)
        
        return f"{petitioner}\n-vs-\n{respondent}"
    
    # ========== STRATEGY 3: Company/Organization Format ==========
    # Pattern: "M/s Company Name ... Petitioner"
    
    company_pattern = re.search(
        r'(M/s\s+[A-Za-z\s,\.]+?)(?:\s*--?\s*Petitioner|\s+Versus)',
        header_section,
        re.IGNORECASE
    )
    
    if company_pattern:
        petitioner = company_pattern.group(1).strip()
        petitioner = re.sub(r'\s*,.*$', '', petitioner)  # Remove address after comma
        
        # Find respondent after Versus
        versus_position = header_section[company_pattern.end():]
        respondent_match = re.search(
            r'Versus\s+([A-Z][A-Za-z\s,\.&]+?)(?:\s*--?\s*Respondent|\s+With)',
            versus_position,
            re.IGNORECASE
        )
        
        if respondent_match:
            respondent = respondent_match.group(1).strip()
            respondent = re.sub(r'\s*,.*$', '', respondent)
            return f"{petitioner}\n-vs-\n{respondent}"
    
    # ========== STRATEGY 4: All-Caps With Dots Format ==========
    # Pattern: "NAME ... PETITIONER\nVERSUS\nNAME ... RESPONDENT"
    
    caps_dots_pattern = re.search(
        r'([A-Z][A-Z\s]+[A-Z])\s+\.{3,}\s*PETITIONER\s+VERSUS\s+([A-Z][A-Z\s]+[A-Z])\s+\.{3,}\s*RESPONDENT',
        header_section,
        re.IGNORECASE
    )
    
    if caps_dots_pattern:
        petitioner = caps_dots_pattern.group(1).strip()
        respondent = caps_dots_pattern.group(2).strip()
        
        # Convert to title case for readability
        petitioner = ' '.join(word.capitalize() for word in petitioner.split())
        respondent = ' '.join(word.capitalize() for word in respondent.split())
        
        return f"{petitioner}\n-vs-\n{respondent}"
    
    # ========== STRATEGY 5: Title With Address Format ==========
    # Pattern: "Smt./Shri Name, address details ... Petitioner"
    
    title_pattern = re.search(
        r'((?:Smt\.|Shri|Sri|Dr\.|M/s|Mr\.|Mrs\.|Ms\.)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
        header_section,
        re.IGNORECASE
    )
    
    if title_pattern:
        petitioner = title_pattern.group(1).strip()
        
        # Find Versus marker after petitioner
        versus_marker = re.search(
            r'(?:VERSUS|Versus|V/S|VS)', 
            header_section[title_pattern.end():], 
            re.I
        )
        
        if versus_marker:
            after_versus = header_section[title_pattern.end() + versus_marker.end():]
            
            # Look for State/Union pattern
            government_pattern = re.search(
                r'(?:1\.\s*)?(?:The\s+)?((?:State|Union)\s+of\s+[A-Z][a-z]+)',
                after_versus[:500],
                re.IGNORECASE
            )
            
            if government_pattern:
                respondent = government_pattern.group(1)
                return f"{petitioner}\n-vs-\n{respondent}"
    
    # ========== STRATEGY 6: Standard Versus Split ==========
    # Pattern: "Name VERSUS Name"
    
    versus_split = re.search(
        r'([A-Z][A-Za-z\s\.]{5,80}?)\s+(?:VERSUS|versus|Versus|V/S|v/s|VS|vs\.?)\s+([A-Z][A-Za-z\s\.&]{5,80})',
        header_section[:5000]
    )
    
    if versus_split:
        petitioner = versus_split.group(1).strip()
        respondent = versus_split.group(2).strip()
        
        # Remove date references
        petitioner = re.sub(r'\s*on\s+\d+.*$', '', petitioner, flags=re.I)
        respondent = re.sub(r'\s*on\s+\d+.*$', '', respondent, flags=re.I)
        
        # Filter out court names and noise
        noise_keywords = [
            'SUPREME COURT', 'HIGH COURT', 'SESSIONS', 'COURT OF',
            'Approved', 'Reporting', 'YesNo', 'Appearance'
        ]
        
        if not any(keyword in petitioner for keyword in noise_keywords):
            return f"{petitioner}\n-vs-\n{respondent}"
    
    return "Parties Not Detected"


def extract_parties(document_text):
    """
    Main party extraction function with fallback logic.
    Tries advanced strategies first, then falls back to original logic.
    """
    header = document_text[:6000].replace('\r', '')
    header = re.split(r"\n\s*WITH\s*\n", header, flags=re.I)[0]

    # Try advanced extraction first
    advanced_result = extract_parties_advanced(document_text)
    if advanced_result and advanced_result != "Parties Not Detected":
        return advanced_result

    # Fallback Strategy 1: Anchor Block Match
    anchor_pattern = r"(?:No\.?|Petition|Appeal|SLP|CRL\.A)(?:[\s\S]+?\d{4})?([\s\S]+?)\s+(?:VERSUS|V/S|VS\.?)\s+([\s\S]+?)(?=\n\s*(?:ORDER|JUDGMENT|BEFORE|JUSTICE|CORAM|DATED|PRESENT))"
    match = re.search(anchor_pattern, header, re.I)
    
    if match:
        petitioner = normalize_party_name(match.group(1))
        respondent = normalize_party_name(match.group(2))
        
        if "Court" not in petitioner and len(petitioner) > 2:
            return f"{petitioner}\n-vs-\n{respondent}"

    # Fallback Strategy 2: Positional Split
    versus_markers = [r"\n\s*VERSUS\s*\n", r"\n\s*V/S\s*\n", r"\n\s*VS\.?\s*\n"]
    
    for marker in versus_markers:
        split_result = re.split(marker, header, maxsplit=1, flags=re.I)
        
        if len(split_result) == 2:
            # Get last 4 lines before VERSUS
            before_lines = [l for l in split_result[0].strip().split('\n') if len(l.strip()) > 2][-4:]
            # Get first 4 lines after VERSUS
            after_lines = [l for l in split_result[1].strip().split('\n') if len(l.strip()) > 2][:4]
            
            petitioner_text = "\n".join(before_lines)
            respondent_text = "\n".join(after_lines)
            
            petitioner = normalize_party_name(petitioner_text)
            respondent = normalize_party_name(respondent_text)
            
            if len(petitioner) > 3:
                return f"{petitioner}\n-vs-\n{respondent}"

    return "Parties Not Detected"


# 5. NLP MODEL INITIALIZATION


@st.cache_resource
def load_summarization_model():
    """
    Loads and caches the DistilBART summarization model.
    Uses Streamlit caching to avoid repeated loading.
    """
    return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")


# 6. MAIN DOCUMENT ANALYSIS ENGINE


@st.cache_data
def get_summarized_data(raw_document_text):
    """
    Primary analysis function that orchestrates all document processing.
    Returns comprehensive metadata and summaries.
    """
    
    # Step 1: Clean the document
    cleaned_text = clean_legal_text(raw_document_text)
    
    # Load summarization model
    summarizer = load_summarization_model()
    
    # Prepare text sections for analysis
    safe_summary_text = cleaned_text[:3200]
    header_section = cleaned_text[:5000].replace('\r', '')

    # ========== EXTRACT COURT NAME ==========
    court_name = "COURT NOT DETECTED"
    
    for pattern in COURT_PATTERNS:
        court_match = re.search(pattern, header_section, re.I)
        if court_match:
            court_name = court_match.group(0).upper()
            # Clean up prefixes
            court_name = re.sub(r'^(?:IN THE|BEFORE THE|THE)\s+', '', court_name)
            court_name = re.sub(r"HON'?BLE\s+", '', court_name)
            break
    
    # Flexible fallback for unmatched courts
    if court_name == "COURT NOT DETECTED":
        flexible_court = re.search(
            r'(HIGH COURT OF [A-Z\s]+ AT [A-Z\s]+|SESSIONS COURT AT [A-Z\s]+)',
            header_section,
            re.I
        )
        if flexible_court:
            court_name = flexible_court.group(1).upper()

    # ========== EXTRACT CASE NUMBER ==========
    case_number = "N/A"
    
    for pattern in CASE_NO_PATTERNS:
        case_match = re.search(pattern, header_section, re.I)
        if case_match:
            case_number = case_match.group(0).strip()
            break

    # ========== EXTRACT PARTIES ==========
    parties_result = extract_parties(cleaned_text)
    
    # ========== DETERMINE JURISDICTION ==========
    jurisdiction_patterns = {
        "Writ Jurisdiction": r"(?i)(article\s+(?:226|32|227|136|142|141)|writ\s+petition|constitutional\s+remedy|writ\s+of\s+(?:habeas corpus|mandamus|prohibition|certiorari|quo warranto))",
        "Appellate Jurisdiction": r"(?i)(civil\s+appeal|criminal\s+appeal|special\s+leave\s+petition|appellate\s+jurisdiction|regular\s+(?:first|second)\s+appeal)",
        "Original Jurisdiction": r"(?i)(original\s+suit|civil\s+original|original\s+side|original\s+jurisdiction)",
        "Bail Jurisdiction": r"(?i)(bail\s+application|anticipatory\s+bail|section\s+(?:438|439|437|436|167)|regular\s+bail)",
        "Revisional Jurisdiction": r"(?i)(civil\s+revision|criminal\s+revision|revisional\s+jurisdiction|revision\s+petition)",
        "Criminal Original Jurisdiction": r"(?i)(criminal\s+complaint|complaint\s+case|section\s+(?:138|156|200|340|482))",
        "Contempt Jurisdiction": r"(?i)(contempt\s+of\s+court|criminal\s+contempt|civil\s+contempt)",
        "Execution Jurisdiction": r"(?i)(execution\s+petition|execution\s+proceedings|decree\s+execution)",
        "General Jurisdiction": r".*",
    }
    
    jurisdiction_type = "General Jurisdiction"
    
    for label, pattern in jurisdiction_patterns.items():
        if label != "General Jurisdiction" and re.search(pattern, header_section, re.I):
            jurisdiction_type = label
            break

    # ========== GENERATE NLP SUMMARIES ==========
    try:
        # Executive Summary
        executive_summary = summarizer(
            safe_summary_text, 
            max_length=180, 
            min_length=100, 
            truncation=True
        )[0]['summary_text']
        
        # Background Summary
        petition_start = cleaned_text.lower().find("this petition")
        if petition_start != -1:
            background_text = cleaned_text[petition_start:petition_start+3000]
        else:
            background_text = cleaned_text[:3000]
        
        background_summary = summarizer(
            background_text, 
            max_length=250, 
            min_length=120, 
            truncation=True
        )[0]['summary_text']

        # Issues Summary
        issues_summary = summarizer(
            cleaned_text[1000:4000][:3000], 
            max_length=130, 
            min_length=60, 
            truncation=True
        )[0]['summary_text']
        
        # Observations Summary
        middle_section = len(cleaned_text) // 3
        observations_summary = summarizer(
            cleaned_text[middle_section:middle_section + 3000], 
            max_length=200, 
            min_length=80, 
            truncation=True
        )[0]['summary_text']
        
        # Decision Summary
        decision_summary = summarizer(
            cleaned_text[-3000:], 
            max_length=120, 
            min_length=50, 
            truncation=True
        )[0]['summary_text']
        
    except Exception as processing_error:
        # Graceful degradation if summarization fails
        executive_summary = "Summarization processing encountered an error."
        background_summary = "Unable to generate background summary."
        issues_summary = "Unable to extract issues."
        observations_summary = "Unable to extract observations."
        decision_summary = "Unable to extract decision."

    # ========== EXTRACT VERBATIM SENTENCES ==========
    def extract_verbatim_sentences(keyword_list):
        """
        Finds and extracts complete sentences containing specified keywords.
        Returns sentences with reference IDs for traceability.
        """
        extracted_sentences = []
        
        for keyword in keyword_list:
            # Pattern to match complete sentences containing the keyword
            sentence_pattern = rf"([A-Z][^.!?]*?\b{keyword}\b[^.!?]*?[.!?])"
            
            for match in re.finditer(sentence_pattern, cleaned_text, re.I | re.DOTALL):
                sentence = match.group().strip()
                
                # Filter by length (avoid fragments and overly long matches)
                if 30 < len(sentence) < 500:
                    reference_id = match.start()
                    extracted_sentences.append(f"[Ref ID: {reference_id}] {sentence}")
        
        # Remove duplicates while preserving order
        unique_sentences = list(dict.fromkeys(extracted_sentences))
        
        return unique_sentences[:5]  # Return top 5 matches

    # Compile final analysis result
    analysis_result = {
        "court": court_name,
        "case_no": case_number,
        "jurisdiction": jurisdiction_type,
        "parties": parties_result,
        "exec_summary": executive_summary,
        "background": background_summary,
        "issues": issues_summary,
        "observations": observations_summary,
        "decision": decision_summary,
        "source_log": {
            "Case Background Trace": extract_verbatim_sentences([
                "fact", "background", "incident", "allegation", 
                "accused", "victim", "petitioner", "appellant", "case"
            ]),
            "Court Observation Trace": extract_verbatim_sentences([
                "observed", "held", "court", "opined", "noted", 
                "finding", "concluded", "reasoning"
            ]),
            "Final Decision Trace": extract_verbatim_sentences([
                "directed", "ordered", "dismissed", "allowed", 
                "disposed", "decree", "judgment", "held that"
            ])
        }
    }
    
    return analysis_result