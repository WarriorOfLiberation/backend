# 📚 Imports and PDF Text Extraction
import os
import json
import uuid
import fitz  # PyMuPDF - excellent PDF text extraction
from langchain.text_splitter import RecursiveCharacterTextSplitter
from openai import OpenAI
from dotenv import load_dotenv
import threading
import concurrent.futures
from typing import Dict, List, Tuple
import time
import backoff
import re
import spacy  # Added for Legal NER
import requests
from bs4 import BeautifulSoup
import urllib.parse
from typing import Optional
from duckduckgo_search import DDGS
import base64
from concurrent.futures import ThreadPoolExecutor
import datetime


def search_case_url_ddg_library(case_name: str) -> Optional[str]:
    """Use duckduckgo-search library"""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(case_name, max_results=1))
            if results:
                return results[0]['href']
        return None
    except Exception as e:
        print(f"   ⚠️ DDG library search error: {e}")
        return None
def search_case_url_google(case_name: str) -> Optional[str]:
    """Use Google to search for case on legal sites"""
    try:
        time.sleep(1)  # Be respectful to Google
        
        # Search on specific sites
        sites = ["site:lawyerservices.in", "site:taxmann.com", "site:indiankanoon.org"]
        search_query = f'"{case_name}" {" OR ".join(sites)}'
        encoded_query = urllib.parse.quote(search_query)
        
        search_url = f"https://www.google.com/search?q={encoded_query}"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        }
        
        response = requests.get(search_url, headers=headers, timeout=10)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find first result link
            search_results = soup.find_all('div', {'class': 'g'})
            for result in search_results[:3]:
                link = result.find('a')
                if link and link.get('href'):
                    url = link['href']
                    # Check if it's from our target sites
                    if any(site.replace('site:', '') in url for site in sites):
                        return url
        
        return None
    except Exception as e:
        print(f"   ⚠️ Google search error: {e}")
        return None
# Load environment variables
load_dotenv()
api_key = os.getenv("openai_api_key")

# Initialize OpenAI client with explicit timeout settings
client = OpenAI(
    api_key=api_key,
    base_url="https://oai.helicone.ai/v1",
    default_headers={
        "Helicone-Auth": f"Bearer {os.getenv('HELICONE_API_KEY')}",
    },
    timeout=300,  # 5 minutes timeout for long prompts
    max_retries=3
)
# In your rag_enhanced.py, add at top:
import requests

SUPABASE_URL = "https://bitwjwnuoacqcfuqbkpr.supabase.co"
SUPABASE_KEY = os.getenv("SUPABASE_ANON_KEY")

def store_analysis_result(session_id: str, pdf_name: str, primary_issues: str, final_analysis: str):
    """Store analysis results in Supabase"""
    
    url = f"{SUPABASE_URL}/rest/v1/analyses"
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json"
    }
    
    data = {
        "session_id": session_id,
        "pdf_name": pdf_name,
        "primary_issues": primary_issues[:1000],  # First 1000 chars
        "full_analysis": final_analysis
    }
    
    response = requests.post(url, json=data, headers=headers)
    if response.status_code == 201:
        print(f"✅ Analysis stored for session {session_id}")
    else:
        print(f"❌ Failed to store: {response.text}")

class O3MiniPDFLabeler:
    """Use o3-mini to intelligently label legal PDFs based on comprehensive analysis"""
    
    def __init__(self, client):
        self.client = client
        self.colors = {
            'FINDINGS_REASONING': (0.3, 0.5, 0.9, 0.3),    # Blue - for court findings, reasoning, precedents
            'CONTENTION_CHALLENGE': (0.9, 0.5, 0.2, 0.3),  # Orange - for contentions, challenges
        }
    
    def analyze_legal_content_with_o3mini(self, text: str, page_num: int, session_id: str = None) -> List[Dict]:
        """Use o3-mini to analyze legal content and identify key passages"""
        
        prompt = """Analyze this legal text and identify passages that relate to:

1. EXECUTIVE SUMMARY elements:
   - Main findings of the court
   - Court's reasoning
   - Case law or statutory provisions referenced

2. POINTS OF CONTENTION:
   - Disagreements between parties
   - Issues the court adjudicated
   - Key arguments raised

3. PROCEDURAL CHALLENGES (if any):
   - Procedural irregularities
   - Limitation issues
   - Notice problems

4. LEGAL REASONING:
   - References to binding precedents
   - Statutory interpretation

Return a JSON array where each item identifies a relevant passage:
[
  {
    "text": "exact text passage (up to 200 characters)",
    "category": "FINDINGS_REASONING" or "CONTENTION_CHALLENGE",
    "type": "main_finding|reasoning|case_law|contention|procedural|precedent",
    "importance": "high|medium|low"
  }
]

**CRITICAL RULES:**
- Return ONLY passages that clearly relate to the above categories
- Each passage must be EXACT text from the document (case-sensitive)
- Keep passages concise but meaningful (20-200 characters)
- Mark procedural challenges as CONTENTION_CHALLENGE category
- Mark all findings/reasoning/precedents as FINDINGS_REASONING category

TEXT TO ANALYZE:
""" + text

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",  # Using faster model
                messages=[
                    {"role": "system", "content": "You are a legal document analyzer. Extract key passages for highlighting. Return ONLY valid JSON arrays."},
                    {"role": "user", "content": prompt}
                ],
                timeout=60,
                temperature=1,
                extra_headers={
                    "Helicone-Session-Id": session_id or "default",
                }
            )
            
            result = response.choices[0].message.content.strip()
            result = result.replace('```json', '').replace('```', '').strip()
            
            passages = json.loads(result)
            for passage in passages:
                passage['page'] = page_num
            
            return passages
            
        except Exception as e:
            print(f"❌ Error analyzing legal content: {e}")
            return []
    
    def label_pdf_parallel(self, pdf_path: str, session_id: str) -> str:
        """Label PDF with legal analysis highlights in parallel with main analysis"""
        try:
            print(f"🏷️ Starting parallel PDF labeling for session {session_id}")
            
            # Update progress - starting
            update_progress(session_id, "pdf-labeling", "active", "Starting legal content analysis...")
            
            # Open the PDF
            doc = fitz.open(pdf_path)
            total_pages = len(doc)
            all_passages = []
            
            # Process each page
            for page_num, page in enumerate(doc):
                page_text = page.get_text()
                
                # Skip empty pages
                if len(page_text.strip()) < 50:
                    continue
                
                # Update progress
                progress_pct = ((page_num + 1) / total_pages) * 100
                update_progress(
                    session_id,
                    "pdf-labeling",
                    "active", 
                    f"Analyzing page {page_num + 1}/{total_pages} ({progress_pct:.0f}%)"
                )
                
                # Analyze content for key passages
                passages = self.analyze_legal_content_with_o3mini(page_text, page_num, session_id)
                all_passages.extend(passages)
                
                # Apply highlights based on category
                for passage in passages:
                    if passage['category'] in self.colors:
                        color = self.colors[passage['category']]
                        text_instances = page.search_for(passage['text'])
                        
                        # Limit highlights per passage to avoid over-highlighting
                        max_highlights = 5 if passage.get('importance') == 'high' else 3
                        
                        for inst in text_instances[:max_highlights]:
                            highlight = page.add_highlight_annot(inst)
                            
                            # Adjust opacity for high importance items
                            if passage.get('importance') == 'high':
                                adjusted_color = (color[0], color[1], color[2], min(color[3] * 1.5, 0.5))
                            else:
                                adjusted_color = color
                            
                            highlight.set_colors(stroke=adjusted_color[:3])
                            highlight.set_opacity(adjusted_color[3])
                            highlight.update()
            
            # Save labeled PDF to memory
            pdf_bytes = doc.write()
            doc.close()
            
            # Convert to base64
            pdf_base64 = base64.b64encode(pdf_bytes).decode('utf-8')
            
            # Send completion with labeled PDF
            update_progress(
                session_id,
                "pdf-labeling", 
                "complete",
                f"Highlighted {len(all_passages)} key legal passages",
                {"labeled_pdf": pdf_base64, "passage_count": len(all_passages)}
            )
            
            return pdf_base64
            
        except Exception as e:
            print(f"❌ Error in PDF labeling: {e}")
            update_progress(
                session_id,
                "pdf-labeling",
                "failed",
                f"Error: {str(e)}"
            )
            return None
    
    def _create_summary(self, passages: List[Dict], total_pages: int) -> str:
        """Create a summary of highlighted passages"""
        if not passages:
            return ""
        
        category_counts = {}
        type_counts = {}
        high_importance = []
        
        for passage in passages:
            cat = passage['category']
            typ = passage['type']
            category_counts[cat] = category_counts.get(cat, 0) + 1
            type_counts[typ] = type_counts.get(typ, 0) + 1
            
            if passage.get('importance') == 'high':
                high_importance.append(passage)
        
        summary = f"Legal Analysis Summary\n"
        summary += f"Total Pages: {total_pages}\n"
        summary += f"Total Passages Highlighted: {len(passages)}\n\n"
        
        summary += "Highlights by Category:\n"
        if 'FINDINGS_REASONING' in category_counts:
            summary += f"  • Court Findings & Legal Reasoning: {category_counts['FINDINGS_REASONING']}\n"
        if 'CONTENTION_CHALLENGE' in category_counts:
            summary += f"  • Points of Contention & Challenges: {category_counts['CONTENTION_CHALLENGE']}\n"
        
        summary += "\nHighlights by Type:\n"
        for typ, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True):
            summary += f"  • {typ.replace('_', ' ').title()}: {count}\n"
        
        if high_importance:
            summary += "\nHigh Importance Passages:\n"
            for passage in high_importance[:5]:
                summary += f"  • {passage['type']}: {passage['text'][:80]}...\n"
        
        return summary
# Initialize the labeler globally
pdf_labeler = O3MiniPDFLabeler(client)

# Global progress tracking
progress_tracker = {}

# 🧠 Legal NER Integration
class LegalNERExtractor:
    """Simple Legal NER for case name extraction using en_legal_ner_trf"""
    
    def __init__(self):
        self.nlp = None
        self._init_legal_ner()
    
    def _init_legal_ner(self):
        """Initialize the legal NER model"""
        try:
            self.nlp = spacy.load("en_legal_ner_trf")
            print("✅ Legal NER model (en_legal_ner_trf) loaded successfully")
        except OSError:
            print("❌ Legal NER model not found. Install with: python -m spacy download en_legal_ner_trf")
            print("⚠️ Falling back to regex-based extraction")
            self.nlp = None
    
    def extract_case_name_from_text(self, text: str) -> str:
        """Extract case name from first 300 characters using Legal NER"""
        if not self.nlp:
            return self._fallback_regex_extraction(text)
        
        # Take first 300 characters where case names usually appear
        first_part = text[:600]
        
        # Clean the text for better NER processing
        cleaned_text = self._clean_text_for_ner(first_part)
        
        # Process with Legal NER
        doc = self.nlp(cleaned_text)
        
        # Extract entities
        persons = []
        organizations = []
        
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                clean_name = self._clean_entity_name(ent.text)
                if len(clean_name) > 2 and clean_name not in persons:
                    persons.append(clean_name)
            elif ent.label_ in ["ORG", "ORGANIZATION"]:
                clean_name = self._clean_entity_name(ent.text)
                if len(clean_name) > 2 and clean_name not in organizations:
                    organizations.append(clean_name)
        
        # Construct case name
        case_name = self._construct_case_name(persons, organizations, cleaned_text)
        
        return case_name if case_name else self._fallback_regex_extraction(text)
    
    def _clean_text_for_ner(self, text: str) -> str:
        """Clean text for better NER processing - handles multi-line legal documents"""
        # Remove website headers
        text = re.sub(r'(?:Latestlaws|Centaxonline)\.com[^\n]*', '', text, flags=re.IGNORECASE)
        
        # First, reconstruct broken lines for legal entities
        text = self._reconstruct_legal_lines(text)
        
        # Normalize all "versus" variations to "vs"
        text = re.sub(r'\b(?:versus|Versus|VERSUS)\b', 'vs', text, flags=re.IGNORECASE)
        
        # Remove excessive whitespace but preserve line breaks initially
        text = re.sub(r'[ \t]+', ' ', text)  # Only collapse spaces/tabs, not newlines
        
        # Now collapse multiple newlines to single spaces for final processing
        text = re.sub(r'\n+', ' ', text)
        
        # Remove citation patterns that might confuse NER
        text = re.sub(r'\[\d{4}\]|\(\d{4}\)|Citation\s*:', '', text)
        
        # Final whitespace cleanup
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def _reconstruct_legal_lines(self, text: str) -> str:
        """Reconstruct legal entity names that are broken across lines"""
        lines = text.split('\n')
        reconstructed_lines = []
        i = 0
        
        while i < len(lines):
            current_line = lines[i].strip()
            
            # Skip empty lines
            if not current_line:
                i += 1
                continue
            
            # Check if this line and next few lines form a legal case pattern
            case_block = self._detect_case_block(lines, i)
            if case_block:
                # Join the case block into a single line
                reconstructed_lines.append(case_block['joined_text'])
                i = case_block['end_index']
            else:
                reconstructed_lines.append(current_line)
                i += 1
        
        return '\n'.join(reconstructed_lines)
    
    def _detect_case_block(self, lines: List[str], start_index: int) -> Dict:
        """Detect if lines starting at start_index form a legal case name block"""
        if start_index >= len(lines):
            return None
        
        # Look ahead up to 4 lines to find case patterns
        look_ahead = min(4, len(lines) - start_index)
        block_lines = []
        
        for i in range(look_ahead):
            line = lines[start_index + i].strip()
            if line:
                block_lines.append(line)
        
        # Join the block and check for legal case patterns
        joined_text = ' '.join(block_lines)
        
        # Check if this looks like a case name pattern
        vs_patterns = [
            r'[A-Z][A-Za-z\s&\.\,\(\)Ltd\.Co\.Inc\.]+?\s+(?:vs?\.?|v\.?|versus|Versus|VERSUS)\s+[A-Z][A-Za-z\s&\.\,\(\)Ltd\.Co\.Inc\.]+',
            r'[A-Z\s]+\s+(?:vs?\.?|v\.?|versus|Versus|VERSUS)\s+[A-Z\s]+'
        ]
        
        for pattern in vs_patterns:
            if re.search(pattern, joined_text, re.IGNORECASE):
                # Count how many lines this pattern spans
                pattern_end = start_index
                temp_text = ""
                
                for j in range(start_index, min(start_index + 4, len(lines))):
                    temp_text += " " + lines[j].strip()
                    if re.search(pattern, temp_text.strip(), re.IGNORECASE):
                        pattern_end = j + 1
                        break
                
                return {
                    'joined_text': temp_text.strip(),
                    'end_index': pattern_end
                }
        
        return None
    
    def _clean_entity_name(self, entity_name: str) -> str:
        """Clean extracted entity names"""
        # Remove common prefixes
        entity_name = re.sub(r'^(The\s+|M/s\s+|Ms\.\s+|Mr\.\s+|Shri\s+|Commissioner\s+of\s+)', '', entity_name, flags=re.IGNORECASE)
        
        # Remove trailing punctuation
        entity_name = re.sub(r'[,\.\;]+$', '', entity_name)
        
        # Clean whitespace
        entity_name = re.sub(r'\s+', ' ', entity_name.strip())
        
        return entity_name
    
    def _construct_case_name(self, persons: List[str], organizations: List[str], text: str) -> str:
        """Construct case name from extracted entities"""
        # Strategy 1: Two organizations (most common in tax cases)
        if len(organizations) >= 2:
            return f"{organizations[0]} vs {organizations[1]}"
        
        # Strategy 2: One organization and one person
        if len(organizations) >= 1 and len(persons) >= 1:
            return f"{organizations[0]} vs {persons[0]}"
        
        # Strategy 3: Two persons
        if len(persons) >= 2:
            return f"{persons[0]} vs {persons[1]}"
        
        # Strategy 4: Look for "vs" pattern in text (now normalized) and use entities around it
        vs_match = re.search(r'(.{10,50})\s+vs\s+(.{10,50})', text, re.IGNORECASE)
        if vs_match and (persons or organizations):
            # Try to match entities to the vs pattern
            left_part = vs_match.group(1).strip()
            right_part = vs_match.group(2).strip()
            
            # Find best matching entities
            left_entity = self._find_best_entity_match(left_part, persons + organizations)
            right_entity = self._find_best_entity_match(right_part, persons + organizations)
            
            if left_entity and right_entity:
                return f"{left_entity} vs {right_entity}"
        
        # Strategy 5: If we have entities but no clear vs pattern, try to construct from the cleaned text
        if persons or organizations:
            all_entities = persons + organizations
            # Look for any vs pattern in the text and try to map entities
            simple_vs_match = re.search(r'vs\s+', text, re.IGNORECASE)
            if simple_vs_match and len(all_entities) >= 2:
                return f"{all_entities[0]} vs {all_entities[1]}"
        
        return ""  # No valid case name found
    
    def _find_best_entity_match(self, text_part: str, entities: List[str]) -> str:
        """Find the best entity match in a text part"""
        for entity in entities:
            if entity.lower() in text_part.lower() or text_part.lower() in entity.lower():
                return entity
        return ""
    
    def _fallback_regex_extraction(self, text: str) -> str:
        """Fallback regex extraction (enhanced version of original method)"""
        cleaned_text = text.replace('latestlaws.com', '').replace('Latestlaws.com', '')
        
        # Normalize all versus variations to vs first
        cleaned_text = re.sub(r'\b(?:versus|Versus|VERSUS)\b', 'vs', cleaned_text, flags=re.IGNORECASE)
        
        first_part = cleaned_text[:300]  # Keep 300 characters as requested
        
        lines = [line.strip() for line in first_part.split('\n') if line.strip()]
        vs_pattern = r'\b(?:vs?\.?|v\.?)\b'
        
        for i, line in enumerate(lines):
            if re.search(vs_pattern, line, re.IGNORECASE):
                parts = re.split(vs_pattern, line, flags=re.IGNORECASE)
                if len(parts) >= 2:
                    party1_text = parts[0].strip()
                    if not party1_text and i > 0:
                        party1_text = ' '.join(lines[max(0, i-2):i]).strip()
                    else:
                        if len(party1_text.split()) < 2 and i > 0:
                            party1_text = (lines[i-1] + ' ' + party1_text).strip()
                    
                    party2_text = parts[1].strip()
                    party2_text = re.sub(r'\[.*?\]', '', party2_text).strip()
                    
                    if party1_text and party2_text:
                        party1 = re.sub(r'\s+', ' ', party1_text.strip())
                        party2 = re.sub(r'\s+', ' ', party2_text.strip())
                        return f"{party1} vs {party2}"
        
        return "Case name not found"

# Initialize Legal NER extractor
legal_ner = LegalNERExtractor()

def update_progress(session_id: str, step: str, status: str, details: str = "", data: dict = None):
    """Update progress for a specific analysis session"""
    if session_id not in progress_tracker:
        progress_tracker[session_id] = {}
    
    progress_tracker[session_id][step] = {
        'status': status,
        'details': details,
        'timestamp': time.time(),
        'data': data or {}
    }
    
    print(f"📊 Progress Update [{session_id}]: {step} -> {status}")
    if details:
        print(f"   📝 {details}")
    
    # Add small delay to ensure frontend can catch state changes
    time.sleep(0.5)

def get_progress(session_id: str) -> dict:
    """Get current progress for a session"""
    return progress_tracker.get(session_id, {})

def load_paragraphs_from_pdf(pdf_path):
    """
    Extract text from PDF using PyMuPDF and split into paragraphs.
    PyMuPDF is one of the best PDF text extractors - fast, accurate, handles complex layouts.
    """
    try:
        # Open the PDF
        doc = fitz.open(pdf_path)
        text = ""
        
        # Extract text from each page
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            # Get text with layout preservation
            page_text = page.get_text("text")
            text += page_text + "\n"
        
        doc.close()
        
        # Split text into chunks using RecursiveCharacterTextSplitter
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=100,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )
        
        return splitter.split_text(text)
        
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return []

# Alternative function for getting full text (useful for some use cases)
def extract_full_text_from_pdf(pdf_path):
    """Extract complete text from PDF as a single string."""
    try:
        doc = fitz.open(pdf_path)
        full_text = ""
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            full_text += page.get_text("text") + "\n"
        
        doc.close()
        return full_text.strip()
        
    except Exception as e:
        print(f"Error extracting full text from PDF: {e}")
        return ""

print("✅ PDF extraction functions loaded successfully!")
print("💡 Install PyMuPDF with: pip install PyMuPDF")

# 🧠 Tax Copilot Flow Implementation

def extract_primary_issues_and_challenges(text, session_id: str = None):
    """Extract comprehensive analysis of the case including executive summary, contentions, and legal reasoning."""
    if session_id:
        update_progress(session_id, "primary-issues", "active", "Analyzing legal document to identify key disputes and court findings...")
        time.sleep(0.5)  # Give frontend time to catch active state
    
    prompt = (
        "Conduct a comprehensive analysis of this legal case and provide a structured response covering the following aspects:\n\n"
        
        "## 1. EXECUTIVE SUMMARY OF THE CASE\n"
        "- What were the main findings of the court?\n"
        "- What reasoning did the court offer?\n"
        "- What case law or statutory provisions did the court refer to or apply in coming to its findings?\n\n"
        
        "## 2. MAIN POINTS OF CONTENTION\n"
        "- All the different points the two parties disagreed on in court regarding the matter\n"
        "- What were the different issues on which the court adjudicated the matter?\n"
        "- What were the main points raised by the two parties?\n"
        "- Key words from main points of contention\n\n"
        
        "## 3. PROCEDURAL CHALLENGES\n"
        "**CRITICAL INSTRUCTION: Only include this entire section (including the heading) if there are clear and specific procedural challenges identified in the case.**\n\n"
        "If procedural challenges exist, identify any irregularities of procedure such as:\n"
        "- One party not being heard fairly\n"
        "- Matter being barred by limitation\n"
        "- Improper service of notice\n"
        "- Unsubstantiated claim in the notice\n\n"
        "**MANDATORY: If no clear procedural challenges are found, DO NOT include this section at all. Skip directly to section 4. DO NOT write 'No procedural challenges found' or similar statements.**\n\n"
        
        "## 4. LEGAL REASONING OF THE COURT\n"
        "- Did the reasoning of the court miss a binding precedent? (A binding precedent is a ruling of another matter in a higher court, ideally Supreme Court or High Court, which directly addresses the current legal issue)\n"
        "- Did the court engage with the precedents cited by the parties?\n"
        "- Did the court engage with the statutory provisions accurately?\n\n"
        
        "Provide a detailed, structured analysis focusing on these core areas to understand the complete legal landscape of the case.\n\n"
        f"Case text:\n'''{text}'''"
    )

    @backoff.on_exception(
        backoff.expo,
        Exception,
        max_tries=3,
        max_time=600  # 10 minutes max total time
    )
    def make_openai_call():
        """Make the OpenAI API call with retry logic"""
        return client.chat.completions.create(
            model="o3-mini",
            messages=[
                {"role": "system", "content": "You are a legal analyst specializing in tax law disputes."},
                {"role": "user", "content": prompt}
            ],
            timeout=300,  # 5 minutes per call
            extra_headers={
                "Helicone-Session-Id": session_id or "default",
            }
        )

    try:
        print("🔄 Making OpenAI API call for primary issues extraction...")
        response = make_openai_call()
        result = response.choices[0].message.content
        
        if session_id:
            update_progress(session_id, "primary-issues", "complete", 
                          "Key legal issues and disputes identified", 
                          {"primary_issues": result})
            time.sleep(0.5)  # Give frontend time to catch complete state
        
        print("✅ Primary issues extraction completed successfully")
        return result
    except Exception as e:
        error_msg = f"Error extracting primary issues: {str(e)}"
        print(f"❌ {error_msg}")
        if session_id:
            update_progress(session_id, "primary-issues", "failed", f"Error: {str(e)}")
        return "Unable to extract primary issues due to API timeout or error"

def generate_search_phrases(primary_issues, case_context, session_id: str = None):
    """Generate key phrases for semantic search based on primary issues and case context."""
    if session_id:
        update_progress(session_id, "search-phrases", "active", "Generating optimized search phrases for case law research...")
        time.sleep(0.5)  # Give frontend time to catch active state
    
    prompt = (
    "Given the primary issues or challenges of both parties and the entire case document in the current court, "
    "what should be the key phrases that I should use to semantically search for relevant Supreme Court and High Court cases "
    "from a database of Supreme Court and High Court cases?\n\n"
    "Keep in mind that this is very simple semantics search using Vectara.\n\n"
    "Generate search phrases in the following format:\n"
    "{\n"
    "  \"HC Phrases\": [\"phrase1\", \"phrase2\", \"phrase3\"],\n"
    "  \"SC phrases\": [\"phrase1\", \"phrase2\", \"phrase3\"]\n"
    "}\n\n"
    "Rank the HC phrases and SC phrases in your intrisinc ranking on how relevant they are to our case.\n\n"
    f"Primary Issues:\n{primary_issues}\n\n"
    f"Case Context:\n{case_context}...\n"  # Limit context to avoid token limits
)

    @backoff.on_exception(
        backoff.expo,
        Exception,
        max_tries=3,
        max_time=600  # 10 minutes max total time
    )
    def make_openai_call():
        """Make the OpenAI API call with retry logic"""
        return client.chat.completions.create(
            model="o3-mini",
            messages=[
                {"role": "system", "content": "You are a legal research expert specializing in case law search strategies."},
                {"role": "user", "content": prompt}
            ],
            timeout=300,  # 5 minutes per call
            extra_headers={
                "Helicone-Session-Id": session_id or "default",
            }
        )

    try: 
        print("🔄 Making OpenAI API call for search phrase generation...")
        response = make_openai_call()
        result = json.loads(response.choices[0].message.content)
        
        if session_id:
            hc_count = len(result.get("HC Phrases", []))
            sc_count = len(result.get("SC phrases", []))
            update_progress(session_id, "search-phrases", "complete", 
                          f"Generated {hc_count} HC phrases and {sc_count} SC phrases", 
                          {"search_phrases": result})
            time.sleep(0.5)  # Give frontend time to catch complete state
        
        print("✅ Search phrase generation completed successfully")
        return result
    except Exception as e:
        error_msg = f"Error generating search phrases: {str(e)}"
        print(f"❌ {error_msg}")
        if session_id:
            update_progress(session_id, "search-phrases", "failed", f"Error: {str(e)}")
        return {"HC Phrases": [], "SC phrases": []}

# 🔍 Vectara Search and Document Retrieval Functions
import requests

# Environment variables for Vectara
VECTARA_API_KEY = os.getenv("VECTARA_API_KEY")
SC_CORPUS_KEY = "SC_Judgements"
HC_CORPUS_KEY = "hc-judgements"

def search_vectara_corpus(query, corpus_key, top_k=5):
    """
    Search specific Vectara corpus (SC or HC) for relevant results.
    Returns document IDs and snippets for further processing.
    """
    url = "https://api.vectara.io/v2/query"
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "x-api-key": VECTARA_API_KEY
    }
    
    payload = {
        "query": query,
        "search": {
            "corpora": [
                {
                    "corpus_key": corpus_key,
                    "metadata_filter": "",
                    "lexical_interpolation": 0.005,
                    "custom_dimensions": {}
                }
            ],
            "offset": 0,
            "limit": top_k,
            "context_configuration": {
                "sentences_before": 2,
                "sentences_after": 2,
                "start_tag": "%START_SNIPPET%",
                "end_tag": "%END_SNIPPET%"
            },
            "reranker": {
                "type": "customer_reranker",
                "reranker_id": "rnk_272725719"
            }
        },
        "generation": {
            "max_used_search_results": top_k,
            "response_language": "eng",
            "enable_factual_consistency_score": True
        },
        "save_history": True
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        results = response.json()
        
        # Extract document IDs and relevant info
        documents = []
        for hit in results.get('search_results', []):
            documents.append({
                'document_id': hit.get('document_id'),
                'corpus_key': corpus_key,
                'text_snippet': hit.get('text', ''),
                'score': hit.get('score', 0)
            })
            print(hit.get('document_id'))
        
        return documents
        
    except requests.exceptions.RequestException as e:
        print(f"Search error for {corpus_key}:", str(e))
        return []

def retrieve_full_document(corpus_key, document_id):
    """
    Retrieve complete document from Vectara using document ID.
    """
    url = f"https://api.vectara.io/v2/corpora/{corpus_key}/documents/{document_id}"
    headers = {
        'Accept': 'application/json',
        'x-api-key': VECTARA_API_KEY
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        document_data = response.json()
        
        # Extract full text from document sections
        full_text = ""
        for part in document_data['parts']:
            full_text += part.get('text', '') + "\n\n"
        
        return {
            'document_id': document_id,
            'corpus_key': corpus_key,
            'title': document_data.get('title', ''),
            'full_text': full_text.strip(),
            'metadata': document_data.get('metadata', {})
        }
        
    except requests.exceptions.RequestException as e:
        print(f"Document retrieval error for {document_id}:", str(e))
        return None

def search_corpus_concurrent(phrases: List[str], corpus_key: str, court_type: str, session_id: str = None) -> List[dict]:
    """Search a corpus with multiple phrases concurrently"""
    # Use consistent step names that match the frontend
    step_name = 'high court-search' if court_type == 'High Court' else 'supreme court-search'
    
    if session_id:
        update_progress(session_id, step_name, "active", 
                       f"Searching {court_type} database with {len(phrases)} phrases...")
        time.sleep(0.5)  # Give frontend time to catch active state
    
    all_docs = []
    docs_found = 0
    
    def search_and_retrieve(phrase):
        """Search for a phrase and retrieve full documents"""
        search_results = search_vectara_corpus(phrase, corpus_key, top_k=3)
        phrase_docs = []
        
        for result in search_results:
            full_doc = retrieve_full_document(result['corpus_key'], result['document_id'])
            if full_doc:
                full_doc['search_phrase'] = phrase
                full_doc['court_type'] = court_type
                full_doc['relevance_score'] = result['score']
                phrase_docs.append(full_doc)
        
        return phrase_docs
    
    # Use ThreadPoolExecutor for concurrent searches
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(phrases), 5)) as executor:
        # Submit all search tasks
        future_to_phrase = {
            executor.submit(search_and_retrieve, phrase): phrase 
            for phrase in phrases[:3]  # Limit to top 3 phrases
        }
        
        # Collect results as they complete
        for future in concurrent.futures.as_completed(future_to_phrase):
            phrase = future_to_phrase[future]
            try:
                phrase_docs = future.result()
                all_docs.extend(phrase_docs)
                docs_found += len(phrase_docs)
                
                if session_id:
                    # Update progress as each phrase completes
                    update_progress(session_id, step_name, "active", 
                                  f"Found {docs_found} cases so far...")
                    time.sleep(0.3)  # Small delay for progress updates
                    
            except Exception as e:
                print(f"Error searching phrase '{phrase}': {e}")
    
    if session_id:
        update_progress(session_id, step_name, "complete", 
                       f"Found {docs_found} relevant {court_type} cases")
        time.sleep(0.5)  # Give frontend time to catch complete state
    
    return all_docs

print("✅ Vectara search and retrieval functions loaded!")

# 🎯 Chain of Thought Display Function
def display_analysis_progress(step_name, status="In Progress"):
    """Display the current step in the analysis chain of thought."""
    status_icons = {
        "In Progress": "🔄",
        "Complete": "✅", 
        "Failed": "❌"
    }
    
    icon = status_icons.get(status, "🔄")
    print(f"{icon} {step_name}")

def display_chain_of_thought():
    """Display the complete chain of thought for the analysis process."""
    print("\n" + "="*80)
    print("🧠 TAX COPILOT ANALYSIS CHAIN OF THOUGHT")
    print("="*80)
    print("Following steps will be executed:")
    print("1. 📄 PDF Text Extraction")
    print("2. 🧠 Primary Issues Identification") 
    print("3. 🔍 Search Phrase Generation")
    print("4. 🏛️ High Court Cases Search")
    print("5. ⚖️ Supreme Court Cases Search")
    print("6. 📋 Document Retrieval & Processing (with Legal NER)")
    print("7. 🎯 Comprehensive Legal Analysis Generation")
    print("8. 💾 Save Analysis to File")
    print("="*80 + "\n")

# 🚀 Enhanced Tax Copilot Flow Implementation

def execute_comprehensive_tax_analysis_flow(pdf_path, session_id: str = None):
    """
    Execute the complete tax copilot flow with comprehensive analysis.
    This is the only analysis option - no quick/legacy analysis available.
    """
    if not session_id:
        session_id = str(uuid.uuid4())
    
    # Display chain of thought first
    display_chain_of_thought()
    
    pdf_labeling_executor = ThreadPoolExecutor(max_workers=1)
    pdf_labeling_future = pdf_labeling_executor.submit(
        pdf_labeler.label_pdf_parallel, pdf_path, session_id
    )
    print("🏷️ PDF labeling started in parallel...")
    
    # Continue with rest of the analysis...
    # Step 1: PDF Extraction
    print(f"🔄 Starting Step 1: PDF Extraction")
    update_progress(session_id, "pdf-extraction", "active", "Extracting text from PDF document...")
    time.sleep(1)

    # Step 1: PDF Extraction
    print(f"🔄 Starting Step 1: PDF Extraction")
    update_progress(session_id, "pdf-extraction", "active", "Extracting text from PDF document...")
    time.sleep(1)  # Give frontend time to catch the "active" state
    
    raw_text = extract_full_text_from_pdf(pdf_path)
    
    if not raw_text:
        update_progress(session_id, "pdf-extraction", "failed", "Failed to extract text from PDF")
        print("❌ Failed to extract text from PDF")
        return None
    
    update_progress(session_id, "pdf-extraction", "complete", 
                   f"Extracted {len(raw_text)} characters from PDF document")
    print(f"✅ Step 1 Complete: PDF Extraction")
    time.sleep(1)  # Give frontend time to catch the "complete" state
    
    # Step 2: Extract Primary Issues (with progress tracking)
    print(f"🔄 Starting Step 2: Primary Issues Identification")
    primary_issues = extract_primary_issues_and_challenges(raw_text, session_id)
    print(f"✅ Step 2 Complete: Primary Issues Identification")
    time.sleep(1)
    
    # Step 3: Generate Search Phrases (with progress tracking)
    print(f"🔄 Starting Step 3: Search Phrase Generation")
    search_phrases = generate_search_phrases(primary_issues, raw_text, session_id)
    hc_phrases = search_phrases.get("HC Phrases", [])
    sc_phrases = search_phrases.get("SC phrases", [])
    print(f"✅ Step 3 Complete: Search Phrase Generation")
    time.sleep(1)
    
    # Step 4 & 5: Concurrent Search of HC and SC cases
    print(f"🔄 Starting Steps 4 & 5: Court Case Searches")
    all_relevant_docs = []
    
    # Use ThreadPoolExecutor for concurrent HC and SC searches
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        # Submit both search tasks concurrently
        hc_future = executor.submit(search_corpus_concurrent, hc_phrases, HC_CORPUS_KEY, "High Court", session_id)
        sc_future = executor.submit(search_corpus_concurrent, sc_phrases, SC_CORPUS_KEY, "Supreme Court", session_id)
        
        # Wait for both to complete
        hc_docs = hc_future.result()
        sc_docs = sc_future.result()
        
        all_relevant_docs.extend(hc_docs)
        all_relevant_docs.extend(sc_docs)
    
    print(f"✅ Steps 4 & 5 Complete: Court Case Searches")
    time.sleep(1)
    
    # Step 6: Process and deduplicate documents WITH LEGAL NER
    print(f"🔄 Starting Step 6: Document Processing with Legal NER")
    update_progress(session_id, "document-processing", "active", "Processing documents with Legal NER for better case names...")
    time.sleep(1)
    
    # Apply Legal NER to extract proper case names
    print("🧠 Applying Legal NER to extract case names...")
    for doc in all_relevant_docs:
        if doc.get('full_text'):
            # Extract case name using Legal NER
            extracted_case_name = legal_ner.extract_case_name_from_text(doc['full_text'])
            doc['case_name'] = extracted_case_name
            print(f"   📝 Extracted: {extracted_case_name}")
    
    # Remove duplicates based on document_id
    unique_docs = {}
    for doc in all_relevant_docs:
        doc_id = doc['document_id']
        if doc_id not in unique_docs or doc['relevance_score'] > unique_docs[doc_id]['relevance_score']:
            unique_docs[doc_id] = doc
    
    final_docs = list(unique_docs.values())
    
    # Sort by relevance score
    final_docs.sort(key=lambda x: x['relevance_score'], reverse=True)
    
    sc_count = len([d for d in final_docs if d['court_type'] == 'Supreme Court'])
    hc_count = len([d for d in final_docs if d['court_type'] == 'High Court'])
    
    # In Step 6, after processing documents:
    update_progress(session_id, "document-processing", "complete", 
                f"Processed {len(final_docs)} unique cases ({sc_count} SC, {hc_count} HC) with Legal NER",
                {"processed_documents": [
                    {
                        "document_id": doc["document_id"],
                        "case_name": doc.get("case_name", "Unknown Case"),
                        "court_type": doc["court_type"],
                        "relevance_score": doc["relevance_score"],
                        "search_phrase": doc.get("search_phrase", ""),
                        "full_text": doc.get("full_text", "")  # Limit text for performance
                    }
                    for doc in final_docs[:10]  # Send top 10 documents
                ]})
    print(f"✅ Step 6 Complete: Document Processing with Legal NER")
    time.sleep(1)
    
    # Step 7: Generate comprehensive final analysis
    print(f"🔄 Starting Step 7: Legal Analysis Generation")
    update_progress(session_id, "legal-analysis", "active", "Generating comprehensive legal analysis with citations...")
    time.sleep(1)
    
    final_analysis = generate_comprehensive_legal_analysis(
        raw_case_text=raw_text,
        primary_issues=primary_issues, 
        relevant_documents=final_docs,
        session_id=session_id
    )
    # In execute_comprehensive_tax_analysis_flow, after generating analysis:
    store_analysis_result(
        session_id,
        os.path.basename(pdf_path),
        primary_issues,
        final_analysis
    )
    
    # Debug the analysis before sending
    print(f"🔍 BACKEND DEBUG: Analysis type: {type(final_analysis)}")
    print(f"🔍 BACKEND DEBUG: Analysis length: {len(final_analysis) if final_analysis else 0}")
    # print(f"🔍 BACKEND DEBUG: Analysis preview: {final_analysis[:200] if final_analysis else 'EMPTY'}")
    
    update_progress(session_id, "legal-analysis", "complete", 
                   "Generated comprehensive legal analysis with citations",
                   {"final_comprehensive_analysis": final_analysis})
    print(f"✅ Step 7 Complete: Legal Analysis Generation")
    print(f"🔍 BACKEND DEBUG: Sent analysis data via update_progress")
    time.sleep(1)
    
    # Prepare complete results
    complete_results = {
        "session_id": session_id,
        "input_pdf_path": pdf_path,
        "raw_text": raw_text,
        "primary_issues": primary_issues,
        "search_phrases": search_phrases,
        "relevant_documents": final_docs,
        "final_comprehensive_analysis": final_analysis,
        "analysis_metadata": {
            "total_documents_analyzed": len(final_docs),
            "sc_documents": sc_count,
            "hc_documents": hc_count,
            "model_used": "o3-mini",
            "legal_ner_enabled": True
        }
    }
    
    # Step 8: Save comprehensive analysis to file
    print(f"🔄 Starting Step 8: Saving Analysis to File")
    output_file = save_comprehensive_analysis_to_file(complete_results, pdf_path)
    if output_file:
        print(f"✅ Step 8 Complete: Analysis saved to {output_file}")
    else:
        print("⚠️  Step 8 Warning: Could not save analysis to file")
    time.sleep(1)
    
    # Display final analysis
    print("\n" + "="*100)
    print("🏆 COMPREHENSIVE LEGAL ANALYSIS")
    print("="*100)
    print(final_analysis)
    print("="*100)
    
    return complete_results

# 🎯 Final Analysis & Prediction Function - UPDATED to use Legal NER case names

def save_document_excerpts_to_file(relevant_documents, session_id: str = None):
    """
    Extract first 200 words from each relevant document and save to a separate file.
    """

    return None
    if not relevant_documents:
        return None
    
    filename = f"document_excerpts_{session_id or 'analysis'}.txt"
    
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("DOCUMENT EXCERPTS - FIRST 200 WORDS FROM EACH RELEVANT CASE\n")
            f.write("=" * 80 + "\n\n")
            
            for i, doc in enumerate(relevant_documents, 1):
                # Use Legal NER extracted case name
                case_name = doc.get('case_name', 'Case name not found')
                
                # Get first 200 words from full text
                full_text = doc.get('full_text', '')
                words = full_text.split()
                first_200_words = ' '.join(words[:200])
                
                # Write to file
                f.write(f"DOCUMENT {i}: {case_name}\n")
                f.write(f"COURT: {doc.get('court_type', 'Unknown Court')}\n")
                f.write(f"DOCUMENT ID: {doc.get('document_id', 'Unknown')}\n")
                f.write(f"RELEVANCE SCORE: {doc.get('relevance_score', 'N/A')}\n")
                f.write(f"SEARCH PHRASE: {doc.get('search_phrase', 'N/A')}\n")
                f.write("-" * 40 + "\n")
                f.write("FIRST 200 WORDS:\n")
                f.write(first_200_words)
                f.write("\n" + "=" * 80 + "\n\n")
        
        print(f"✅ Document excerpts saved to: {filename}")
        return filename
        
    except Exception as e:
        print(f"❌ Error saving document excerpts: {str(e)}")
        return None

def save_comprehensive_analysis_to_file(complete_results, pdf_path):
    """
    Save the comprehensive analysis output to a file named after the input PDF.
    Includes only: primary issues, main points of contention, search phrases, and comprehensive legal analysis.
    """
    import os
    
    try:
        # Generate filename from PDF path
        base_name = os.path.splitext(os.path.basename(pdf_path))[0]
        output_filename = f"{base_name}.txt"
        
        with open(output_filename, 'w', encoding='utf-8') as f:
            # Header
            f.write("=" * 100 + "\n")
            f.write("TAX COPILOT - LEGAL ANALYSIS REPORT (with Legal NER)\n")
            f.write("=" * 100 + "\n")
            f.write(f"Input PDF: {os.path.basename(pdf_path)}\n")
            f.write(f"Analysis Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Legal NER Model: en_legal_ner_trf\n")
            f.write("=" * 100 + "\n\n")
            
            # Primary issues section
            f.write("📋 PRIMARY ISSUES IDENTIFIED\n")
            f.write("=" * 80 + "\n")
            f.write(complete_results['primary_issues'])
            f.write("\n\n")
            
            # Search phrases section
            f.write("🔍 SEARCH PHRASES GENERATED\n")
            f.write("=" * 80 + "\n")
            
            hc_phrases = complete_results['search_phrases'].get('HC Phrases', [])
            sc_phrases = complete_results['search_phrases'].get('SC phrases', [])
            
            f.write("HIGH COURT SEARCH PHRASES:\n")
            f.write("-" * 40 + "\n")
            for i, phrase in enumerate(hc_phrases, 1):
                f.write(f"{i}. {phrase}\n")
            
            f.write("\nSUPREME COURT SEARCH PHRASES:\n")
            f.write("-" * 40 + "\n")
            for i, phrase in enumerate(sc_phrases, 1):
                f.write(f"{i}. {phrase}\n")
            f.write("\n")
            
            # Final comprehensive analysis
            f.write("🏆 COMPREHENSIVE LEGAL ANALYSIS\n")
            f.write("=" * 100 + "\n")
            f.write(complete_results['final_comprehensive_analysis'])
            f.write("\n\n")
            
            # Footer
            f.write("=" * 100 + "\n")
            f.write("DISCLAIMER: This analysis serves informational purposes only and does not constitute legal advice.\n")
            f.write("Final appellate strategy requires comprehensive case law research and consultation with specialized tax counsel.\n")
            f.write("Legal NER powered by en_legal_ner_trf model for enhanced case name extraction.\n")
            f.write("=" * 100 + "\n")
        
        print(f"✅ Comprehensive analysis saved to: {output_filename}")
        return output_filename
        
    except Exception as e:
        print(f"❌ Error saving comprehensive analysis: {str(e)}")
        return None

def search_case_url_indiankanoon(case_name: str) -> Optional[str]:
    """Search for exact case URL on IndianKanoon"""
    try:
        time.sleep(0.5)  # Avoid rate limiting
        
        # Clean case name
        search_query = case_name.strip()
        encoded_query = urllib.parse.quote(search_query)
        
        # Search on IndianKanoon
        search_url = f"https://indiankanoon.org/search/?formInput={encoded_query}"
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        
        response = requests.get(search_url, headers=headers, timeout=10)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Get first result
            result_div = soup.find('div', {'class': 'result'})
            if result_div:
                title_link = result_div.find('a', {'class': 'result_title'})
                if title_link and title_link.get('href'):
                    return f"https://indiankanoon.org{title_link['href']}"
        
        return None
    except Exception as e:
        print(f"Error searching case: {e}")
        return None

def extract_case_names_from_analysis(analysis_text: str, session_id: str = None) -> List[str]:
    """Extract case names from analysis using OpenAI"""
    prompt = """Extract all case names mentioned in the following legal analysis text.
    
Return ONLY a JSON list of case names in "X vs Y" format.
Do not include years, court names, or any other information.
Just the party names in "X vs Y" format.

Example output:
["CIT vs ABC Ltd", "Commissioner vs XYZ Industries", "M/s GM Alloys Pvt Ltd vs Union of India"]

Text to analyze:
""" + analysis_text

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Use faster model for extraction
            messages=[
                {"role": "system", "content": "You are a legal text analyzer. Extract case names only."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            timeout=30,
            extra_headers={
                "Helicone-Session-Id": session_id or "default",
            }
        )
        
        case_names_json = response.choices[0].message.content.strip()
        # Remove markdown code blocks if present
        case_names_json = re.sub(r'```json\s*|\s*```', '', case_names_json)
        
        case_names = json.loads(case_names_json)
        return case_names
        
    except Exception as e:
        print(f"Error extracting case names: {e}")
        return []

def search_and_replace_case_names(text: str, case_names: List[str]) -> str:
    """Search for URLs and replace case names with clickable links"""
    
    print("🔍 Searching for case URLs...")
    for case_name in case_names:
        url = search_case_url_ddg_library(case_name)
        if url:
            # Simple text replacement
            link = f"[{case_name}]({url})"
            print(text.find(case_name))
            text = text.replace(case_name, link)
            print(f"   ✅ Replaced: {case_name} -> {url}")
        else:
            print(f"   ❌ Not found: {case_name}")
    
    return text

def create_case_name_links_mapping(case_names: List[str]) -> Dict[str, str]:
    """
    Create a JSON mapping of case names to their linked versions.
    Returns: {case_name: "case_name(link)"}
    """
    
    print("🔍 Creating case name to link mapping...")
    case_links_mapping = {}
    
    for case_name in case_names:
        url = search_case_url_ddg_library(case_name)
        if url:
            # Create the linked version
            linked_case = f"[{case_name}]({url})"
            case_links_mapping[case_name] = linked_case
            print(f"   ✅ Mapped: {case_name} -> {url}")
        else:
            # Keep original if no URL found
            case_links_mapping[case_name] = case_name
            print(f"   ❌ No URL found for: {case_name}")
    
    return case_links_mapping

def create_case_name_links_mapping_v2(case_names: List[str]) -> Dict[str, str]:
    """
    Create a JSON mapping of case names to their linked versions using HTML anchor tags.
    Returns: {case_name: "<a href='url' target='_blank' rel='noopener noreferrer'>case_name</a>"}
    """
    
    print("🔍 Creating case name to link mapping...")
    case_links_mapping = {}
    
    for case_name in case_names:
        url = search_case_url_ddg_library(case_name)
        if url:
            # Create the HTML anchor tag with target="_blank" for React
            linked_case = f'<a href="{url}" target="_blank" rel="noopener noreferrer">{case_name}</a>'
            case_links_mapping[case_name] = linked_case
            print(f"   ✅ Mapped: {case_name} -> {url}")
        else:
            # Keep original if no URL found
            case_links_mapping[case_name] = case_name
            print(f"   ❌ No URL found for: {case_name}")
    
    return case_links_mapping


def replace_case_names_with_llm(text: str, case_links_mapping: Dict[str, str], session_id: str = None) -> str:
    """
    Use LLM to intelligently replace case names with their linked versions
    while preserving the document format.
    """
    
    # Create a clear instruction for the LLM
    prompt = f"""You are a legal document formatter. Your task is to replace case names in the text with their linked versions while preserving the exact format and structure of the document.

IMPORTANT RULES:
1. ONLY replace exact matches of case names
2. Preserve ALL formatting, structure, headings, bullet points, etc.
3. Do NOT change any other content
4. Replace case names case-sensitively
5. If a case name appears multiple times, replace ALL occurrences
6. Its of utmost importance that you replace the case names with the linked versions.
7. Throuughout the analysis, you should use the linked versions of the case names.
8. If no case name is present, then do nothing, else without fail replace them, entirely non negotiable

CASE NAME MAPPINGS:
{json.dumps(case_links_mapping, indent=2)}

DOCUMENT TO PROCESS:
{text}

OUTPUT INSTRUCTIONS:
Return the document with ONLY the case names replaced according to the mappings, replace them without fail, non negotiable. Keep everything else exactly the same."""

    try:
        print("🤖 Using LLM to intelligently replace case names...")
        response = client.chat.completions.create(
            model="o3-mini",  # or your preferred model
            messages=[
                {"role": "system", "content": "You are a precise document formatter that replaces text according to given mappings while preserving all formatting."},
                {"role": "user", "content": prompt}
            ],
            timeout=60,
            extra_headers={
                "Helicone-Session-Id": session_id or "default",
            }
        )
        
        result = response.choices[0].message.content
        print("✅ LLM replacement completed successfully")
        return result
        
    except Exception as e:
        print(f"❌ Error in LLM replacement: {str(e)}")
        # Fallback to original text if LLM fails
        return text


def generate_comprehensive_legal_analysis(raw_case_text, primary_issues, relevant_documents, session_id: str = None):
    """
    Generate final comprehensive legal analysis combining all inputs.
    This is the final step after concatenating all components.
    UPDATED: Now uses LLM for intelligent case name replacement.
    """
    
    # Save document excerpts to separate file
    save_document_excerpts_to_file(relevant_documents, session_id)
    
    # Prepare context from relevant documents - UPDATED with Legal NER case names
    context_blocks = []
    for doc in relevant_documents[:10]:  # Use top 10 most relevant documents
        # Use Legal NER extracted case name instead of regex
        case_name = doc.get('case_name', 'Case name not found')
        print(f"📋 Using Legal NER case name: {case_name}")
        
        case_summary = f"""
CASE: {case_name}
COURT: {doc.get('court_type', 'Unknown Court')}
RELEVANCE SCORE: {doc.get('relevance_score', 'N/A')}

FULL TEXT:
{doc.get('full_text', '')}
---
"""
        context_blocks.append(case_summary)
       
    context = "\n".join(context_blocks)
    
    # Save context to file for debugging
    try:
        with open(f"context_debug_{session_id or 'analysis'}.txt", 'w', encoding='utf-8') as f:
            f.write("CONTEXT VARIABLE DEBUG OUTPUT (with Legal NER)\n")
            f.write("=" * 80 + "\n")
            f.write(f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Number of context blocks: {len(context_blocks)}\n")
            f.write(f"Total context length: {len(context)} characters\n")
            f.write(f"Legal NER Model: en_legal_ner_trf\n")
            f.write("=" * 80 + "\n\n")
            f.write(context)
        print(f"✅ Context (with Legal NER case names) saved to: context_debug_{session_id or 'analysis'}.txt")
    except Exception as e:
        print(f"❌ Error saving context to file: {str(e)}")
    
    prompt = (
    "As a senior legal expert specializing in Indian tax jurisprudence, conduct a comprehensive Supreme Court appeal prediction analysis for the provided judgment.\n\n"
    
    "## CRITICAL PRELIMINARY ASSESSMENT - TAX DEPARTMENT STATUS\n"
    "**MANDATORY FIRST STEP: Before any analysis, determine and output as the first line of the analysis the current status of the tax department in this case: Clearly Write wether the Initial Judgement was in favour of Tax Department or Not, Initial Word should be there and it should be just single sentence**\n\n"
    
    "1. **If Tax Department WON this case (judgment in favor of Revenue/CIT/Tax Officer):**\n"
    "   - STOP ANALYSIS IMMEDIATELY\n"
    "   - Return ONLY: 'Tax Department has already won this case. No appeal analysis required.'\n\n"
    
    "2. **If Non-Tax Department's appeal was DECLINED/DISMISSED:**\n"
    "   - STOP ANALYSIS IMMEDIATELY\n"
    "   - Return ONLY: 'Non-tax department's appeal was declined. Tax Department's position upheld. No further analysis required.'\n\n"
    
    "3. **If Tax Department LOST this case (judgment against Revenue/CIT/Tax Officer):**\n"
    "   - PROCEED with full analysis below to predict if Tax Department's appeal to higher court will succeed\n"
    "   - Analyze whether Tax Department should appeal to the next higher court\n"
    "   - If this is already a Supreme Court judgment against Tax Department, state: 'Supreme Court has given final judgment against Tax Department. No further appeal possible.'\n\n"
    
    "4. **If Tax Department's appeal was REJECTED at current level:**\n"
    "   - PROCEED with analysis for appeal to next higher court\n"
    "   - If High Court rejected Tax Department's appeal → Analyze Supreme Court appeal prospects\n"
    "   - If Supreme Court rejected → State: 'Supreme Court has rejected Tax Department's appeal. No further appeal possible.'\n\n"
    
    "## CASE CONTEXT & PARTIES\n"
    "**After confirming Tax Department lost/appeal rejected, note that the appellant for higher court appeal will be TAX OFFICER (Revenue Department). This impacts analysis as:**\n"
    "- Revenue appeals typically involve statutory interpretation and procedural compliance issues\n"
    "- Tax officers have specific statutory duties and limited grounds for appeal\n"
    "- Supreme Court scrutiny of revenue appeals follows established precedential patterns\n"
    "- Consider departmental appeal limitations and public revenue implications\n\n"
    
    "## PRIMARY ISSUES IN DISPUTE\n"
    f"{primary_issues}\n\n"
    
    "## CITATION REQUIREMENTS - MINIMAL AND STRATEGIC\n"
    "**CRITICAL: Use only 2-3 total citations throughout the entire analysis at the most relevant strategic points.**\n\n"
    
    "**Citation Sources (IN ORDER OF PRIORITY):**\n"
    "1. **PRIMARY SOURCE:** Cases from semantic search results (provided in context section below) - USE THESE FIRST\n"
    "2. **SECONDARY SOURCE:** Cases mentioned in the input judgment (raw_text) - USE ONLY if semantic search results don't provide adequate support\n"
    "3. **Format:** Always use 'X vs Y' format (e.g., 'CIT vs ABC Ltd.', 'Commissioner vs XYZ Industries')\n\n"
    
    "**Strategic Citation Placement (Maximum 2-3 total):**\n"
    "1. One citation when establishing the key legal principle central to the appeal\n"
    "2. One citation when making the strongest argument for Tax Department's position\n"
    "3. One optional citation if there's a directly contradictory precedent that needs distinguishing\n\n"
    
    "**Citation Priority Rules:**\n"
    "- ALWAYS check semantic search results FIRST before considering input judgment cases\n"
    "- Only use input judgment cases if semantic results lack relevant precedents\n"
    "- If using input judgment case, it must be directly relevant to the point being made\n\n"
    
    "**Strictly Prohibited:**\n"
    "- NO citation in every paragraph or section\n"
    "- NO repetitive use of the same case\n"
    "- NO generic references like 'as held by the Supreme Court'\n"
    "- NO over-citation that disrupts readability\n"
    "- MAXIMUM 2-3 citations in the entire analysis\n\n"
    
    "## REQUIRED OUTPUT STRUCTURE (ONLY IF TAX DEPARTMENT LOST/NEEDS TO APPEAL)\n\n"
    
    "### 1. **APPEAL PREDICTION FOR TAX DEPARTMENT**\n"
    "State ONE outcome only (verbatim):\n"
    "- **Tax Department's Appeal Likely to be Accepted** (High likelihood of higher court accepting Tax Department's appeal)\n"
    "- **Tax Department's Appeal Likely to be Declined** (High likelihood of higher court declining Tax Department's appeal)\n\n"
    
    "### 2. **Heres why**\n"
    "(Maximum 4 strategic bullet points - NO citations in summary)\n"
    "- Primary legal ground supporting Tax Department's appeal prospects\n"
    "- Key statutory interpretation favorable to Tax Department\n"
    "- Procedural or substantive error in lower court that favors Tax Department\n"
    "- Strategic recommendation for Tax Department's appeal\n\n"
    
    "### 3. **GOVERNING LEGAL FRAMEWORK**\n"
    "**Primary Statutes:**\n"
    "- [Cite specific sections with complete details, e.g., 'Income Tax Act, 1961: Sections 68, 69A']\n"
    "- [Include only directly applicable provisions]\n\n"
    
    "**Subordinate Legislation:**\n"
    "- [Cite applicable rules with details, e.g., 'Income Tax Rules, 1962: Rule 8D']\n"
    "- [Include notifications only if directly relevant]\n\n"
    
    "**Constitutional Provisions:** (if applicable)\n"
    "- [e.g., 'Constitution of India: Article 265']\n\n"
    
    "*If no directly applicable statutory provisions exist, state: 'No directly applicable statutory provisions identified.'*\n\n"
    
    "### 4. **COMPREHENSIVE LEGAL ANALYSIS FOR TAX DEPARTMENT'S APPEAL**\n\n"
    
    "#### A. **Errors in Lower Court Judgment**\n"
    "Identify specific errors of law or fact that favor Tax Department's appeal. Assess whether the lower court misinterpreted statutory provisions or overlooked binding precedents favorable to Revenue. Examine if factual findings lack evidentiary support. Focus on grounds that typically succeed in Revenue appeals.\n\n"
    
    "#### B. **Tax Department's Procedural Compliance**\n"
    "Confirm Tax Department followed all mandatory procedures including proper notice issuance, adherence to limitation periods, and compliance with assessment procedures. Identify any procedural lapses by the assessee that strengthen Tax Department's case. Emphasize Revenue's statutory compliance.\n\n"
    
    "#### C. **Substantive Legal Merit for Tax Department**\n"
    "- **Pro-Revenue Statutory Interpretation:** Apply interpretations that favor revenue collection and prevent tax avoidance. Consider anti-avoidance provisions and beneficial construction for Revenue. Highlight interpretations that protect public revenue. [PLACE FIRST STRATEGIC CITATION HERE IF NEEDED]\n"
    "- **Analysis of Precedents:** Examine available precedents critically. If precedents favor assessee, distinguish them based on factual differences or argue for narrow interpretation. If no favorable Revenue precedents exist, argue for first principles and statutory intent.\n\n"
    
    "#### D. **Analysis of Available Precedents**\n"
    "**PRIMARY CITATION SOURCE - Semantic Search Results with Legal NER Case Names (USE THESE FIRST):**\n\n"
    f"{context}\n\n"
    
    "**Precedent Strategy:**\n"
    "- FIRST: Look for relevant cases from semantic search results above (case names extracted using Legal NER)\n"
    "- If cases support Tax Department: Emphasize factual similarities and binding nature\n"
    "- If cases support assessee: Distinguish based on facts, argue narrow application, or highlight subsequent developments\n"
    "- If mixed precedents: Focus on more recent or higher authority favorable to Revenue\n"
    "- If no relevant cases in semantic results: ONLY THEN consider cases from input judgment [PLACE STRATEGIC CITATION HERE IF NEEDED]\n"
    "- If no favorable precedents at all: Argue based on first principles, statutory language, and public policy\n\n"
    
    "**Note:** Always prioritize semantic search results for citations. The absence of favorable precedents does not preclude successful appeal if strong statutory or policy arguments exist.\n\n"
    
    "#### E. **Tax Department's Appeal Viability**\n"
    "Assess whether substantial questions of law favor Revenue's interpretation. Consider public revenue implications and tax avoidance prevention aspects. Evaluate if the case sets dangerous precedent for tax collection. Focus on grounds that typically succeed in Revenue SLPs. [PLACE THIRD STRATEGIC CITATION HERE IF ABSOLUTELY NECESSARY]\n\n"
    
    "#### F. **Strategic Recommendations for Tax Department**\n"
    "Identify strongest grounds for Tax Department's appeal based on successful Revenue strategies. Suggest emphasis on statutory interpretation favorable to Revenue. Recommend highlighting public interest and revenue protection angles. Focus on arguments that resonate with higher courts in tax matters.\n\n"
    
    "## QUALITY ASSURANCE FOR TAX DEPARTMENT ANALYSIS\n\n"
    
    "### Pre-Analysis Verification:\n"
    "- ✓ CONFIRMED Tax Department lost/needs to appeal before proceeding\n"
    "- ✓ If Tax Department won, returned appropriate message and stopped\n"
    "- ✓ Analysis focuses on Tax Department's appeal prospects\n"
    "- ✓ All recommendations favor Revenue/Tax Department position\n\n"
    
    "### Citation Quality Control:\n"
    "- ✓ MAXIMUM 2-3 citations in entire analysis\n"
    "- ✓ ALWAYS use semantic search results as PRIMARY citation source\n"
    "- ✓ Only use input judgment cases if semantic results lack relevant precedents\n"
    "- ✓ NO citations in summary section\n"
    "- ✓ Each citation must add substantial value, not just fill space\n"
    "- ✓ Verify citation is from semantic results BEFORE using input judgment\n\n"
    
    "### Handling Unfavorable Precedents:\n"
    "- ✓ If all precedents favor assessee, focus on distinguishing facts\n"
    "- ✓ Argue for narrow interpretation of unfavorable precedents\n"
    "- ✓ Emphasize statutory language and legislative intent\n"
    "- ✓ Highlight public policy and revenue protection considerations\n"
    "- ✓ Note that absence of favorable precedents doesn't preclude successful appeal\n\n"
    
    "### Revenue-Focused Standards:\n"
    "- ✓ Interpret statutes to protect public revenue\n"
    "- ✓ Emphasize tax avoidance prevention\n"
    "- ✓ Highlight Revenue's statutory duties\n"
    "- ✓ Focus on precedents favoring Tax Department\n\n"
    
    "**FINAL COMPLIANCE NOTICE: \n"
    "1. First determine if Tax Department won/lost\n"
    "2. If won or non-tax appeal declined, stop immediately\n"
    "3. If Tax Department lost, proceed with analysis\n"
    "4. Use ONLY 2-3 citations maximum - PRIORITIZE semantic search results with Legal NER case names\n"
    "5. Check semantic search results FIRST before considering input judgment cases\n"
    "6. If no favorable precedents exist in semantic results, then check input judgment\n"
    "7. If still no favorable precedents, argue based on statutory interpretation and policy\n"
    "8. Focus on revenue protection and public interest arguments**\n\n"
    
    "## INPUT DATA FOR ANALYSIS\n\n"
    
    "**Input Judgment for Analysis:**\n"
    f"'''{raw_case_text}...'''\n\n"
    
    "**Identified Primary Issues:**\n"
    f"'''{primary_issues}'''\n\n"
    
    "**Available Precedential Authority:**\n"
    f"[{len(relevant_documents)} semantic search results with Legal NER case names - PRIMARY CITATION SOURCE - USE THESE FIRST]\n"
    "[Input judgment cases are SECONDARY source - use only if semantic results lack relevant precedents]\n\n"
    
    "---\n\n"
    
    "**DISCLAIMER:** This analysis serves informational purposes only and does not constitute legal advice. Analysis conducted only if Tax Department lost and needs to appeal. Case names extracted using Legal NER (en_legal_ner_trf) for improved accuracy."
)

    @backoff.on_exception(
        backoff.expo,
        Exception,
        max_tries=3,
        max_time=900  # 15 minutes max total time for the most complex analysis
    )
    def make_openai_call():
        """Make the OpenAI API call with retry logic"""
        return client.chat.completions.create(
            model="o3-mini",
            messages=[
                {"role": "system", "content": "You are a senior legal expert specializing in Indian tax law and Supreme Court appellate practice."},
                {"role": "user", "content": prompt}
            ],
            timeout=600,  # 10 minutes per call for complex analysis
            extra_headers={
                "Helicone-Session-Id": session_id or "default",
            }
        )

    try:
        print("🔄 Making OpenAI API call for comprehensive legal analysis (this may take several minutes)...")
        print(f"📝 Prompt length: {len(prompt)} characters")
        print(f"📚 Processing {len(relevant_documents)} relevant case documents with Legal NER case names")
        
        response = make_openai_call()
        result = response.choices[0].message.content
        
        print("✅ Comprehensive legal analysis completed successfully")
        print(f"📄 Generated analysis length: {len(result)} characters")
        
        # Extract case names from the analysis
        print("🔍 Extracting case names from analysis...")
        case_names = extract_case_names_from_analysis(result, session_id)
        print(f"📋 Found {len(case_names)} case names: {case_names}")
        
        # Create case name to link mapping
        case_links_mapping = create_case_name_links_mapping_v2(case_names)
        
        # Save mapping for debugging
        try:
            with open(f"case_links_mapping_{session_id or 'analysis'}.json", 'w', encoding='utf-8') as f:
                json.dump(case_links_mapping, f, indent=2)
            print(f"✅ Case links mapping saved to: case_links_mapping_{session_id or 'analysis'}.json")
        except Exception as e:
            print(f"⚠️ Warning: Could not save mapping file: {str(e)}")
        
        # Use LLM to intelligently replace case names with links
        result_with_links = replace_case_names_with_llm(result, case_links_mapping, session_id)
        
        return result_with_links
        
    except Exception as e:
        error_msg = f"Error generating legal analysis: {str(e)}"
        print(f"❌ {error_msg}")
        return f"Unable to generate comprehensive legal analysis due to API timeout or error: {str(e)}"
# 🎯 Main Analysis Function (Only Comprehensive Analysis Available)

def run_tax_copilot_analysis(pdf_path, session_id: str = None):
    """
    Main function to run the comprehensive tax copilot analysis.
    
    This is the only analysis option available - comprehensive legal analysis with full citation support.
    No quick analysis or legacy API option available.
    
    Args:
        pdf_path: Path to the PDF file to analyze
        session_id: Optional session ID for progress tracking
    """
    try:
        print("🚀 Starting Comprehensive Tax Copilot Analysis with Legal NER...")
        print("📋 Only comprehensive legal analysis available (no quick analysis option)")
        print("🧠 Legal NER enabled for better case name extraction")
        
        # Run the comprehensive analysis flow
        results = execute_comprehensive_tax_analysis_flow(pdf_path, session_id)
        
        if results:
            print("\n✅ Analysis completed successfully!")
            print(f"📊 Summary:")
            print(f"   - Total documents analyzed: {results['analysis_metadata']['total_documents_analyzed']}")
            print(f"   - Supreme Court cases: {results['analysis_metadata']['sc_documents']}")
            print(f"   - High Court cases: {results['analysis_metadata']['hc_documents']}")
            print(f"   - Legal NER enabled: {results['analysis_metadata']['legal_ner_enabled']}")
            return results
        else:
            print("❌ Analysis failed - no results generated")
            return None
            
    except Exception as e:
        print(f"❌ Error during analysis: {str(e)}")
        return None

# 📋 Updated Usage Instructions
print("🚀 Enhanced Tax Copilot System with Legal NER Ready!")
print("="*60)
print("📝 To use the system:")
print("1. Make sure you have installed: pip install PyMuPDF spacy")
print("2. Install Legal NER model: python -m spacy download en_legal_ner_trf")
print("3. Set your environment variables:")
print("   - VECTARA_API_KEY")
print("   - openai_api_key") 
print("4. Run: results = run_tax_copilot_analysis('your_case_file.pdf')")
print()
print("🔧 System Features:")
print("✅ Advanced PDF text extraction with PyMuPDF")
print("✅ Real-time chain of thought analysis progress")
print("✅ AI-powered primary issues identification")
print("✅ Intelligent search phrase generation")  
print("✅ CONCURRENT SC/HC corpus searching")
print("✅ Full document retrieval (not just snippets)")
print("✅ LEGAL NER INTEGRATION (en_legal_ner_trf) for precise case name extraction")
print("✅ ONLY comprehensive legal analysis (no quick analysis option)")
print("✅ Supreme Court appeal prediction with citations")
print("✅ Risk assessment and strategic recommendations")

print("✅ Enhanced Tax Copilot System with Legal NER loaded!")